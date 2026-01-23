# -*- coding: utf-8 -*-
"""
TriMul (outgoing) forward pass.

Algorithm
---------
1. Layer‑norm the input tensor over the last dimension (`dim`).
2. Cast to float‑16 for the heavy arithmetic.
3. Linear projections (left / right) and three gating projections
   (left_gate, right_gate, out_gate) – all without bias.
4. Apply the (optional) pairwise mask together with the two
   multiplicative gates.  This element‑wise fusion is performed by a
   custom Triton kernel (`mask_gate_kernel`).
5. Compute the N³ multiplicative update
      out[b,i,j,d] = Σₖ left[b,i,k,d] * right[b,j,k,d]
   using torch.einsum (which internally maps to a highly‑optimized
   batched GEMM on the H100).
6. Layer‑norm the intermediate result over the hidden dimension,
   multiply by `out_gate`, and finally linearly project back to `dim`.
7. Cast the result back to float‑32 and return.

The Triton kernel only handles the mask‑gate fusion; the O(N³)
contraction is delegated to PyTorch’s cuBLAS‑backed einsum.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mask_gate_kernel(tensor_ptr, gate_ptr, mask_ptr, out_ptr,
                     total_rows, hidden_dim,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Fuse (tensor * gate * mask) where `tensor` and `gate` have shape
    [total_rows, hidden_dim] and `mask` has shape [total_rows] (a scalar
    per row).  The result is written to `out_ptr` with the same shape.
    """
    pid_m = tl.program_id(0)          # row block
    pid_n = tl.program_id(1)          # column block

    # offsets inside the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # out‑of‑bounds masks
    mask_m = offs_m < total_rows
    mask_n = offs_n < hidden_dim
    valid = mask_m[:, None] & mask_n[None, :]

    # linear offset for each row (row_stride = hidden_dim)
    row_offset = offs_m * hidden_dim

    # Load the required slices (fp16)
    t = tl.load(tensor_ptr + row_offset[:, None] + offs_n[None, :],
                mask=valid, other=0.0)
    g = tl.load(gate_ptr + row_offset[:, None] + offs_n[None, :],
                mask=valid, other=0.0)

    # Load mask scalar per row and broadcast
    m = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0)   # [BLOCK_M]
    m = m[:, None]                                          # broadcast to cols

    # Compute fused result
    out = t * g * m

    # Store result
    tl.store(out_ptr + row_offset[:, None] + offs_n[None, :],
             out, mask=valid)


def mask_gate(tensor: torch.Tensor,
              gate: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """
    Wrapper around `mask_gate_kernel`. All tensors must be contiguous
    and of dtype torch.float16.
    """
    B, N, _, H = tensor.shape
    total_rows = B * N * N
    hidden_dim = H

    # Flatten to 2‑D matrices (row‑major)
    t_flat = tensor.reshape(total_rows, hidden_dim)
    g_flat = gate.reshape(total_rows, hidden_dim)
    m_flat = mask.reshape(total_rows)               # [total_rows]

    out_flat = torch.empty_like(t_flat)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(total_rows, BLOCK_M),
            triton.cdiv(hidden_dim, BLOCK_N))

    mask_gate_kernel[grid](
        t_flat, g_flat, m_flat, out_flat,
        total_rows, hidden_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return out_flat.view(B, N, N, H)


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.

    Parameters
    ----------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor shape [B, N, N, dim]
        - mask         : torch.Tensor shape [B, N, N] (may be all ones)
        - weights      : dict of torch.Tensors containing all model weights
        - config       : dict with keys "dim", "hidden_dim", optionally "nomask"

    Returns
    -------
    torch.Tensor
        Tensor of shape [B, N, N, dim] (float32)
    """
    # ------------------------------------------------------------------
    # Unpack
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    # ------------------------------------------------------------------
    # Input LayerNorm (over `dim`)
    # ------------------------------------------------------------------
    norm_w = weights["norm.weight"].to(input_tensor.dtype)
    norm_b = weights["norm.bias"].to(input_tensor.dtype)
    x = F.layer_norm(input_tensor, (dim,), weight=norm_w, bias=norm_b)

    # Use half precision for the bulk of the computation
    x = x.to(torch.float16)

    # ------------------------------------------------------------------
    # Linear projections (no bias)
    # ------------------------------------------------------------------
    left_w  = weights["left_proj.weight"].to(x.dtype)
    right_w = weights["right_proj.weight"].to(x.dtype)

    left  = F.linear(x, left_w)   # [B, N, N, hidden_dim]
    right = F.linear(x, right_w)

    # ------------------------------------------------------------------
    # Gating projections + sigmoid
    # ------------------------------------------------------------------
    left_gate_w  = weights["left_gate.weight"].to(x.dtype)
    right_gate_w = weights["right_gate.weight"].to(x.dtype)
    out_gate_w   = weights["out_gate.weight"].to(x.dtype)

    left_gate  = torch.sigmoid(F.linear(x, left_gate_w))
    right_gate = torch.sigmoid(F.linear(x, right_gate_w))
    out_gate   = torch.sigmoid(F.linear(x, out_gate_w))

    # ------------------------------------------------------------------
    # Apply mask + gate (fused via Triton)
    # ------------------------------------------------------------------
    if not nomask:
        mask_h = mask.to(torch.float16)
        left  = mask_gate(left, left_gate, mask_h)
        right = mask_gate(right, right_gate, mask_h)
    else:
        left = left * left_gate
        right = right * right_gate

    # ------------------------------------------------------------------
    # Pairwise multiplicative update (N³ contraction)
    # ------------------------------------------------------------------
    # out[b,i,j,d] = Σₖ left[b,i,k,d] * right[b,j,k,d]
    out = torch.einsum("b i k d, b j k d -> b i j d", left, right)

    # ------------------------------------------------------------------
    # Output LayerNorm over hidden_dim, gate and final linear
    # ------------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(out.dtype)
    to_out_norm_b = weights["to_out_norm.bias"].to(out.dtype)

    out = F.layer_norm(out, (hidden_dim,), weight=to_out_norm_w, bias=to_out_norm_b)

    out = out * out_gate                     # broadcast over hidden_dim

    to_out_w = weights["to_out.weight"].to(out.dtype)
    out = F.linear(out, to_out_w)            # back to `dim`

    # Return in float32 as required by the interface
    return out.to(torch.float32)