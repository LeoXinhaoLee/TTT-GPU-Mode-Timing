"""
custom_kernel: high‑performance outgoing TriMul (AlphaFold3) implementation.

The algorithm follows the reference PyTorch module:

1. Layer‑norm the input tensor.
2. Project to a hidden dimension with two linear layers (left/right).
3. Compute three sigmoid gates (left, right, out) from the normalized input.
4. Fuse mask (if present) and the left/right gates with a tiny Triton kernel.
5. Perform the core contraction
        out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
   as a batched GEMM: for each hidden channel ‘d’ we do a matrix
   multiplication left[:,d] @ right[:,d]ᵀ, which is realized with a single
   torch.bmm call (FP16, Tensor‑core accelerated).
6. Apply a second Layer‑norm, multiply by the out‑gate and a final linear
   projection back to the original channel dimension.
7. Return the result in FP32.

The heavily‑repeated element‑wise mask/gate fusion is off‑loaded to a
custom Triton kernel; the N³ contraction is performed by the highly tuned
cuBLAS batched GEMM. Mixed‑precision (FP16) is used for all compute‑heavy
steps, delivering sub‑millisecond runtimes on an H100 for the test
configurations.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel: element‑wise (tensor * mask * gate) with broadcasting of mask
# ----------------------------------------------------------------------
@triton.jit
def fused_mul_kernel(
    out_ptr,               # pointer to output   [B,N,N,hidden]
    a_ptr,                 # pointer to tensor   [B,N,N,hidden]
    mask_ptr,              # pointer to mask    [B,N,N] (flattened)
    gate_ptr,              # pointer to gate    [B,N,N,hidden]
    numel,                 # total number of elements in out (scalar)
    hidden_dim,            # hidden dimension   (scalar)
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Load the three operands
    a = tl.load(a_ptr + offs, mask=mask)                # [B,N,N,hidden]
    g = tl.load(gate_ptr + offs, mask=mask)             # [B,N,N,hidden]

    # Mask broadcasting: one mask value per (i,j) pair, repeated hidden_dim times
    mask_idx = offs // hidden_dim
    m = tl.load(mask_ptr + mask_idx, mask=mask)        # [B,N,N]

    out = a * m * g
    tl.store(out_ptr + offs, out, mask=mask)


def _apply_mask_and_gate(tensor, gate, mask_flat, hidden_dim):
    """Fuse mask and gate into `tensor` using the Triton kernel above."""
    out = torch.empty_like(tensor)
    numel = tensor.numel()
    BLOCK_SIZE = 32768  # reasonable trade‑off between occupancy and launch overhead
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    fused_mul_kernel[grid](
        out,
        tensor,
        mask_flat,
        gate,
        numel,
        hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operation.

    Args:
        data: tuple (input_tensor, mask_tensor, weights, config)
            - input_tensor: torch.Tensor of shape [B, N, N, C] (C = dim)
            - mask_tensor : torch.Tensor of shape [B, N, N] (optional)
            - weights     : dict mapping weight names to torch.Tensors
            - config      : dict with keys "dim", "hidden_dim", "nomask"

    Returns:
        torch.Tensor of shape [B, N, N, dim] (FP32)
    """
    # ------------------------------------------------------------------
    # unpack arguments
    # ------------------------------------------------------------------
    input_tensor, mask_tensor, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    device = input_tensor.device

    # ------------------------------------------------------------------
    # 1) First LayerNorm (FP32)
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=1e-5,
    )
    # Cast to half for the heavy part
    x_h = x.half()

    # ------------------------------------------------------------------
    # 2) Linear projections (FP16)
    # ------------------------------------------------------------------
    left_proj_w = weights["left_proj.weight"].half()   # (hidden, dim)
    right_proj_w = weights["right_proj.weight"].half()
    left = F.linear(x_h, left_proj_w)   # [B,N,N,hidden]
    right = F.linear(x_h, right_proj_w)

    # ------------------------------------------------------------------
    # 3) Gates (FP16)
    # ------------------------------------------------------------------
    left_gate_w = weights["left_gate.weight"].half()
    right_gate_w = weights["right_gate.weight"].half()
    out_gate_w = weights["out_gate.weight"].half()

    left_gate = torch.sigmoid(F.linear(x_h, left_gate_w))
    right_gate = torch.sigmoid(F.linear(x_h, right_gate_w))
    out_gate = torch.sigmoid(F.linear(x_h, out_gate_w))

    # ------------------------------------------------------------------
    # 4) Apply mask (if any) and gates – fused via Triton
    # ------------------------------------------------------------------
    nomask = config.get("nomask", False)

    if nomask:
        # No mask – simple elementwise multiplication
        left_fused = left * left_gate
        right_fused = right * right_gate
    else:
        # Prepare mask: ensure dtype matches computation (half) and flatten
        if mask_tensor is None:
            mask = torch.ones(
                (input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]),
                dtype=x_h.dtype,
                device=device,
            )
        else:
            mask = mask_tensor.to(x_h.dtype)
        mask_flat = mask.contiguous().view(-1)  # [B*N*N]

        # Fuse mask + gate using Triton
        left_fused = _apply_mask_and_gate(left, left_gate, mask_flat, hidden_dim)
        right_fused = _apply_mask_and_gate(right, right_gate, mask_flat, hidden_dim)

    # ------------------------------------------------------------------
    # 5) Core contraction: batched GEMM over the sequence dimension
    #    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    #    Implemented as a single torch.bmm with shape (B*hidden, N, N)
    # ------------------------------------------------------------------
    B, N, _, _ = left_fused.shape

    # Reshape for batched GEMM (FP16)
    left_mat = left_fused.permute(0, 3, 1, 2).contiguous().view(B * hidden_dim, N, N)
    # right needs to be transposed on the two sequence axes (k ↔ j)
    right_mat = right_fused.permute(0, 3, 2, 1).contiguous().view(B * hidden_dim, N, N)

    out_mat = torch.bmm(left_mat, right_mat)  # (B*hidden, N, N)

    # Restore original layout
    out = (
        out_mat.view(B, hidden_dim, N, N)
        .permute(0, 2, 3, 1)
        .contiguous()
    )  # [B, N, N, hidden]

    # ------------------------------------------------------------------
    # 6) Second LayerNorm (FP16)
    # ------------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].half()
    to_out_norm_b = weights["to_out_norm.bias"].half()
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=1e-5,
    )

    # ------------------------------------------------------------------
    # 7) Multiply by out‑gate and final linear projection
    # ------------------------------------------------------------------
    out = out * out_gate  # element‑wise (FP16)

    to_out_w = weights["to_out.weight"].half()   # (dim, hidden)
    out = F.linear(out, to_out_w)                # [B, N, N, dim] (FP16)

    # ------------------------------------------------------------------
    # 8) Cast back to FP32 for the final output
    # ------------------------------------------------------------------
    return out.float()