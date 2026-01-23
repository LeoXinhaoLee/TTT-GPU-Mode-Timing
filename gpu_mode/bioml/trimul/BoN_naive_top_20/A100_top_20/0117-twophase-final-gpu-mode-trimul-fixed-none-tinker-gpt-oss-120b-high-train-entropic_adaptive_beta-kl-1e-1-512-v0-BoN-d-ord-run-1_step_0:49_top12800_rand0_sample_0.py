"""
TriMul (outgoing) forward pass with a fused Triton kernel.

The kernel fuses three operations that are element‑wise and memory bound:
    left  = (x @ W_left)   * sigmoid(x @ W_left_gate)   * mask
    right = (x @ W_right)  * sigmoid(x @ W_right_gate)  * mask

All other heavy work (LayerNorms and the large batched matrix
multiplication over the sequence dimension) is delegated to highly‑optimised
PyTorch kernels (cublas / ATen).  The implementation supports both the
masked and mask‑free variants and works with float32 inputs (the heavy
computations are performed in float16 for speed on H100).

The function signature follows the specification:
    custom_kernel((input, mask, weights, config)) -> output
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: fuse the linear projection + gate + optional mask.
# Input tensors are flattened 1‑D views of shape (B, N, N, H) in C‑order:
#   offset = ((b * N + i) * N + j) * H + h
# The mask (if present) has shape (B, N, N) and is broadcasted over H.
# ---------------------------------------------------------------------------
@triton.jit
def _fuse_gate_mask_kernel(
    lp_ptr, rp_ptr,                # left/right projection (H‑dim)
    lg_ptr, rg_ptr,                # left/right gate (H‑dim)
    mask_ptr,                      # mask (B,N,N) – unused when HAS_MASK=False
    left_out_ptr, right_out_ptr,   # fused outputs
    numel, H,                      # total #elements, hidden dim
    HAS_MASK: tl.constexpr,        # compile‑time flag
    BLOCK_SIZE: tl.constexpr,      # threads per program
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    valid = offs < numel

    # Load the four tensors
    lp = tl.load(lp_ptr + offs, mask=valid, other=0.0)
    rp = tl.load(rp_ptr + offs, mask=valid, other=0.0)
    lg = tl.load(lg_ptr + offs, mask=valid, other=0.0)
    rg = tl.load(rg_ptr + offs, mask=valid, other=0.0)

    if HAS_MASK:
        # Offset into the mask: drop the fast H dimension
        mask_idx = offs // H
        m = tl.load(mask_ptr + mask_idx, mask=valid, other=1.0)
        left = lp * lg * m
        right = rp * rg * m
    else:
        left = lp * lg
        right = rp * rg

    tl.store(left_out_ptr + offs, left, mask=valid)
    tl.store(right_out_ptr + offs, right, mask=valid)


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.

    Args:
        data: tuple of (input_tensor, mask, weights, config)
            - input_tensor: torch.Tensor [B, N, N, D] (float32)
            - mask:        torch.Tensor [B, N, N] (float32/ bool) – may be ignored.
            - weights:     dict with all linear / layer‑norm parameters.
            - config:      dict with keys "dim", "hidden_dim", "nomask" ...

    Returns:
        torch.Tensor of shape [B, N, N, D] (float32)
    """
    # -----------------------------------------------------------------------
    # Unpack arguments
    # -----------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    B, N, _, D = input_tensor.shape
    hidden = config["hidden_dim"]
    nomask = config.get("nomask", False)

    # -----------------------------------------------------------------------
    # LayerNorm over the channel dimension (dim = D)
    # -----------------------------------------------------------------------
    norm_w = weights["norm.weight"].to(input_tensor.dtype)
    norm_b = weights["norm.bias"].to(input_tensor.dtype)
    x = F.layer_norm(
        input_tensor,
        (D,),
        weight=norm_w,
        bias=norm_b,
        eps=1e-5,
    )  # [B,N,N,D]

    # Cast to float16 for the bulk of the computation
    compute_dtype = torch.float16
    x = x.to(compute_dtype)

    # -----------------------------------------------------------------------
    # Linear projections (no bias) – keep everything in half precision
    # -----------------------------------------------------------------------
    W_left = weights["left_proj.weight"].to(compute_dtype)
    W_right = weights["right_proj.weight"].to(compute_dtype)
    W_left_gate = weights["left_gate.weight"].to(compute_dtype)
    W_right_gate = weights["right_gate.weight"].to(compute_dtype)
    W_out_gate = weights["out_gate.weight"].to(compute_dtype)

    #   shape after linear: [B,N,N,hidden]
    left_proj = F.linear(x, W_left)          # L = x @ W_left.T
    right_proj = F.linear(x, W_right)

    left_gate = torch.sigmoid(F.linear(x, W_left_gate))
    right_gate = torch.sigmoid(F.linear(x, W_right_gate))
    out_gate = torch.sigmoid(F.linear(x, W_out_gate))   # used later

    # -----------------------------------------------------------------------
    # Optional mask (broadcast over hidden dimension)
    # -----------------------------------------------------------------------
    if not nomask:
        mask_f = mask.to(compute_dtype)
        mask_f = mask_f.unsqueeze(-1)                 # [B,N,N,1]
        mask_f = mask_f.contiguous()
    else:
        # dummy mask – never read in the kernel (HAS_MASK=False)
        mask_f = left_proj  # any tensor works; will be ignored

    # -----------------------------------------------------------------------
    # Fuse gate + mask with a Triton kernel
    # -----------------------------------------------------------------------
    # Ensure all tensors are contiguous for a 1‑D view
    left_proj = left_proj.contiguous()
    right_proj = right_proj.contiguous()
    left_gate = left_gate.contiguous()
    right_gate = right_gate.contiguous()
    mask_f = mask_f.contiguous()

    left_out = torch.empty_like(left_proj)
    right_out = torch.empty_like(right_proj)

    total_elems = left_proj.numel()                     # B*N*N*hidden
    BLOCK = 1024                                        # tunable

    grid = lambda meta: (triton.cdiv(total_elems, meta["BLOCK_SIZE"]),)

    _HAS_MASK = not nomask

    _fuse_gate_mask_kernel[grid](
        left_proj,
        right_proj,
        left_gate,
        right_gate,
        mask_f,
        left_out,
        right_out,
        total_elems,
        hidden,
        HAS_MASK=_HAS_MASK,
        BLOCK_SIZE=BLOCK,
    )

    # -----------------------------------------------------------------------
    # Batched matrix multiplication over the sequence dimension.
    # For each hidden channel we compute:
    #   out_h = left_h @ right_h^T   (sum over the K dimension)
    # -----------------------------------------------------------------------
    # reshape to (B, hidden, N, N) for torch.matmul
    left_perm = left_out.permute(0, 3, 1, 2).contiguous()
    right_perm = right_out.permute(0, 3, 1, 2).contiguous()

    # (B, hidden, N, N) <- matmul of (B, hidden, N, N) x (B, hidden, N, N)^T
    out_perm = torch.matmul(left_perm, right_perm.transpose(-2, -1))
    # back to (B, N, N, hidden)
    out = out_perm.permute(0, 2, 3, 1).contiguous()

    # -----------------------------------------------------------------------
    # Final LayerNorm over the hidden dimension, gate and linear projection
    # -----------------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(compute_dtype)
    to_out_norm_b = weights["to_out_norm.bias"].to(compute_dtype)

    out = F.layer_norm(
        out,
        (hidden,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=1e-5,
    )

    # Apply the output gate (shape matches)
    out = out * out_gate

    # Linear projection back to original channel dimension
    W_to_out = weights["to_out.weight"].to(compute_dtype)   # (D, hidden)
    out = F.linear(out, W_to_out)                          # [B,N,N,D]

    # Cast back to the input dtype (usually float32)
    return out.to(input_tensor.dtype)