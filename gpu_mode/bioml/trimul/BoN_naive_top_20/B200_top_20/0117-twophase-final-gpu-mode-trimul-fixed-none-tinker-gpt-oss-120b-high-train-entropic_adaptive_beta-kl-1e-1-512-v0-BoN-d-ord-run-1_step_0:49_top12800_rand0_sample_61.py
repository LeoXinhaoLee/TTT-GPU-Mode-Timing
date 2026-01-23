"""
TriMul (outgoing) – Triton‑accelerated forward pass.

Algorithm
---------
1. LayerNorm on the last dimension (`dim`).  Implemented with a custom
   Triton kernel (`_layernorm_triton`) that computes mean/variance per
   row and writes the normalized values (with learned weight & bias).
2. Cast the normalized tensor to FP16 – the heavy part of the model
   is much faster in half‑precision on an H100.
3. Linear projections (`left_proj`, `right_proj`) and the three gates are
   computed with `torch.nn.functional.linear` (FP16) followed by a sigmoid.
4. Apply the optional mask (broadcast over the hidden dimension).
5. The core “TriMul” reduction is
        out[i,j,d] = Σ_k left[i,k,d] * right[j,k,d]
   which is exactly a batched matrix multiplication:
        out_perm = (left_perm) @ (right_perm)   # (B,hidden,i,j)
   where `left_perm = left.permute(0,3,1,2)` and
   `right_perm = right.permute(0,3,2,1)`.
6. Second LayerNorm over the hidden dimension (`hidden_dim`) – again using
   the Triton kernel (auto‑selected FP16 version).
7. Scale by the output gate, apply the final linear projection back to
   `dim`, and return a float‑32 tensor.

Only the two LayerNorms are written in Triton; the rest leverages
high‑performance cuBLAS kernels.  This satisfies the requirement of
“at least part of the operations in a kernel” while keeping overall
runtime competitive on an H100.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernels: LayerNorm (FP32 & FP16 variants)
# ----------------------------------------------------------------------
@triton.jit
def layernorm_kernel_fp32(
    in_ptr, out_ptr,
    weight_ptr, bias_ptr,
    epsilon,
    stride_in_row, stride_in_col,
    stride_out_row, stride_out_col,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """LayerNorm for float32 tensors."""
    pid = tl.program_id(0)                      # each program = one row
    row_off = pid * stride_in_row

    # ----- first pass: compute mean & variance -----
    sum_val = tl.zeros([1], dtype=tl.float32)
    sum_sq  = tl.zeros([1], dtype=tl.float32)

    for offset in range(0, D, BLOCK_SIZE):
        col = offset + tl.arange(0, BLOCK_SIZE)
        mask = col < D
        ptr = in_ptr + row_off + col * stride_in_col
        x = tl.load(ptr, mask=mask, other=0.0)   # FP32
        sum_val += tl.sum(x, axis=0)
        sum_sq  += tl.sum(x * x, axis=0)

    mean = sum_val / D
    var  = sum_sq / D - mean * mean
    inv_std = tl.rsqrt(var + epsilon)

    # ----- second pass: write normalized values -----
    for offset in range(0, D, BLOCK_SIZE):
        col = offset + tl.arange(0, BLOCK_SIZE)
        mask = col < D
        ptr = in_ptr + row_off + col * stride_in_col
        x = tl.load(ptr, mask=mask, other=0.0)   # FP32
        x_hat = (x - mean) * inv_std

        w = tl.load(weight_ptr + col, mask=mask, other=0.0)   # FP32
        b = tl.load(bias_ptr   + col, mask=mask, other=0.0)   # FP32
        x_hat = x_hat * w + b

        out_ptr_i = out_ptr + pid * stride_out_row + col * stride_out_col
        tl.store(out_ptr_i, x_hat, mask=mask)


@triton.jit
def layernorm_kernel_fp16(
    in_ptr, out_ptr,
    weight_ptr, bias_ptr,
    epsilon,
    stride_in_row, stride_in_col,
    stride_out_row, stride_out_col,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """LayerNorm for half‑precision tensors (computations in FP32)."""
    pid = tl.program_id(0)
    row_off = pid * stride_in_row

    sum_val = tl.zeros([1], dtype=tl.float32)
    sum_sq  = tl.zeros([1], dtype=tl.float32)

    for offset in range(0, D, BLOCK_SIZE):
        col = offset + tl.arange(0, BLOCK_SIZE)
        mask = col < D
        ptr = in_ptr + row_off + col * stride_in_col
        x = tl.load(ptr, mask=mask, other=0.0)    # FP16
        x_f = x.to(tl.float32)
        sum_val += tl.sum(x_f, axis=0)
        sum_sq  += tl.sum(x_f * x_f, axis=0)

    mean = sum_val / D
    var  = sum_sq / D - mean * mean
    inv_std = tl.rsqrt(var + epsilon)

    for offset in range(0, D, BLOCK_SIZE):
        col = offset + tl.arange(0, BLOCK_SIZE)
        mask = col < D
        ptr = in_ptr + row_off + col * stride_in_col
        x = tl.load(ptr, mask=mask, other=0.0)    # FP16
        x_f = x.to(tl.float32)
        x_hat = (x_f - mean) * inv_std

        w = tl.load(weight_ptr + col, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr   + col, mask=mask, other=0.0).to(tl.float32)
        x_hat = x_hat * w + b

        out_ptr_i = out_ptr + pid * stride_out_row + col * stride_out_col
        tl.store(out_ptr_i, x_hat.to(tl.float16), mask=mask)


def _layernorm_triton(x: torch.Tensor,
                      weight: torch.Tensor,
                      bias: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
    """
    Apply LayerNorm on the *last* dimension using a Triton kernel.
    Supports FP32 and FP16 tensors.
    """
    assert x.dim() == 2, "LayerNorm kernel expects a 2‑D tensor (M, D)."
    M, D = x.shape
    out = torch.empty_like(x)

    # Choose block size heuristically (must divide by 2 for warp efficiency)
    BLOCK_SIZE = 128 if D > 64 else 64

    if x.dtype == torch.float16:
        kernel = layernorm_kernel_fp16
    else:
        kernel = layernorm_kernel_fp32

    # Launch one program per row
    grid = (M,)
    kernel[grid](
        x,
        out,
        weight,
        bias,
        eps,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Triton‑accelerated forward pass of the “outgoing” TriMul module.

    Args:
        data: tuple (input_tensor, mask, weights, config)
            - input_tensor: torch.Tensor of shape [B, N, N, dim]
            - mask: torch.Tensor of shape [B, N, N] (may be ignored)
            - weights: dict of model parameters
            - config: dict with keys "dim", "hidden_dim", "nomask" (optional)

    Returns:
        torch.Tensor of shape [B, N, N, dim] (float32)
    """
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    eps = 1e-5

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)

    # -----------------------------------------------------------------
    # 1) First LayerNorm over `dim`
    # -----------------------------------------------------------------
    # reshape to (M, dim) where M = B * N * N
    x_flat = input_tensor.reshape(-1, dim)
    x_norm = _layernorm_triton(
        x_flat,
        weights["norm.weight"],
        weights["norm.bias"],
        eps,
    ).reshape_as(input_tensor)                     # (B, N, N, dim)

    # -----------------------------------------------------------------
    # 2) Cast to FP16 for compute‑heavy part
    # -----------------------------------------------------------------
    x_fp16 = x_norm.to(torch.float16)

    # -----------------------------------------------------------------
    # 3) Linear projections + gates (all in FP16)
    # -----------------------------------------------------------------
    left_proj_w   = weights["left_proj.weight"].to(torch.float16)
    right_proj_w  = weights["right_proj.weight"].to(torch.float16)
    left_gate_w   = weights["left_gate.weight"].to(torch.float16)
    right_gate_w  = weights["right_gate.weight"].to(torch.float16)
    out_gate_w    = weights["out_gate.weight"].to(torch.float16)

    left  = F.linear(x_fp16, left_proj_w)            # (..., hidden_dim)
    right = F.linear(x_fp16, right_proj_w)

    left_gate  = torch.sigmoid(F.linear(x_fp16, left_gate_w))
    right_gate = torch.sigmoid(F.linear(x_fp16, right_gate_w))
    out_gate   = torch.sigmoid(F.linear(x_fp16, out_gate_w))

    left  = left * left_gate
    right = right * right_gate

    if not nomask:
        # mask is broadcast over the hidden dimension
        mask_fp = mask.to(torch.float16).unsqueeze(-1)   # (B, N, N, 1)
        left  = left * mask_fp
        right = right * mask_fp

    # -----------------------------------------------------------------
    # 4) TriMul reduction (batched matmul)
    # -----------------------------------------------------------------
    # left/right shape: (B, N, N, hidden_dim)
    left_perm  = left.permute(0, 3, 1, 2)   # (B, hidden_dim, i, k)
    right_perm = right.permute(0, 3, 2, 1)  # (B, hidden_dim, k, j)

    out_perm = torch.matmul(left_perm, right_perm)   # (B, hidden_dim, i, j)
    out = out_perm.permute(0, 2, 3, 1)               # (B, i, j, hidden_dim)

    # -----------------------------------------------------------------
    # 5) Second LayerNorm over the hidden dimension
    # -----------------------------------------------------------------
    out_flat = out.reshape(-1, hidden_dim)
    out_norm = _layernorm_triton(
        out_flat,
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
        eps,
    ).reshape_as(out)                                 # (B, N, N, hidden_dim)

    # -----------------------------------------------------------------
    # 6) Apply output gate and final projection back to `dim`
    # -----------------------------------------------------------------
    out_gated = out_norm * out_gate
    to_out_w = weights["to_out.weight"].to(torch.float16)
    out_final = F.linear(out_gated, to_out_w)          # (..., dim)

    # Return float32 as required by the reference implementation
    return out_final.to(torch.float32)