"""
TriMul (outgoing) implementation with a fused Triton batched matmul.
The algorithm follows the AlphaFold3 reference:

1. LayerNorm on the 4‑D pair representation.
2. Two linear projections (left/right) and three gating projections.
3. Optional pairwise mask is applied before gating.
4. The core “tri‑multiplicative” step:
       out[b,i,j,d] = Σₖ left[b,i,k,d] * right[b,j,k,d]
   This is a per‑channel batched matrix multiplication, implemented
   by a custom Triton kernel (no Python loops, full GPU parallelism).
5. LayerNorm on the hidden dimension, gated by out_gate and projected back
   to the original channel dimension.

Only the heavy N³ matrix multiplication is performed in Triton; the rest
uses fast PyTorch kernels. The implementation works for any batch size
(1‑2), sequence length up to 1024 and hidden dimensions 128‑384.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# --------------------------------------------------------------------
# Triton kernel: batched matrix multiplication C = A @ B
#   A : [B, M, K]
#   B : [B, K, N]
#   C : [B, M, N]
# --------------------------------------------------------------------
@triton.jit
def _batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    batch, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 3‑D grid: (batch, row tile, col tile)
    bidx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # offsets within the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # masks for out‑of‑bounds rows / cols
    mask_m = offs_m < M
    mask_n = offs_n < N

    # accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A[b, i, k]   -> shape (BLOCK_M, BLOCK_K)
        a_ptrs = (
            a_ptr
            + bidx * stride_ab
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # B[b, k, j]   -> shape (BLOCK_K, BLOCK_N)
        b_ptrs = (
            b_ptr
            + bidx * stride_bb
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        # accumulate the dot product
        acc += tl.dot(a, b)   # (BLOCK_M, BLOCK_N)

    # write back C
    c_ptrs = (
        c_ptr
        + bidx * stride_cb
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def _batched_matmul(A: torch.Tensor, B: torch.Tensor,
                    BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32) -> torch.Tensor:
    """
    Wrapper around the Triton kernel.  A and B must be 3‑D
    float32 tensors on the same CUDA device.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    batch, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2, "inner dimensions must match"
    C = torch.empty((batch, M, N), dtype=torch.float32, device=A.device)

    grid = (
        batch,
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )
    _batched_matmul_kernel[grid](
        a_ptr=A,
        b_ptr=B,
        c_ptr=C,
        batch=batch,
        M=M,
        N=N,
        K=K,
        stride_ab=A.stride(0),
        stride_am=A.stride(1),
        stride_ak=A.stride(2),
        stride_bb=B.stride(0),
        stride_bk=B.stride(1),
        stride_bn=B.stride(2),
        stride_cb=C.stride(0),
        stride_cm=C.stride(1),
        stride_cn=C.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


# --------------------------------------------------------------------
# Entry point used by the test harness
# --------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.
    Arguments:
        data : tuple (input_tensor, mask, weights, config)
            - input_tensor : [B, N, N, dim]  (float32)
            - mask         : [B, N, N]      (float32 or bool)
            - weights      : dict of model parameters
            - config       : dict with keys "dim", "hidden_dim", optional "nomask"
    Returns:
        out : [B, N, N, dim] (float32)
    """
    # ----------------------------------------------------------------
    # unpack inputs
    # ----------------------------------------------------------------
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-5

    # ----------------------------------------------------------------
    # fetch weights
    # ----------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]

    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]

    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]

    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]          # shape (dim, hidden_dim)

    # ----------------------------------------------------------------
    # 1) LayerNorm over the 4‑D tensor (last dim)
    # ----------------------------------------------------------------
    x = F.layer_norm(input_tensor,
                     normalized_shape=(dim,),
                     weight=norm_weight,
                     bias=norm_bias,
                     eps=eps)

    # ----------------------------------------------------------------
    # 2) Linear projections + gating
    # ----------------------------------------------------------------
    left_proj = F.linear(x, left_proj_weight)          # [B, N, N, hidden_dim]
    right_proj = F.linear(x, right_proj_weight)

    left_gate = torch.sigmoid(F.linear(x, left_gate_weight))
    right_gate = torch.sigmoid(F.linear(x, right_gate_weight))
    out_gate = torch.sigmoid(F.linear(x, out_gate_weight))

    # ----------------------------------------------------------------
    # 3) Apply mask (if present) and gating
    # ----------------------------------------------------------------
    if config.get("nomask", False) or mask is None:
        # No masking – just multiply by the gates
        left = left_proj * left_gate
        right = right_proj * right_gate
    else:
        # mask: [B, N, N] -> broadcast over hidden dim
        mask_f = mask.to(dtype=left_proj.dtype).unsqueeze(-1)   # [B, N, N, 1]
        left = left_proj * mask_f * left_gate
        right = right_proj * mask_f * right_gate

    # ----------------------------------------------------------------
    # 4) Core Tri‑multiplicative step: batched matmul per hidden channel
    #    out[b,i,j,d] = Σₖ left[b,i,k,d] * right[b,j,k,d]
    #    We reshape to (B*hidden, N, N) and use a Triton matmul.
    # ----------------------------------------------------------------
    B, N, _, H = left.shape
    # move hidden dim to the batch axis and make the two matrix dimensions explicit
    left_perm = left.permute(0, 3, 1, 2).contiguous()      # (B, H, N, N)  --> (i, k)
    right_perm = right.permute(0, 3, 2, 1).contiguous()    # (B, H, N, N)  --> (k, j)

    # collapse batch+hidden into a single dimension for the kernel
    left_flat = left_perm.view(B * H, N, N)                # (B*H, M=N, K=N)
    right_flat = right_perm.view(B * H, N, N)              # (B*H, K=N, N)

    # Batched matrix multiplication via Triton (C = A @ B)
    out_flat = _batched_matmul(left_flat, right_flat)     # (B*H, N, N)

    # Restore original layout: (B, N, N, H)
    out = out_flat.view(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # ----------------------------------------------------------------
    # 5) Output normalization, gating and final projection
    # ----------------------------------------------------------------
    out_norm = F.layer_norm(out,
                            normalized_shape=(hidden_dim,),
                            weight=to_out_norm_weight,
                            bias=to_out_norm_bias,
                            eps=eps)

    out_gated = out_norm * out_gate   # elementwise gating

    # final linear projection back to dim
    final = F.linear(out_gated, to_out_weight)   # (B, N, N, dim)

    return final