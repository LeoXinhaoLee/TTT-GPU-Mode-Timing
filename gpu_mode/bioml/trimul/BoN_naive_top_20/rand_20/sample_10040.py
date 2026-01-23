"""
TriMul (outgoing) custom kernel for AlphaFold3.

The heavy N³ tensor contraction
    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
is mathematically a batched matrix multiplication
    out[b,d] = left[b,d] @ right[b,d].T
for each hidden channel d.  All surrounding operations
(layer‑norm, linear projections, gating and optional masking) are
performed with PyTorch, while the core contraction is executed by a
hand‑written Triton kernel that tiles the output in (BLOCK_M,
BLOCK_N) and accumulates over the K dimension.  This yields a fast
implementation on H100 GPUs while keeping the final result identical to
the reference PyTorch code.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def trmul_matmul_kernel(
    left_ptr,                 # *float32
    right_ptr,                # *float32
    out_ptr,                  # *float32
    B, N, H,
    stride_left_batch, stride_left_i, stride_left_k, stride_left_h,
    stride_right_batch, stride_right_i, stride_right_k, stride_right_h,
    stride_out_batch, stride_out_i, stride_out_j, stride_out_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched matmul across hidden channels.

    For each (b, h) we compute
        out[b, :, :, h] = left[b, :, :, h] @ right[b, :, :, h].T
    where left/right are (N, N) matrices.  The kernel tiles the output in
    (BLOCK_M, BLOCK_N) and loops over K in blocks of size BLOCK_K.
    """
    pid_m = tl.program_id(0)   # tile row index
    pid_n = tl.program_id(1)   # tile col index
    pid_bh = tl.program_id(2)  # combined batch * hidden index

    # decode batch and hidden dim from the combined index
    batch = pid_bh // H
    hidden = pid_bh % H

    # offsets of the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < N
    mask_n = offs_n < N

    # base pointers for the current (batch, hidden) slice
    left_base  = left_ptr  + batch * stride_left_batch  + hidden * stride_left_h
    right_base = right_ptr + batch * stride_right_batch + hidden * stride_right_h
    out_base   = out_ptr   + batch * stride_out_batch   + hidden * stride_out_h

    # accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over the shared K dimension
    for k in range(0, tl.cdiv(N, BLOCK_K)):
        k_off = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_off < N

        # ---------- load left tile (M x K) ----------
        # left[b, i, k, h]
        left_ptrs = left_base + offs_m[:, None] * stride_left_i + k_off[None, :] * stride_left_k
        left_tile = tl.load(
            left_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        # ---------- load right tile transposed (K x N) ----------
        # we need right[j, k, h] but as a K x N tile:
        # right_transposed[k, j] = right[j, k, h]
        right_ptrs = right_base + k_off[:, None] * stride_right_k + offs_n[None, :] * stride_right_i
        right_tile = tl.load(
            right_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        # ---------- accumulate ----------
        acc += tl.dot(left_tile, right_tile)   # (M,K) @ (K,N) -> (M,N)

    # ---------- write output ----------
    out_ptrs = out_base + offs_m[:, None] * stride_out_i + offs_n[None, :] * stride_out_j
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def trmul_einsum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """
    Launches ``trmul_matmul_kernel`` to evaluate the Einstein summation
    used in TriMul:
        out[..., i, j, d] = Σ_k left[..., i, k, d] * right[..., j, k, d]

    Args:
        left:  torch.Tensor of shape [B, N, N, H], contiguous, float32
        right: torch.Tensor of shape [B, N, N, H], contiguous, float32

    Returns:
        out: torch.Tensor of shape [B, N, N, H] (float32)
    """
    assert left.is_contiguous() and right.is_contiguous()
    B, N, _, H = left.shape
    out = torch.empty_like(left)

    # strides in element units (not bytes)
    ls = left.stride()
    rs = right.stride()
    os = out.stride()

    # tile sizes – chosen to fit H100 shared memory comfortably
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        B * H,
    )

    trmul_matmul_kernel[grid](
        left,
        right,
        out,
        B, N, H,
        ls[0], ls[1], ls[2], ls[3],
        rs[0], rs[1], rs[2], rs[3],
        os[0], os[1], os[2], os[3],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul module.

    Args:
        data: tuple (input_tensor, mask, weights, config)
            - input_tensor: torch.Tensor of shape [B, N, N, dim]
            - mask: torch.Tensor of shape [B, N, N] (or None)
            - weights: dict mapping weight names to torch.Tensors
            - config: dict with at least keys ``dim`` and ``hidden_dim``,
                      optional ``nomask`` (bool, default True)

    Returns:
        torch.Tensor of shape [B, N, N, dim]
    """
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)

    # ----------------------------------------------------------------------
    # Load model parameters onto the same device
    # ----------------------------------------------------------------------
    norm_weight = weights["norm.weight"].to(device)
    norm_bias   = weights["norm.bias"].to(device)

    left_proj_weight   = weights["left_proj.weight"].to(device)
    right_proj_weight  = weights["right_proj.weight"].to(device)

    left_gate_weight   = weights["left_gate.weight"].to(device)
    right_gate_weight  = weights["right_gate.weight"].to(device)
    out_gate_weight    = weights["out_gate.weight"].to(device)

    to_out_norm_weight = weights["to_out_norm.weight"].to(device)
    to_out_norm_bias   = weights["to_out_norm.bias"].to(device)

    to_out_weight      = weights["to_out.weight"].to(device)

    # ----------------------------------------------------------------------
    # LayerNorm on the input tensor (dim = last dimension)
    # ----------------------------------------------------------------------
    eps = 1e-5
    mean = input_tensor.mean(dim=-1, keepdim=True)
    var  = ((input_tensor - mean) ** 2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    x_norm = (input_tensor - mean) * inv_std
    x_norm = x_norm * norm_weight + norm_bias   # broadcast over B,N,N

    # ----------------------------------------------------------------------
    # Linear projections (no bias)
    # ----------------------------------------------------------------------
    left  = torch.nn.functional.linear(x_norm, left_proj_weight)   # [B,N,N,H]
    right = torch.nn.functional.linear(x_norm, right_proj_weight)

    # ----------------------------------------------------------------------
    # Optional masking (mask shape [B, N, N])
    # ----------------------------------------------------------------------
    if not nomask and mask is not None:
        mask_f = mask.unsqueeze(-1).to(x_norm.dtype)   # [B,N,N,1]
        left  = left * mask_f
        right = right * mask_f

    # ----------------------------------------------------------------------
    # Gating (sigmoid) applied after masking
    # ----------------------------------------------------------------------
    left_gate  = torch.nn.functional.linear(x_norm, left_gate_weight).sigmoid()
    right_gate = torch.nn.functional.linear(x_norm, right_gate_weight).sigmoid()
    out_gate   = torch.nn.functional.linear(x_norm, out_gate_weight).sigmoid()

    left  = left * left_gate
    right = right * right_gate

    # ----------------------------------------------------------------------
    # Core O(N³) contraction via Triton
    # ----------------------------------------------------------------------
    left  = left.contiguous()
    right = right.contiguous()
    out = trmul_einsum(left, right)          # [B,N,N,H]

    # ----------------------------------------------------------------------
    # LayerNorm on the hidden dimension (H)
    # ----------------------------------------------------------------------
    mean2 = out.mean(dim=-1, keepdim=True)
    var2  = ((out - mean2) ** 2).mean(dim=-1, keepdim=True)
    inv_std2 = torch.rsqrt(var2 + eps)
    out_norm = (out - mean2) * inv_std2
    out_norm = out_norm * to_out_norm_weight + to_out_norm_bias

    # ----------------------------------------------------------------------
    # Output gating and final linear projection back to dim
    # ----------------------------------------------------------------------
    out_norm = out_norm * out_gate
    result = torch.nn.functional.linear(out_norm, to_out_weight)   # [B,N,N,dim]

    return result