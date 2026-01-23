"""
TriMul (outgoing) forward pass – Triton‑accelerated implementation.

The module receives a 4‑D tensor X ∈ ℝ^{B×N×N×C} and a pairwise mask.
The computation follows the PyTorch reference:

1. LayerNorm over the channel dimension C.
2. Two linear projections (left/right) → hidden dimension H (no bias).
3. Optional mask applied to the projected tensors.
4. Gating (sigmoid) for left, right and output streams.
5. Pair‑wise multiplicative update:
      out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]
   which is a batched matrix multiplication
      out_h = left_h @ right_hᵀ   for each hidden channel h.
   This step is performed by a custom Triton kernel that processes all
   hidden channels (and the batch) in one launch.
6. LayerNorm over the hidden dimension, multiplication by the output gate,
   and a final linear projection back to C.

Only the heavy O(B·H·N³) GEMM is executed on Triton; all other
operations stay in PyTorch.  The kernel works in FP16 for speed
while accumulations are performed in FP32 to preserve numerical quality.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Dict


@triton.jit
def _batch_matmul_kernel(
    a_ptr, b_ptr, c_ptr,                 # pointers
    M, N, K,                             # matrix sizes (all equal to N)
    stride_ab, stride_am, stride_ak,      # A strides: batch, row, col
    stride_bb, stride_bn, stride_bk,      # B strides: batch, row, col  (B is N×K)
    stride_cb, stride_cm, stride_cn,      # C strides: batch, row, col
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched GEMM C = A @ Bᵀ  (A,B ∈ FP16, C ∈ FP16, acc in FP32)."""
    pid_m = tl.program_id(0)            # block index over M dimension
    pid_n = tl.program_id(1)            # block index over N dimension
    pid_b = tl.program_id(2)            # batch * hidden channel index

    # Block start offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for the tails
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for k in range(num_k_blocks):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A block (M x K)
        a_ptrs = a_ptr + pid_b * stride_ab + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0)

        # Bᵀ block (K x N) – we load B transposed on‑the‑fly
        # Original B layout: (batch, N, K) → stride_bn (row), stride_bk (col)
        # Bᵀ[ k, n ] = B[ n, k ]
        b_ptrs = b_ptr + pid_b * stride_bb + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0)

        # Rank‑K update
        acc += tl.dot(a, b)   # (M,K)·(K,N) → (M,N)

    # Write C block
    c_ptrs = c_ptr + pid_b * stride_cb + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs,
             acc.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]) -> torch.Tensor:
    """
    Triton‑accelerated forward of the outgoing TriMul operator.

    Arguments
    ---------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor[bs, N, N, dim] (float32)
        - mask         : torch.Tensor[bs, N, N]   (bool/float)
        - weights      : dict of model parameters
        - config       : dict with keys "dim", "hidden_dim", "nomask"

    Returns
    -------
    torch.Tensor
        Output tensor of shape [bs, N, N, dim] (float32)
    """
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-6

    # ---------- 1. LayerNorm over channel ------------------------------------
    x = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
        eps=eps,
    )  # [B, N, N, C]

    # ---------- 2. Linear projections ----------------------------------------
    left = torch.nn.functional.linear(x, weights["left_proj.weight"])   # [B,N,N,H]
    right = torch.nn.functional.linear(x, weights["right_proj.weight"]) # [B,N,N,H]

    # ---------- 3. Optional mask ---------------------------------------------
    if not config.get("nomask", True):
        mask_exp = mask.unsqueeze(-1).to(left.dtype)   # [B,N,N,1]
        left = left * mask_exp
        right = right * mask_exp

    # ---------- 4. Gates ----------------------------------------------------
    left_gate = torch.nn.functional.linear(x, weights["left_gate.weight"]).sigmoid()
    right_gate = torch.nn.functional.linear(x, weights["right_gate.weight"]).sigmoid()
    out_gate = torch.nn.functional.linear(x, weights["out_gate.weight"]).sigmoid()

    left = left * left_gate
    right = right * right_gate

    # ---------- 5. Batched GEMM via Triton ---------------------------------
    # Cast to FP16 for the heavy compute
    left = left.to(torch.float16)
    right = right.to(torch.float16)

    B, N, _, _ = left.shape

    # Move hidden dimension to the batch axis and flatten batch*hidden
    left_perm = left.permute(0, 3, 1, 2).contiguous()   # [B, H, N, N]
    right_perm = right.permute(0, 3, 1, 2).contiguous()

    left_flat = left_perm.view(-1, N, N)   # [B*H, N, N] (FP16)
    right_flat = right_perm.view(-1, N, N)

    out_flat = torch.empty_like(left_flat)  # FP16 output

    # Kernel launch configuration
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    num_m_blocks = (N + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    batch_dim = left_flat.shape[0]   # B * H

    # Strides (in elements, not bytes)
    stride_ab, stride_am, stride_ak = left_flat.stride()
    stride_bb, stride_bn, stride_bk = right_flat.stride()
    stride_cb, stride_cm, stride_cn = out_flat.stride()

    # Launch the Triton kernel
    _batch_matmul_kernel[(num_m_blocks, num_n_blocks, batch_dim)](
        left_flat,
        right_flat,
        out_flat,
        N, N, N,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bn, stride_bk,
        stride_cb, stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=3,
        num_warps=4,
    )

    # Reshape back to [B, N, N, H] (float32)
    out = out_flat.to(torch.float32)
    out = out.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    # ---------- 6. Output LayerNorm, gating and final projection ----------
    out = torch.nn.functional.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
        eps=eps,
    )

    out = out * out_gate  # apply output gate (both FP32)

    out = torch.nn.functional.linear(out, weights["to_out.weight"])  # final projection to dim

    return out