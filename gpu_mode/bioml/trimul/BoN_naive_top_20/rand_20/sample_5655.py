"""
TriMul outgoing kernel.

Implements the forward pass of the TriMul (AlphaFold3) module.
All cheap operations (LayerNorm, linear projections, gating) are done with
PyTorch. The expensive N³ contraction

    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]

is fused into a batched GEMM and executed with a custom Triton kernel.
The hidden dimension is folded into the batch axis, so the kernel computes a
standard batch‐matrix‑multiply C = A @ B for many (batch*hidden) matrices.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    BATCH, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """C = A @ B  with shapes
       A : (BATCH, M, K)
       B : (BATCH, K, N)
       C : (BATCH, M, N)
    """
    pid = tl.program_id(0)

    # number of tiles covering the output matrix
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)

    # decode a 3‑D grid (batch, tile_m, tile_n) from the 1‑D pid
    batch_idx = pid // (num_m_blocks * num_n_blocks)
    pid_in_batch = pid % (num_m_blocks * num_n_blocks)
    pid_m = pid_in_batch // num_n_blocks
    pid_n = pid_in_batch % num_n_blocks

    # offsets inside the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # k‑tile offsets (shared across rows/cols)
    offs_k = tl.arange(0, BLOCK_K)

    # base pointers for this batch
    a_batch = a_ptr + batch_idx * stride_ab
    b_batch = b_ptr + batch_idx * stride_bb
    c_batch = c_ptr + batch_idx * stride_cb

    # pointers to the beginning of the current M×K and K×N tiles
    a_tile_ptr = a_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_tile_ptr = b_batch + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cur_k = k * BLOCK_K
        a = tl.load(
            a_tile_ptr + cur_k * stride_ak,
            mask=mask_m[:, None] & (cur_k + offs_k < K),
            other=0.0,
        )
        b = tl.load(
            b_tile_ptr + cur_k * stride_bk,
            mask=(cur_k + offs_k[:, None] < K) & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)        # (BLOCK_M, BLOCK_N)

    # write back
    c_tile_ptr = c_batch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_tile_ptr, acc, mask=mask_m[:, None] & mask_n[None, :])


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.

    Parameters
    ----------
    data : tuple
        (input_tensor, mask_tensor, weights, config)
        - input_tensor : torch.Tensor, shape [B, N, N, dim]
        - mask_tensor  : torch.Tensor, shape [B, N, N] (ignored if config["nomask"] is True)
        - weights      : dict of weight tensors (see keys below)
        - config       : dict containing at least "dim", "hidden_dim", "nomask"

    Returns
    -------
    torch.Tensor
        Output tensor, shape [B, N, N, dim] (float32)
    """
    # ------------------------------------------------------------------
    # unpack
    # ------------------------------------------------------------------
    input_tensor, mask_tensor, weights, config = data
    device = input_tensor.device

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-5

    # ------------------------------------------------------------------
    # weight tensors
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    # ------------------------------------------------------------------
    # 1) Input layer‑norm
    # ------------------------------------------------------------------
    x = torch.nn.functional.layer_norm(
        input_tensor,
        (dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
    )  # [B, N, N, dim]

    # ------------------------------------------------------------------
    # 2) Linear projections (no bias)
    # ------------------------------------------------------------------
    left = torch.nn.functional.linear(x, left_proj_weight)   # [B, N, N, hidden]
    right = torch.nn.functional.linear(x, right_proj_weight)

    # ------------------------------------------------------------------
    # 3) Optional masking (applied after projection)
    # ------------------------------------------------------------------
    if not config.get("nomask", False):
        mask = mask_tensor.unsqueeze(-1).type_as(left)   # broadcast over hidden
        left = left * mask
        right = right * mask

    # ------------------------------------------------------------------
    # 4) Gating – sigmoid of additional linear transforms of the *original* x
    # ------------------------------------------------------------------
    left_gate = torch.sigmoid(torch.nn.functional.linear(x, left_gate_weight))
    right_gate = torch.sigmoid(torch.nn.functional.linear(x, right_gate_weight))
    out_gate = torch.sigmoid(torch.nn.functional.linear(x, out_gate_weight))

    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5) Core N³ contraction via Triton batched GEMM
    #    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    # ------------------------------------------------------------------
    # Move hidden dimension into the batch axis and transpose right for matmul
    left_perm = left.permute(0, 3, 1, 2).contiguous()           # [B, hidden, N, N]
    right_perm = right.permute(0, 3, 1, 2).contiguous()        # [B, hidden, N, N]
    right_T = right_perm.transpose(-2, -1).contiguous()       # swap the two N axes

    B, H, N, _ = left_perm.shape
    batch_hidden = B * H
    M = N
    K = N

    # Flatten batch*hidden → single batch dimension expected by the kernel
    A = left_perm.view(batch_hidden, M, K)          # (BH, M, K)
    B_ = right_T.view(batch_hidden, K, N)          # (BH, K, N)
    C = torch.empty((batch_hidden, M, N), dtype=torch.float32, device=device)

    # Strides in *elements* (triton expects element‑wise strides)
    stride_ab = A.stride(0)
    stride_am = A.stride(1)
    stride_ak = A.stride(2)

    stride_bb = B_.stride(0)
    stride_bk = B_.stride(1)
    stride_bn = B_.stride(2)

    stride_cb = C.stride(0)
    stride_cm = C.stride(1)
    stride_cn = C.stride(2)

    # Tile sizes – balanced for seq_len up to 1024 on H100
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # One program per (batch_hidden, tile_m, tile_n)
    grid = (batch_hidden * triton.cdiv(N, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    batched_matmul_kernel[grid](
        A,
        B_,
        C,
        batch_hidden,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    torch.cuda.synchronize()

    # Reshape back to [B, N, N, hidden]
    out = C.view(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------------
    # 6) Output layer‑norm, gating and final projection back to dim
    # ------------------------------------------------------------------
    out_norm = torch.nn.functional.layer_norm(
        out,
        (hidden_dim,),
        weight=to_out_norm_weight,
        bias=to_out_norm_bias,
        eps=eps,
    )
    out_scaled = out_norm * out_gate
    out_final = torch.nn.functional.linear(out_scaled, to_out_weight)  # [B, N, N, dim]

    return out_final