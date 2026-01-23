"""
TriMul "outgoing" implementation.

Algorithm
---------
1. Layer‑Normalize the input tensor (shape [B, N, N, C]).
2. Linear projections to a hidden dimension H:
      left  = x @ W_left   (no bias)
      right = x @ W_right
3. Apply the optional pairwise mask.
4. Compute three sigmoid gating vectors from the normalized input:
      left_gate  = σ(x @ W_left_gate)
      right_gate = σ(x @ W_right_gate)
      out_gate   = σ(x @ W_out_gate)
5. Apply left/right gates element‑wise.
6. Rearrange to [B, H, N, N] and cast to fp16.
   Perform a batched matrix‑multiply using a custom Triton kernel:
        out[b, h] = left[b, h] @ right[b, h].T
   (each batch entry corresponds to one (batch, hidden) pair.)
   The kernel uses fp16 inputs, accumulates in fp32 and writes fp16.
7. Cast the result back to fp32 and reshape to [B, N, N, H].
8. Layer‑normalize over the hidden dimension, multiply by `out_gate`,
   and apply the final output linear projection (H → C).
The heavy N³ computation is fully fused in the Triton kernel,
while the surrounding linear layers, LayerNorms and sigmoids stay in PyTorch.

The kernel works for any batch size ≤ 2, sequence length up to 1024,
and hidden dimension up to 384.  It is written for Triton 3.3.1
and tested on NVIDIA H100.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr,                     # fp16 left  [B*H, M, K]
    B_ptr,                     # fp16 right [B*H, K, N]  (actually stored as [B*H, N, K])
    C_ptr,                     # fp16 output [B*H, M, N]
    M, N, K,                   # problem sizes (M = N = K = seq_len)
    stride_am, stride_ak,      # strides for A (row, col)
    stride_bk, stride_bn,      # strides for B (col of B_T, row of B_T)
    stride_cm, stride_cn,      # strides for C (row, col)
    stride_a_batch,            # stride between consecutive batch*H matrices of A
    stride_b_batch,            # stride between consecutive batch*H matrices of B
    stride_c_batch,            # stride between consecutive batch*H matrices of C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    # number of blocks in the M and N dimensions
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)

    # decode flat program id into (batch, block_m, block_n)
    batch = pid // (num_blocks_m * num_blocks_n)
    block = pid % (num_blocks_m * num_blocks_n)
    pid_m = block // num_blocks_n
    pid_n = block % num_blocks_n

    # Compute offsets for the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Pointers with batch offset
        a_ptrs = (
            A_ptr
            + batch * stride_a_batch
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            B_ptr
            + batch * stride_b_batch
            + offs_k[:, None] * stride_bk   # treat B as transposed: (K,N) -> (K,N) using swapped strides
            + offs_n[None, :] * stride_bn
        )

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b)   # fp16·fp16 → fp32 accumulation

    # Write result back (store as fp16)
    c_ptrs = (
        C_ptr
        + batch * stride_c_batch
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.
    Args:
        data: tuple (input, mask, weights, config)
            - input: torch.Tensor of shape [B, N, N, C] (float32)
            - mask:  torch.Tensor of shape [B, N, N] (bool/float) or None
            - weights: dict of model parameters (float32 tensors)
            - config: dict with keys "dim" and "hidden_dim"
    Returns:
        torch.Tensor of shape [B, N, N, C] (float32)
    """
    # unpack arguments
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-5

    # ------------------------------------------------------------------
    # 1. Normalize
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    x = F.layer_norm(input_tensor, (dim,), weight=norm_weight, bias=norm_bias, eps=eps)

    # ------------------------------------------------------------------
    # 2. Linear projections
    # ------------------------------------------------------------------
    left = F.linear(x, weights["left_proj.weight"])      # [B, N, N, H]
    right = F.linear(x, weights["right_proj.weight"])

    # ------------------------------------------------------------------
    # 3. Optional mask (broadcast on the hidden dim)
    # ------------------------------------------------------------------
    if mask is None:
        mask_tensor = torch.ones_like(left[..., :1])
    else:
        mask_tensor = mask.unsqueeze(-1).to(left.dtype)
    left = left * mask_tensor
    right = right * mask_tensor

    # ------------------------------------------------------------------
    # 4. Gating vectors (sigmoids)
    # ------------------------------------------------------------------
    left_gate = F.linear(x, weights["left_gate.weight"]).sigmoid()
    right_gate = F.linear(x, weights["right_gate.weight"]).sigmoid()
    out_gate = F.linear(x, weights["out_gate.weight"]).sigmoid()

    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5. Rearrange for batched matmul and cast to fp16
    # ------------------------------------------------------------------
    # shape -> [B, H, N, N]
    left = left.permute(0, 3, 1, 2).contiguous()
    right = right.permute(0, 3, 1, 2).contiguous()

    left = left.to(torch.float16)
    right = right.to(torch.float16)

    B, H, N, _ = left.shape                     # (batch, hidden, seq_len, seq_len)

    # flatten (B*H) as the batch dimension for the kernel
    left_flat = left.view(-1, N, N)              # [B*H, N, N]
    right_flat = right.view(-1, N, N)            # [B*H, N, N]
    out_flat = torch.empty_like(left_flat)       # fp16 output

    # ------------------------------------------------------------------
    # 6. Launch Triton kernel for batched matmul (A @ B^T)
    # ------------------------------------------------------------------
    # Strides are in elements (not bytes)
    stride_a_batch, stride_am, stride_ak = left_flat.stride()
    stride_b_batch, stride_br, stride_bc = right_flat.stride()
    stride_c_batch, stride_cm, stride_cn = out_flat.stride()

    # Transposed view of B: swap inner strides
    stride_bk = stride_bc   # column stride (inner-most)
    stride_bn = stride_br   # row stride

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    total_batches = left_flat.shape[0]                      # B * H
    grid = lambda META: (total_batches * ((N + META["BLOCK_M"] - 1) // META["BLOCK_M"])
                         * ((N + META["BLOCK_N"] - 1) // META["BLOCK_N"]),)

    batched_matmul_kernel[grid](
        left_flat, right_flat, out_flat,
        N, N, N,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_a_batch, stride_b_batch, stride_c_batch,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # ------------------------------------------------------------------
    # 7. Restore original layout and dtype (fp32)
    # ------------------------------------------------------------------
    out = (
        out_flat.view(B, H, N, N)   # [B, H, N, N]
        .permute(0, 2, 3, 1)       # [B, N, N, H]
        .contiguous()
        .to(torch.float32)
    )

    # ------------------------------------------------------------------
    # 8. Output LayerNorm, gating, final projection
    # ------------------------------------------------------------------
    out_norm = F.layer_norm(
        out,
        (hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
        eps=eps,
    )
    out_norm = out_norm * out_gate
    output = F.linear(out_norm, weights["to_out.weight"])

    return output