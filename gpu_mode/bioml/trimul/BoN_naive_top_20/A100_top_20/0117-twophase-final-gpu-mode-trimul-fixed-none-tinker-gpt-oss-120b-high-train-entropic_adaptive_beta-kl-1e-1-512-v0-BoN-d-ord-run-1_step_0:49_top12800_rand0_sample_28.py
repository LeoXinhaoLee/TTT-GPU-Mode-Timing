"""
TriMul “outgoing” forward pass implemented with a custom Triton GEMM kernel.

Algorithm
---------
1. Layer‑Norm the input tensor over the last channel dimension.
2. Compute three linear projections of the normalized tensor:
   * left_proj  : dim → hidden_dim
   * right_proj : dim → hidden_dim
   * left_gate/right_gate/out_gate : dim → hidden_dim (followed by σ)
3. Apply the optional pair‑mask and multiply the projections by their gates.
   (mask is binary; when ``nomask`` is True the mask is skipped.)
4. The core O(N³·hidden_dim) contraction
        out_{i,j,d} = Σ_k left_{i,k,d} * right_{j,k,d}
   is performed as a batched matrix multiplication:
        for each (batch, d)   →   out_{d} = left_{d} @ right_{d}ᵀ .
   A custom Triton kernel computes the batched GEMM in half‑precision
   with a float32 accumulator and writes the result in FP16.
5. Layer‑Norm the hidden‑dim output, apply ``out_gate`` and a final
   linear projection back to ``dim``.
6. Cast the final tensor back to float32 (the model convention).

Only the heavy contraction (step 4) is written in Triton; all other
operations use fast PyTorch kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _tri_mul_gemm_kernel(
    a_ptr, b_ptr, c_ptr,                     # pointers
    M, N, K,                                 # matrix sizes (all = seq_len)
    stride_am, stride_ak,                    # A: row‑stride, col‑stride
    stride_bk, stride_bn,                    # B (original) : col‑stride, row‑stride
    stride_cm, stride_cn,                    # C: row‑stride, col‑stride
    batch_stride_a, batch_stride_b, batch_stride_c,  # stride between batches
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched GEMM:
        C[b, i, j] = Σ_k A[b, i, k] * B[b, j, k]   (notice B’s k‑axis is the second one)
    """
    pid_m = tl.program_id(0)                # block row
    pid_n = tl.program_id(1)                # block col
    pid_b = tl.program_id(2)                # batch index (B*hidden_dim)

    # ----------- compute offsets ----------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # ----- base pointers for this batch -----
    a_batch = a_ptr + pid_b * batch_stride_a
    b_batch = b_ptr + pid_b * batch_stride_b
    c_batch = c_ptr + pid_b * batch_stride_c

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension in tiles
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_k_tiles):
        cur_k = k * BLOCK_K + tl.arange(0, BLOCK_K)            # (BLOCK_K,)
        mask_k = cur_k < K

        # Load A tile  (M × K)
        a_ptrs = a_batch + (offs_m[:, None] * stride_am
                             + cur_k[None, :] * stride_ak)
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0)

        # Load B transposed tile (K × N)
        # B is stored as (row=j, col=k); we need Bᵀ[k, j] = B[j, k]
        b_ptrs = b_batch + (cur_k[:, None] * stride_bk
                             + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0)

        # Accumulate
        acc += tl.dot(a, b)          # (M,N) += (M,K)*(K,N)

    # Write result
    c_ptrs = c_batch + (offs_m[:, None] * stride_cm
                        + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def custom_kernel(data):
    """
    Forward pass of the ``TriMul`` outgoing operator.

    Arguments
    ---------
    data : tuple
        (input_tensor, mask_tensor, weights_dict, config_dict)

    Returns
    -------
    torch.Tensor
        Tensor of shape [bs, seq_len, seq_len, dim] (float32)
    """
    # unpack
    input_tensor, mask_tensor, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    device = input_tensor.device

    # ------------------------------------------------------------------
    # 1. Layer‑norm over the last dimension
    # ------------------------------------------------------------------
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
    )

    # Cast to half for the heavy path
    x_h = x.to(torch.float16)

    # ------------------------------------------------------------------
    # 2. Linear projections + gates (all bias‑free)
    # ------------------------------------------------------------------
    left = F.linear(x_h, weights["left_proj.weight"].to(torch.float16))
    right = F.linear(x_h, weights["right_proj.weight"].to(torch.float16))

    left_gate = torch.sigmoid(
        F.linear(x_h, weights["left_gate.weight"].to(torch.float16))
    )
    right_gate = torch.sigmoid(
        F.linear(x_h, weights["right_gate.weight"].to(torch.float16))
    )
    out_gate = torch.sigmoid(
        F.linear(x_h, weights["out_gate.weight"].to(torch.float16))
    )

    # ------------------------------------------------------------------
    # 3. Optional mask
    # ------------------------------------------------------------------
    if not nomask:
        # mask shape: [B, N, N] -> [B, N, N, 1]
        mask = mask_tensor.to(torch.float16).unsqueeze(-1)
        left = left * mask
        right = right * mask

    # ------------------------------------------------------------------
    # 4. Apply element‑wise gates
    # ------------------------------------------------------------------
    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5. Contract over the third axis using a Triton batched GEMM.
    #    We regroup the hidden dimension into the batch axis:
    #        (B, N, N, H) -> (B*H, N, N)
    # ------------------------------------------------------------------
    B, N, _, H = left.shape  # (B, N, N, hidden_dim)

    # reshape to (B, H, N, N) and make contiguous for easy view()
    left = left.permute(0, 3, 1, 2).contiguous()
    right = right.permute(0, 3, 1, 2).contiguous()

    batch = B * H
    left_flat = left.view(batch, N, N)          # (B*H, N, N)
    right_flat = right.view(batch, N, N)

    # output buffer
    out_flat = torch.empty_like(left_flat, dtype=torch.float16, device=device)

    # Triton block sizes – tuned for H100
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        ( (N + BLOCK_M - 1) // BLOCK_M,
          (N + BLOCK_N - 1) // BLOCK_N,
          batch )
    )

    _tri_mul_gemm_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        N,               # M
        N,               # N
        N,               # K
        left_flat.stride(1),   # stride_am (row stride)
        left_flat.stride(2),   # stride_ak (col stride)
        right_flat.stride(2),  # stride_bk  (col stride of original right)
        right_flat.stride(1),  # stride_bn  (row stride of original right)
        out_flat.stride(1),    # stride_cm
        out_flat.stride(2),    # stride_cn
        left_flat.stride(0),   # batch stride A
        right_flat.stride(0),  # batch stride B
        out_flat.stride(0),    # batch stride C
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )

    # reshape back to (B, N, N, H)
    out = out_flat.view(B, H, N, N).permute(0, 2, 3, 1)  # (B, N, N, hidden_dim)

    # ------------------------------------------------------------------
    # 6. Post‑processing: hidden‑dim layer‑norm, out‑gate, final linear
    # ------------------------------------------------------------------
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weight=weights["to_out_norm.weight"].to(torch.float16),
        bias=weights["to_out_norm.bias"].to(torch.float16),
    )
    out = out * out_gate                                 # element‑wise gating
    out = F.linear(out, weights["to_out.weight"].to(torch.float16))

    # Return in the original (float32) dtype
    return out.float()