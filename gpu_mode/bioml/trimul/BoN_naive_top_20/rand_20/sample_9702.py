"""
tri_mul Triton kernel implementation (outgoing version)

The kernel computes the core "TriMul" operation
    out[b, i, j, h] = Σ_k left[b, i, k, h] * right[b, j, k, h]
for all batches `b` and hidden channels `h`.
It is a batched GEMM where the right operand is implicitly transposed.
The implementation:
* LayerNorm, linear projections, and gating are performed in PyTorch.
* The heavy O(N³·H) triple‑product is fused in a Triton kernel.
* The kernel works on the full `float32` tensors (no precision loss for heavy‑tailed inputs).
* Output of the kernel is fed back to PyTorch for the final LayerNorm, output‑gate
  and output linear projection.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def trimul_kernel(
    # pointers
    a_ptr, b_ptr, c_ptr,
    # problem size
    batch, hidden, N,
    # strides for A (left)  [batch, i, k, h]
    stride_a_batch, stride_a_i, stride_a_k, stride_a_h,
    # strides for B (right) [batch, j, k, h] (will be accessed as B_T[k, j])
    stride_b_batch, stride_b_j, stride_b_k, stride_b_h,
    # strides for C (output) [batch, i, j, h]
    stride_c_batch, stride_c_i, stride_c_j, stride_c_h,
    # compile‑time block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """TriMul kernel: out = left @ rightᵀ for each (batch, hidden) slice."""
    pid_m = tl.program_id(0)            # i‑tile
    pid_n = tl.program_id(1)            # j‑tile
    pid_s = tl.program_id(2)            # combined batch‑hidden index

    # decode batch and hidden index
    batch_idx = pid_s // hidden
    hidden_idx = pid_s % hidden

    # base pointers for this slice
    a_base = a_ptr + batch_idx * stride_a_batch + hidden_idx * stride_a_h
    b_base = b_ptr + batch_idx * stride_b_batch + hidden_idx * stride_b_h
    c_base = c_ptr + batch_idx * stride_c_batch + hidden_idx * stride_c_h

    # tile offsets within the i/j dimensions
    offs_i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # mask for out‑of‑bounds i/j tiles
    mask_i = offs_i < N
    mask_j = offs_j < N
    mask_ij = mask_i[:, None] & mask_j[None, :]

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over the reduction dimension k
    num_k = tl.cdiv(N, BLOCK_K)
    for k_idx in range(num_k):
        offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)

        # masks for the current k‑tile
        mask_k = offs_k < N
        mask_a = mask_i[:, None] & mask_k[None, :]          # (M, K)
        mask_b = mask_k[:, None] & mask_j[None, :]          # (K, N)

        # pointers to the current A and Bᵀ tiles
        a_ptrs = a_base + offs_i[:, None] * stride_a_i + offs_k[None, :] * stride_a_k
        b_ptrs = b_base + offs_k[:, None] * stride_b_k + offs_j[None, :] * stride_b_j  # B_T[k, j]

        # load tiles (float32)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)

        # fused dot‑product
        acc += tl.dot(a, b)   # result is fp32

    # store the result (float32)
    c_ptrs = c_base + offs_i[:, None] * stride_c_i + offs_j[None, :] * stride_c_j
    tl.store(c_ptrs, acc, mask=mask_ij)


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul module using a custom Triton kernel.

    Args:
        data: Tuple (input_tensor, mask, weights, config)
            - input_tensor: [B, N, N, dim] float32
            - mask: [B, N, N] (optional, can be None)
            - weights: dict of model parameters
            - config: dict with at least "dim" and "hidden_dim"

    Returns:
        Tensor of shape [B, N, N, dim] (float32)
    """
    # unpack
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden = config["hidden_dim"]
    B, N, _, _ = input_tensor.shape

    # ------------------------------------------------------------------
    # 1. LayerNorm on the input (dim‑wise)
    # ------------------------------------------------------------------
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
        eps=1e-5,
    )

    # ------------------------------------------------------------------
    # 2. Linear projections (no bias)
    # ------------------------------------------------------------------
    left = F.linear(x, weights["left_proj.weight"])      # [B, N, N, hidden]
    right = F.linear(x, weights["right_proj.weight"])    # [B, N, N, hidden]

    # ------------------------------------------------------------------
    # 3. Gating values (sigmoid)
    # ------------------------------------------------------------------
    left_gate = torch.sigmoid(F.linear(x, weights["left_gate.weight"]))   # [B,N,N,hidden]
    right_gate = torch.sigmoid(F.linear(x, weights["right_gate.weight"]))
    out_gate = torch.sigmoid(F.linear(x, weights["out_gate.weight"]))

    # ------------------------------------------------------------------
    # 4. Optional mask (broadcast on hidden dim)
    # ------------------------------------------------------------------
    if mask is not None and not config.get("nomask", False):
        mask_unsq = mask.unsqueeze(-1).to(x.dtype)                     # [B,N,N,1]
        left = left * mask_unsq
        right = right * mask_unsq
    # ------------------------------------------------------------------
    # 5. Apply element‑wise gates
    # ------------------------------------------------------------------
    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 6. Core TriMul via Triton (float32 for full precision)
    # ------------------------------------------------------------------
    out = torch.empty_like(left)   # allocate result tensor (float32)

    # strides for the three tensors (they share the same layout)
    stride_a_batch, stride_a_i, stride_a_k, stride_a_h = left.stride()
    stride_b_batch, stride_b_j, stride_b_k, stride_b_h = right.stride()
    stride_c_batch, stride_c_i, stride_c_j, stride_c_h = out.stride()

    # Define block sizes – these work well on H100 for fp32 matmuls
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # Grid: (tiles over i, tiles over j, batch*hidden)
    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        B * hidden,
    )

    # Launch Triton kernel
    trimul_kernel[grid](
        left,
        right,
        out,
        B,
        hidden,
        N,
        stride_a_batch,
        stride_a_i,
        stride_a_k,
        stride_a_h,
        stride_b_batch,
        stride_b_j,
        stride_b_k,
        stride_b_h,
        stride_c_batch,
        stride_c_i,
        stride_c_j,
        stride_c_h,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    # Ensure kernel has finished before moving on
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 7. Output LayerNorm (hidden‑wise) and output‑gate
    # ------------------------------------------------------------------
    out = F.layer_norm(
        out,
        (hidden,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
        eps=1e-5,
    )
    out = out * out_gate

    # ------------------------------------------------------------------
    # 8. Final linear projection back to model dimension
    # ------------------------------------------------------------------
    out = F.linear(out, weights["to_out.weight"])   # [B, N, N, dim]

    return out