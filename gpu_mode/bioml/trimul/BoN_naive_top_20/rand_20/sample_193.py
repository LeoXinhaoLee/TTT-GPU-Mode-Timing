"""
TriMul “outgoing” forward pass implemented with Triton.

Algorithm
---------
1. Layer‑norm `x` over the last dimension (dim) using the provided
   `norm.weight` / `norm.bias`.
2. Project `x` to the hidden dimension (hidden_dim) with two linear layers
   (`left_proj`, `right_proj`) and compute two gated versions of the
   projections (`left_gate`, `right_gate`).  All linear operations are
   performed in FP16 to exploit Tensor‑cores.
3. Apply the optional pair‑wise mask (if supplied) and the gates:
        left  = (x @ left_proj.W) * left_gate * mask
        right = (x @ right_proj.W) * right_gate * mask
4. Compute the core TriMul operation:
        out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
   This is a batched matrix multiplication of shape
        (B·hidden_dim, N, N) × (B·hidden_dim, N, N)^T .
   The heavy N³ work is performed by a custom Triton kernel
   (`trimul_einsum_kernel`).  The kernel loads FP16 tiles,
   accumulates in FP32 and writes the result in FP32.
5. Layer‑norm the result over the hidden dimension
   (`to_out_norm.weight` / `to_out_norm.bias`), multiply by the output
   gate (`out_gate = sigmoid(x @ out_gate.W)`), and finally project back
   to the original dimension with the `to_out` linear layer.
6. Return the tensor of shape `[bs, seq_len, seq_len, dim]` (float32).

The implementation keeps the memory layout contiguous and fuses as much as
possible while keeping the code easy to read and verify.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------------- #
# Triton kernel for the N³ “einsum” part of TriMul:
# out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
# left  : (total, M=N, K=N)  – FP16
# right : (total, K=N, N)    – FP16 (the K‑N axes are transposed on the fly)
# out   : (total, M=N, N)    – FP32
# -----------------------------------------------------------------------------
@triton.jit
def trimul_einsum_kernel(
    A,                # left   pointer   (FP16)
    B,                # right  pointer   (FP16) – accessed transposed
    C,                # output pointer   (FP32)
    M, N, K,          # matrix sizes (all equal to seq_len)
    stride_a_batch, stride_a_row, stride_a_col,   # A strides (total, M, K)
    stride_b_batch, stride_b_row, stride_b_col,   # B strides (total, K, N) – transposed view
    stride_c_batch, stride_c_row, stride_c_col,   # C strides (total, M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)          # block index in M dimension (i)
    pid_n = tl.program_id(1)          # block index in N dimension (j)
    pid_b = tl.program_id(2)          # combined batch * hidden index

    # --------------------- start indices ---------------------
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    offs_k = 0

    # Batch / hidden offset
    a_batch_off = pid_b * stride_a_batch
    b_batch_off = pid_b * stride_b_batch
    c_batch_off = pid_b * stride_c_batch

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in tiles
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        cur_k = offs_k + _k * BLOCK_K

        # ----- load tile from A (shape BLOCK_M x BLOCK_K) -----
        a_i = offs_m + tl.arange(0, BLOCK_M)
        a_k = cur_k + tl.arange(0, BLOCK_K)
        a_ptr = A + a_batch_off \
                + a_i[:, None] * stride_a_row \
                + a_k[None, :] * stride_a_col
        a_mask = (a_i[:, None] < M) & (a_k[None, :] < K)
        a = tl.load(a_ptr, mask=a_mask, other=0.0).to(tl.float32)

        # ----- load tile from B transposed (shape BLOCK_K x BLOCK_N) -----
        # we treat B as (K, N) on the fly: rows = k, cols = j
        b_k = cur_k + tl.arange(0, BLOCK_K)
        b_j = offs_n + tl.arange(0, BLOCK_N)
        b_ptr = B + b_batch_off \
                + b_k[:, None] * stride_b_row \
                + b_j[None, :] * stride_b_col
        b_mask = (b_k[:, None] < K) & (b_j[None, :] < N)
        b = tl.load(b_ptr, mask=b_mask, other=0.0).to(tl.float32)

        # ----- matrix multiplication of the current tile -----
        acc += tl.dot(a, b)

    # -------------------- write the result --------------------
    c_i = offs_m + tl.arange(0, BLOCK_M)
    c_j = offs_n + tl.arange(0, BLOCK_N)
    c_ptr = C + c_batch_off \
            + c_i[:, None] * stride_c_row \
            + c_j[None, :] * stride_c_col
    c_mask = (c_i[:, None] < M) & (c_j[None, :] < N)
    tl.store(c_ptr, acc, mask=c_mask)


# ----------------------------------------------------------------------------- #
def custom_kernel(data):
    """
    Forward pass for the outgoing TriMul operator.

    Args:
        data: tuple (input_tensor, mask, weights, config)
            input_tensor : torch.Tensor [bs, N, N, dim] (float32)
            mask         : torch.Tensor [bs, N, N] or None
            weights      : dict of model weights (float32)
            config       : dict with keys "dim", "hidden_dim", possibly "nomask"

    Returns:
        torch.Tensor of shape [bs, N, N, dim] (float32)
    """
    # Unpack arguments
    input_tensor, mask, weights, config = data
    bs, N, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]

    ###########################################################################
    # 1. Input LayerNorm
    ###########################################################################
    # LayerNorm over the last dimension (dim)
    x = F.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
        eps=1e-5,
    )  # [bs, N, N, dim]  (float32)

    ###########################################################################
    # 2. Project to hidden_dim (FP16) and compute gates
    ###########################################################################
    # Cast to half for the heavy linear ops
    x_f16 = x.half()

    # Linear projections (no bias)
    left_proj_weight = weights["left_proj.weight"].to(torch.float16)
    right_proj_weight = weights["right_proj.weight"].to(torch.float16)
    left = F.linear(x_f16, left_proj_weight)    # [bs, N, N, hidden_dim]
    right = F.linear(x_f16, right_proj_weight)  # [bs, N, N, hidden_dim]

    # Gates (sigmoid)
    left_gate_weight = weights["left_gate.weight"].to(torch.float16)
    right_gate_weight = weights["right_gate.weight"].to(torch.float16)
    out_gate_weight = weights["out_gate.weight"].to(torch.float32)  # used later in FP32

    left_gate = F.linear(x_f16, left_gate_weight).sigmoid()   # half
    right_gate = F.linear(x_f16, right_gate_weight).sigmoid() # half
    out_gate = F.linear(x, out_gate_weight).sigmoid()        # float32

    # Apply mask if present
    if mask is not None and mask.numel() > 0:
        # mask is [bs, N, N] -> broadcast to hidden_dim
        mask_f16 = mask.to(torch.float16).unsqueeze(-1)  # [bs, N, N, 1]
        left = left * mask_f16
        right = right * mask_f16

    # Apply gates
    left = left * left_gate
    right = right * right_gate

    ###########################################################################
    # 3. Core TriMul: out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    ###########################################################################
    # Rearrange for batched GEMM: (bs, hidden_dim, N, N)
    left = left.permute(0, 3, 1, 2).contiguous()   # [bs, hidden_dim, N, N]
    right = right.permute(0, 3, 1, 2).contiguous() # [bs, hidden_dim, N, N]

    total = bs * hidden_dim  # flattened batch*hidden dimension
    left_flat = left.view(total, N, N)   # FP16
    right_flat = right.view(total, N, N) # FP16

    # Output tensor (FP32)
    out_flat = torch.empty((total, N, N), dtype=torch.float32, device=input_tensor.device)

    # Kernel launch configuration
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        total,
    )

    # Strides (in element counts, not bytes)
    stride_a_batch, stride_a_row, stride_a_col = left_flat.stride()
    stride_b_batch, stride_b_row, stride_b_col = right_flat.stride()
    # For the transposed view of B we need row stride = stride over K (original col) = stride_b_col
    # and column stride = stride over N (original row) = stride_b_row
    # Since right_flat is contiguous, stride_b_row = N, stride_b_col = 1.
    trisilum_einsum_kernel = trimul_einsum_kernel
    trisilum_einsum_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        N, N, N,
        stride_a_batch, stride_a_row, stride_a_col,
        stride_b_batch, 1, N,                # B transposed: row stride=1, col stride=N
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    # Reshape back: [bs, hidden_dim, N, N] -> [bs, N, N, hidden_dim]
    out = out_flat.view(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()  # float32

    ###########################################################################
    # 4. Output LayerNorm, gate, and final projection
    ###########################################################################
    out = F.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
        eps=1e-5,
    )  # [bs, N, N, hidden_dim]

    # Apply output gate (broadcast over the hidden dimension)
    out = out * out_gate

    # Final linear projection back to dim (float32)
    to_out_weight = weights["to_out.weight"]  # shape [dim, hidden_dim]
    output = F.linear(out, to_out_weight)    # [bs, N, N, dim]

    return output