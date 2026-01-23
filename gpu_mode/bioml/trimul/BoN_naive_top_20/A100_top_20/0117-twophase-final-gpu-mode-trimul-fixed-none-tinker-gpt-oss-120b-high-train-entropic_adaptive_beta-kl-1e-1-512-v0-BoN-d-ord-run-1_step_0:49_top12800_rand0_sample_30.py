"""
Outgoing TriMul (AlphaFold3) forward pass.

The operation works on a 4‑D tensor x ∈ ℝ^{B×N×N×D}:

1. Layer‑norm over the last dimension (D).
2. Linear projections to a hidden dimension H (no bias):
       left  = x·W_left  , right = x·W_right
3. Optional per‑pair mask (broadcast on H).
4. Gate the projections with sigmoid‑gates:
       left  = left  * sigmoid(x·W_left_gate)
       right = right * sigmoid(x·W_right_gate)
5. For each hidden channel h compute
       out[...,h] = left[...,h] @ right[...,h]^T
   i.e. a batch of H independent matrix‑multiplications of size N×N.
   This step is the only part written in Triton – a custom
   batched‑GEMM kernel that tiles the M/N/K dimensions.
6. Layer‑norm over the hidden dimension, multiply by an output gate
   (sigmoid(x·W_out_gate)).
7. Final linear projection back to D dimensions.

All heavy arithmetic is performed in fp16 for speed on H100,
the final tensor is returned in fp32.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr,                      # fp16*   (batch, M, K)
    B_ptr,                      # fp16*   (batch, N, K)
    C_ptr,                      # fp16*   (batch, M, N)
    M, N, K,                    # matrix sizes (int)
    stride_am, stride_ak, stride_ab,   # strides for A (M, K, batch)
    stride_bk, stride_bn, stride_bb,   # strides for B (K, N, batch)
    stride_cm, stride_cn, stride_cb,   # strides for C (M, N, batch)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched GEMM:   C[b] = A[b] @ B[b]^T
    The kernel processes one tile (BLOCK_M×BLOCK_N) of C for a specific
    batch index.  The tiling loops over K in steps of BLOCK_K.
    """
    total_blocks_m = tl.cdiv(M, BLOCK_M)
    total_blocks_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = total_blocks_m * total_blocks_n

    pid = tl.program_id(0)
    batch = pid // tiles_per_batch
    tile_id = pid % tiles_per_batch
    pid_m = tile_id // total_blocks_n
    pid_n = tile_id % total_blocks_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A[tile] : (BLOCK_M, BLOCK_K)
    a_ptrs = A_ptr + batch * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B_T[tile] : (BLOCK_K, BLOCK_N)  (load B as transposed)
    b_ptrs = B_ptr + batch * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + batch * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs,
             acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def custom_kernel(data):
    """
    Outgoing TriMul forward pass using a Triton batched matmul kernel.

    Args:
        data: Tuple (input_tensor, mask, weights, config)
            - input_tensor: torch.Tensor[B, N, N, D] (float32)
            - mask: torch.Tensor[B, N, N] (bool/float, may be ignored)
            - weights: dict of model parameters (float32)
            - config: dict with keys `dim`, `hidden_dim`, `nomask` (bool)

    Returns:
        torch.Tensor[B, N, N, D] (float32)
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    fp16 = torch.float16

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    eps = 1e-5

    # ------------------------------------------------------------------
    # 1️⃣ LayerNorm over the feature dimension (float32 → fp16)
    # ------------------------------------------------------------------
    norm_w = weights["norm.weight"].to(fp16)
    norm_b = weights["norm.bias"].to(fp16)
    # use fp32 for the mean/var computation (more accurate) then cast
    x = torch.nn.functional.layer_norm(
        input_tensor, (dim,), weight=norm_w.to(torch.float32),
        bias=norm_b.to(torch.float32), eps=eps
    ).to(fp16)                     # (B, N, N, D) in fp16

    # ------------------------------------------------------------------
    # 2️⃣ Linear projections (no bias)
    # ------------------------------------------------------------------
    left_proj_w = weights["left_proj.weight"].to(fp16)   # (H, D)
    right_proj_w = weights["right_proj.weight"].to(fp16)

    left = torch.nn.functional.linear(x, left_proj_w)    # (B, N, N, H)
    right = torch.nn.functional.linear(x, right_proj_w)

    # ------------------------------------------------------------------
    # 3️⃣ Optional mask (broadcast on hidden dim)
    # ------------------------------------------------------------------
    if not nomask and mask is not None:
        m = mask.to(fp16).unsqueeze(-1)                # (B, N, N, 1)
        left = left * m
        right = right * m

    # ------------------------------------------------------------------
    # 4️⃣ Gating (sigmoid of linear)
    # ------------------------------------------------------------------
    left_gate_w = weights["left_gate.weight"].to(fp16)
    right_gate_w = weights["right_gate.weight"].to(fp16)
    out_gate_w = weights["out_gate.weight"].to(fp16)

    left_gate = torch.sigmoid(torch.nn.functional.linear(x, left_gate_w))
    right_gate = torch.sigmoid(torch.nn.functional.linear(x, right_gate_w))
    out_gate = torch.sigmoid(torch.nn.functional.linear(x, out_gate_w))

    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5️⃣ Batched MatMul via Triton
    #     Compute for each hidden channel h:
    #         out[..., h] = left[..., h] @ right[..., h]^T
    # ------------------------------------------------------------------
    B, N_seq, _, _ = left.shape            # B, N, N, H
    # bring hidden dim to batch dim: (B, H, N, N) → (B*H, N, N)
    left_t = left.permute(0, 3, 1, 2).contiguous()
    right_t = right.permute(0, 3, 1, 2).contiguous()
    batch_total = B * hidden_dim

    left_flat = left_t.view(batch_total, N_seq, N_seq)
    right_flat = right_t.view(batch_total, N_seq, N_seq)

    out_flat = torch.empty_like(left_flat, dtype=fp16, device=device)

    # Strides (batch, M, K) for A; (batch, N, K) for B; (batch, M, N) for C
    a_ab = left_flat.stride(0)
    a_am = left_flat.stride(1)
    a_ak = left_flat.stride(2)

    b_bb = right_flat.stride(0)
    b_bk = right_flat.stride(2)   # stride of K dimension (inner)
    b_bn = right_flat.stride(1)   # stride of N dimension (outer)

    c_cb = out_flat.stride(0)
    c_cm = out_flat.stride(1)
    c_cn = out_flat.stride(2)

    # Tile sizes – chosen for H100 (fit in shared memory, good occupancy)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid_m = (N_seq + BLOCK_M - 1) // BLOCK_M
    grid_n = (N_seq + BLOCK_N - 1) // BLOCK_N
    total_tiles = batch_total * grid_m * grid_n

    batched_matmul_kernel[(total_tiles,)](
        left_flat,
        right_flat,
        out_flat,
        M=N_seq,
        N=N_seq,
        K=N_seq,
        stride_am=a_am,
        stride_ak=a_ak,
        stride_ab=a_ab,
        stride_bk=b_bk,
        stride_bn=b_bn,
        stride_bb=b_bb,
        stride_cm=c_cm,
        stride_cn=c_cn,
        stride_cb=c_cb,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Restore original shape (B, N, N, H)
    out = out_flat.view(B, hidden_dim, N_seq, N_seq).permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------------
    # 6️⃣ Post‑matmul LayerNorm + output gate
    # ------------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(fp16)
    to_out_norm_b = weights["to_out_norm.bias"].to(fp16)

    out = torch.nn.functional.layer_norm(
        out, (hidden_dim,), weight=to_out_norm_w, bias=to_out_norm_b, eps=eps)

    out = out * out_gate   # broadcast on hidden dim

    # ------------------------------------------------------------------
    # 7️⃣ Final linear projection back to D
    # ------------------------------------------------------------------
    to_out_w = weights["to_out.weight"].to(fp16)  # (D, H)
    out = torch.nn.functional.linear(out, to_out_w)   # (B, N, N, D)

    # Return in float32 to match typical downstream expectations
    return out.to(torch.float32)