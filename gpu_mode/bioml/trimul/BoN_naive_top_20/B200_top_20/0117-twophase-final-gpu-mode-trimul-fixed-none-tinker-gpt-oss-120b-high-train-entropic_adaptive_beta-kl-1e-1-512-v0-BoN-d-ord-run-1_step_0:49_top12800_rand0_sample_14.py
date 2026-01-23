"""
TriMul “outgoing” forward pass (AlphaFold3).

Algorithm
---------
1. Layer‑norm the input tensor  X ∈ (B,N,N,D).
2. Project X → left, right  (D → H) and compute sigmoid gates.
3. Apply mask (if provided) and gates to left/right.
4. Compute the core contraction
        out[i,j,:] = Σ_k left[i,k,:] * right[j,k,:] .
   This is a batched matrix multiplication:
        left (B,H,N,N) ‑► (B*H,N,N)  @  rightᵀ (B*H,N,N) → out (B*H,N,N)
   The heavy work is performed by a custom Triton kernel that
   multiplies a batch of (M×K) @ (K×N) matrices (here M=N=K).
5. Layer‑norm the result (over the hidden dimension), apply the
   output gate, and a final linear projection back to D.
All heavy arithmetic is performed in float16 (FP16) with accumulation in
float32 for accuracy; the final output is returned as float32.

The Triton kernel implements a generic batched GEMM and is launched with
grid dimensions (⌈N/BM⌉, ⌈N/BN⌉, B*H) where BM,BN are tile sizes.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------------------
# Triton batched GEMM kernel (A @ Bᵀ) with FP16 inputs.
# --------------------------------------------------------------
@triton.jit
def batched_matmul_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Matrix sizes (M=N, N=N, K=N)
    M, N, K,
    # Strides for A: (batch, row i, col k)
    stride_az, stride_ai, stride_ak,
    # Strides for B: (batch, row j, col k) – note we load B as (k, j)
    stride_bz, stride_bj, stride_bk,
    # Strides for C: (batch, row i, col j)
    stride_cz, stride_ci, stride_cj,
    # Tile sizes (compile‑time constants)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_batch = tl.program_id(2)          # which (b * H) matrix
    pid_m = tl.program_id(0)              # tile along M dimension (i)
    pid_n = tl.program_id(1)              # tile along N dimension (j)

    # ---- tile offsets ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ---- bounds ----
    mask_m = offs_m < M
    mask_n = offs_n < N

    # ---- accumulator (FP32) ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main reduction loop over K
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in range(0, num_k_tiles):
        cur_k = k_tile * BLOCK_K

        # Pointers to the current K‑slice
        a_ptrs = (
            a_ptr
            + pid_batch * stride_az
            + offs_m[:, None] * stride_ai
            + (cur_k + offs_k)[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + pid_batch * stride_bz
            + (cur_k + offs_k)[:, None] * stride_bk   # K dimension first
            + offs_n[None, :] * stride_bj              # j dimension second
        )

        # Load with masks for out‑of‑bounds elements
        a_mask = mask_m[:, None] & ((cur_k + offs_k) < K)[None, :]
        b_mask = ((cur_k + offs_k) < K)[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)   # (BLOCK_M, BLOCK_K) FP16
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)   # (BLOCK_K, BLOCK_N) FP16

        # Accumulate in FP32 (automatic up‑cast)
        acc += tl.dot(a, b)

    # ---- write back result (FP16) ----
    c_ptrs = (
        c_ptr
        + pid_batch * stride_cz
        + offs_m[:, None] * stride_ci
        + offs_n[None, :] * stride_cj
    )
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=store_mask)


# --------------------------------------------------------------
# Python entry point
# --------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.
    Arguments
    ---------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor [B, N, N, D] (float32)
        - mask         : torch.Tensor [B, N, N]   (float32 or bool)
        - weights      : dict of torch.Tensor parameters
        - config       : dict containing "dim" and "hidden_dim"
    Returns
    -------
    out : torch.Tensor [B, N, N, D] (float32)
    """
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dtype = torch.float16    # internal compute dtype

    # --------------------- config ---------------------
    dim = config["dim"]
    hidden = config["hidden_dim"]
    nomask = config.get("nomask", True)

    # --------------------- layer norm ---------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=1e-5,
    )  # [B,N,N,D] float32

    # Cast to half for the expensive ops
    x_h = x.to(dtype)

    # --------------------- linear projections ---------------------
    left_proj_w = weights["left_proj.weight"].to(dtype)
    right_proj_w = weights["right_proj.weight"].to(dtype)
    left_gate_w = weights["left_gate.weight"].to(dtype)
    right_gate_w = weights["right_gate.weight"].to(dtype)
    out_gate_w = weights["out_gate.weight"].to(dtype)

    left = F.linear(x_h, left_proj_w)          # [B,N,N,hidden]
    right = F.linear(x_h, right_proj_w)        # [B,N,N,hidden]

    # --------------------- optional mask ---------------------
    if not nomask:
        mask_ = mask.unsqueeze(-1).to(dtype)   # broadcast over hidden
        left = left * mask_
        right = right * mask_

    # --------------------- gating ---------------------
    left_gate = torch.sigmoid(F.linear(x_h, left_gate_w))
    right_gate = torch.sigmoid(F.linear(x_h, right_gate_w))
    out_gate = torch.sigmoid(F.linear(x_h, out_gate_w))

    left = left * left_gate
    right = right * right_gate

    # --------------------- batched GEMM (core TriMul) ---------------------
    B, N, _, _ = left.shape  # B, seq_len, seq_len, hidden

    # Rearrange to (B, hidden, N, N) then flatten batch*hidden
    left_perm = left.permute(0, 3, 1, 2).contiguous()   # [B, hidden, N, N]
    right_perm = right.permute(0, 3, 1, 2).contiguous()

    batch = B * hidden
    left_flat = left_perm.view(batch, N, N)   # [batch, M=N, K=N] FP16
    right_flat = right_perm.view(batch, N, N) # same layout; we will treat as (batch, N, K)

    # Allocate output buffer
    out_flat = torch.empty_like(left_flat)    # FP16

    # Strides (must be int64)
    stride_az, stride_ai, stride_ak = left_flat.stride()
    stride_bz, stride_bj, stride_bk = right_flat.stride()
    stride_cz, stride_ci, stride_cj = out_flat.stride()

    # Tile sizes – chosen to fit in shared memory on H100
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (
        triton.cdiv(N, BLOCK_M),    # M‑tiles
        triton.cdiv(N, BLOCK_N),    # N‑tiles
        batch,                      # one program per (batch * hidden) matrix
    )

    batched_matmul_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        N, N, N,
        stride_az, stride_ai, stride_ak,
        stride_bz, stride_bj, stride_bk,
        stride_cz, stride_ci, stride_cj,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    # Ensure kernel completed before using the result
    torch.cuda.synchronize()

    # Reshape back to [B,N,N,hidden]
    out = (
        out_flat.view(B, hidden, N, N)
        .permute(0, 2, 3, 1)
        .contiguous()
    )  # [B,N,N,hidden]

    # --------------------- post‑norm and final projection ---------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(dtype)
    to_out_norm_b = weights["to_out_norm.bias"].to(dtype)
    out = F.layer_norm(
        out,
        (hidden,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=1e-5,
    )  # still half

    out = out * out_gate  # gate after norm

    to_out_w = weights["to_out.weight"].to(dtype)   # (hidden, dim)
    out = F.linear(out, to_out_w)                   # [B,N,N,dim] half

    # Cast back to float32 for downstream consumption
    return out.float()