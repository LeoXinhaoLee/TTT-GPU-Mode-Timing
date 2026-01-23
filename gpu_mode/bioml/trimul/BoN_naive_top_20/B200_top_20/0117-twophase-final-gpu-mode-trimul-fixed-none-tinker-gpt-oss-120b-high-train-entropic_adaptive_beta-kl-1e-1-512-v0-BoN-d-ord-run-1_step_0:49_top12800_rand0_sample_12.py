"""
TriMul “outgoing” kernel (AlphaFold3 style) – forward pass only.

Algorithm
---------

1. Layer‑norm the input tensor `x` over the channel dimension `dim`.
2. Compute two linear projections (`left_proj`, `right_proj`) from `dim → hidden_dim`.
   The projections are performed in fp16 for speed.
3. If a mask is supplied (nomask == False) broadcast it and zero‑out the
   projected tensors.
4. Compute three sigmoid gates (`left_gate`, `right_gate`, `out_gate`) with a
   linear layer + `.sigmoid()`.  Gating is fused with the projections:
   `left = left * left_gate`, `right = right * right_gate`.
5. The expensive contraction  
   `out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]`
   is a batched matrix multiplication `C = A @ Bᵀ` where
   `A = left.permute(0,3,1,2)`  (shape B×H×N×K) and
   `B = right.permute(0,3,2,1)` (shape B×H×K×N).
   A custom Triton kernel computes all `B*hidden_dim` independent GEMMs
   in fp16, accumulating in fp32.
6. Layer‑norm the result over the hidden dimension, multiply by `out_gate`,
   and apply a final linear projection `hidden_dim → dim`.
7. Return a float32 tensor of shape `[B, N, N, dim]`.

Only the N³ contraction (step 5) is executed in Triton; all other
operations use PyTorch’s highly‑optimized kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr,                     # *float16
    B_ptr,                     # *float16
    C_ptr,                     # *float16
    M, N, K,                   # matrix sizes (int32)
    stride_ah, stride_am, stride_ak,   # strides of A (int64)
    stride_bh, stride_bk, stride_bn,   # strides of B (int64)
    stride_ch, stride_cm, stride_cn,   # strides of C (int64)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched GEMM: C = A @ Bᵀ  for many independent batches."""
    pid_h = tl.program_id(2)          # batch * head dimension
    pid_m = tl.program_id(0)          # block row
    pid_n = tl.program_id(1)          # block column

    # ------------------------------------------------------------------
    # offsets within the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # linear offsets for the three pointers
    a_ptrs = A_ptr + pid_h * stride_ah + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + pid_h * stride_bh + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------------------------------------------------
    # loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # global offset of the current K‑block
        k_off = k * BLOCK_K

        # masks to avoid out‑of‑bounds reads
        mask_a = (offs_m[:, None] < M) & ((k_off + offs_k)[None, :] < K)
        mask_b = ((k_off + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)      # (BLOCK_M, BLOCK_K)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)      # (BLOCK_K, BLOCK_N)

        # dot‑product (fp16 × fp16 → fp32 accumulation)
        acc += tl.dot(a, b)

        # move pointers to the next K‑block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ------------------------------------------------------------------
    # write result
    c_ptrs = C_ptr + pid_h * stride_ch + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_c)


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.
    Args:
        data: Tuple (input_tensor, mask_tensor, weights_dict, config_dict)
    Returns:
        Tensor of shape [B, N, N, dim] (float32)
    """
    # ------------------------------------------------------------------
    # unpack arguments
    input_tensor, mask_tensor, weights, config = data
    device = input_tensor.device
    dtype_compute = torch.float16

    # configuration
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 1. LayerNorm over the channel dimension
    eps = 1e-5
    norm_weight = weights["norm.weight"].to(input_tensor.dtype)
    norm_bias   = weights["norm.bias"].to(input_tensor.dtype)
    x_norm = F.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
    )   # float32

    # cast to fp16 for the heavy compute
    x_h = x_norm.to(dtype_compute)

    # ------------------------------------------------------------------
    # 2. Linear projections (dim → hidden_dim) – fp16
    left_proj_w = weights["left_proj.weight"].to(dtype_compute)   # (hidden_dim, dim)
    right_proj_w = weights["right_proj.weight"].to(dtype_compute)

    left = F.linear(x_h, left_proj_w)        # (B, N, N, hidden_dim)
    right = F.linear(x_h, right_proj_w)

    # ------------------------------------------------------------------
    # 3. Optional mask (broadcast over hidden_dim)
    if not nomask and mask_tensor is not None:
        mask_h = mask_tensor.to(dtype_compute).unsqueeze(-1)   # (B, N, N, 1)
        left = left * mask_h
        right = right * mask_h

    # ------------------------------------------------------------------
    # 4. Gating
    left_gate_w  = weights["left_gate.weight"].to(dtype_compute)
    right_gate_w = weights["right_gate.weight"].to(dtype_compute)
    out_gate_w   = weights["out_gate.weight"].to(dtype_compute)

    left_gate  = F.linear(x_h, left_gate_w).sigmoid()
    right_gate = F.linear(x_h, right_gate_w).sigmoid()
    out_gate   = F.linear(x_h, out_gate_w).sigmoid()

    left  = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5. Batched matmul: out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    #    Rearrange tensors to (batch*head, M, K) and (batch*head, K, N)
    B, N, _, _ = left.shape
    # (B, hidden_dim, N, N) – make contiguous for view
    left_perm  = left.permute(0, 3, 1, 2).contiguous()   # (B, H, i, k)
    right_perm = right.permute(0, 3, 2, 1).contiguous()  # (B, H, k, j)

    BH = B * hidden_dim
    A = left_perm.view(BH, N, N)      # (BH, M, K)
    B_mat = right_perm.view(BH, N, N)  # (BH, K, N)

    # Allocate output
    C = torch.empty_like(A, dtype=dtype_compute, device=device)  # (BH, M, N)

    # Triton launch configuration
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        BH,
    )

    # Strides (int64)
    stride_ah, stride_am, stride_ak = A.stride()
    stride_bh, stride_bk, stride_bn = B_mat.stride()
    stride_ch, stride_cm, stride_cn = C.stride()

    batched_matmul_kernel[grid](
        A,
        B_mat,
        C,
        N,               # M
        N,               # N
        N,               # K
        stride_ah,
        stride_am,
        stride_ak,
        stride_bh,
        stride_bk,
        stride_bn,
        stride_ch,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Reshape back to (B, N, N, hidden_dim)
    out = C.view(B, hidden_dim, N, N).permute(0, 2, 3, 1)   # (B, N, N, hidden_dim)

    # ------------------------------------------------------------------
    # 6. Output LayerNorm + out_gate
    to_out_norm_w = weights["to_out_norm.weight"].to(dtype_compute)
    to_out_norm_b = weights["to_out_norm.bias"].to(dtype_compute)

    out = F.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=eps,
    )
    out = out * out_gate  # element‑wise gating

    # ------------------------------------------------------------------
    # 7. Final linear projection (hidden_dim → dim)
    to_out_w = weights["to_out.weight"].to(dtype_compute)
    out = F.linear(out, to_out_w)   # (B, N, N, dim)

    # Return float32 as mandated
    return out.to(torch.float32)