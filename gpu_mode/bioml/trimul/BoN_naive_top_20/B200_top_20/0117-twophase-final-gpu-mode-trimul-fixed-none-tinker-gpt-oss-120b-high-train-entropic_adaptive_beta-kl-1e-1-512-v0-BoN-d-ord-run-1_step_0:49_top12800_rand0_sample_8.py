"""
Triton implementation of the “outgoing” TriMul operator (AlphaFold‑3).

Algorithm
---------
1.  Layer‑norm the 4‑D input tensor (float32) – this is cheap compared to the
    N³ computation, so we keep it in float32.
2.  Cast the normalized tensor to float16 and use the provided weights (also
    cast to float16) to compute
        * left  = Linear(x)                → [B, N, N, H]
        * right = Linear(x)                → [B, N, N, H]
        * left_gate  = sigmoid(Linear(x))  → [B, N, N, H]
        * right_gate = sigmoid(Linear(x))  → [B, N, N, H]
        * out_gate   = sigmoid(Linear(x))  → [B, N, N, H]
    (H = hidden_dim)
3.  Apply the optional mask, then element‑wise gate the projections.
4.  The core “TriMul” is the batched matrix product
        out[b,h,i,j] = Σ_k left[b,h,i,k] * right[b,h,j,k] .
    This is exactly a GEMM for each (b,h) pair:
        out_{bh} = left_{bh} @ right_{bh}ᵀ .
    We launch a Triton kernel that processes many (b,h) matrices in parallel,
    each GEMM is performed in FP16 with a FP32 accumulator.
5.  Reshape the result back to [B, N, N, H] (still FP16).
6.  Apply a second Layer‑Norm (over the hidden dimension), multiply by the
    previously computed `out_gate`, and finally a linear projection back to
    the original feature dimension.
7.  The final tensor is returned in float32.

The Triton kernel implements a blocked GEMM with three‑dimensional grid:
    (block_row, block_col, batch_h) where `batch_h = B * H`.
All heavy arithmetic (step 4) runs on the GPU with minimal Python overhead,
while the remaining cheap ops stay in PyTorch for simplicity.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton GEMM kernel used for the TriMul contraction.
# ----------------------------------------------------------------------
@triton.jit
def triton_trimul_gemm(
    a_ptr, b_ptr, c_ptr,               # pointers
    M, N, K,                           # matrix sizes (M=N=N, K=N)
    stride_am, stride_ak,              # A strides: (row, col)
    stride_bk, stride_bn,              # B strides (B is transposed inside)
    stride_cm, stride_cn,              # C strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)               # block row
    pid_n = tl.program_id(1)               # block col
    pid_b = tl.program_id(2)               # batch (B * H)

    # ----- offsets for the current block -----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Size of one matrix (per batch element) in elements
    batch_offset_a = pid_b * M * K          # A and B share the same element count
    batch_offset_c = pid_b * M * N

    a_ptr = a_ptr + batch_offset_a
    b_ptr = b_ptr + batch_offset_a
    c_ptr = c_ptr + batch_offset_c

    # acc in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ----- loop over K dimension in tiles -----
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        offs_k = k_offset + tl.arange(0, BLOCK_K)

        # Masks for OOB (out‑of‑bounds) handling
        mask_k = offs_k < K
        mask_m = offs_m < M
        mask_n = offs_n < N

        # Load A tile (M × K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0)

        # Load B tile (K × N) – note that B is accessed as transposed:
        # original B shape is (batch, N, N); we want Bᵀ so we swap strides.
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0)

        # Accumulate the block product. a and b are fp16, tl.dot promotes to fp32.
        acc += tl.dot(a, b)

    # Write the result back to C (fp16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_c)


# ----------------------------------------------------------------------
# Public entry point used by the evaluation harness.
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.
    Args:
        data: (input_tensor, mask, weights, config)
            - input_tensor: torch.Tensor [B, N, N, dim] (float32)
            - mask        : torch.Tensor [B, N, N] (float32 or bool) – may be None
            - weights     : dict of torch.Tensors (model parameters)
            - config      : dict containing at least:
                * "dim"          – input feature dimension
                * "hidden_dim"   – hidden dimension H
                * "nomask" (bool, optional) – whether to ignore mask
    Returns:
        torch.Tensor of shape [B, N, N, dim] (float32)
    """
    # ------------------------------------------------------------------
    # unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    bs, N, N2, dim = input_tensor.shape
    assert N == N2, "pair representation must be square."
    hidden_dim = config["hidden_dim"]
    eps = 1e-5

    # ------------------------------------------------------------------
    # 1️⃣  Layer‑Norm over the last dimension (float32)
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias   = weights["norm.bias"]
    x_norm_f32 = F.layer_norm(input_tensor,
                               (dim,),
                               weight=norm_weight,
                               bias=norm_bias,
                               eps=eps)                     # [B,N,N,dim] (fp32)

    # ------------------------------------------------------------------
    # 2️⃣  Cast to fp16 and cast all linear weights once
    # ------------------------------------------------------------------
    dtype = torch.float16
    x = x_norm_f32.to(dtype)                              # [B,N,N,dim] (fp16)

    # Linear weights – fp16 for speed
    left_proj_w   = weights["left_proj.weight"].to(dtype)
    right_proj_w  = weights["right_proj.weight"].to(dtype)
    left_gate_w   = weights["left_gate.weight"].to(dtype)
    right_gate_w  = weights["right_gate.weight"].to(dtype)
    out_gate_w    = weights["out_gate.weight"].to(dtype)
    to_out_norm_w = weights["to_out_norm.weight"].to(dtype)
    to_out_norm_b = weights.get("to_out_norm.bias")
    if to_out_norm_b is not None:
        to_out_norm_b = to_out_norm_b.to(dtype)
    to_out_w      = weights["to_out.weight"].to(dtype)   # [dim, hidden_dim]

    # ------------------------------------------------------------------
    # 3️⃣  Project and gate
    # ------------------------------------------------------------------
    # Linear projections
    left  = F.linear(x, left_proj_w)     # [B,N,N,hidden_dim] (fp16)
    right = F.linear(x, right_proj_w)

    # Optional mask (broadcasted on hidden dimension)
    if not config.get("nomask", True) and mask is not None:
        mask_f = mask.to(dtype).unsqueeze(-1)           # [B,N,N,1]
        left  = left * mask_f
        right = right * mask_f

    # Gating sigmoids
    left_gate  = torch.sigmoid(F.linear(x, left_gate_w))
    right_gate = torch.sigmoid(F.linear(x, right_gate_w))
    out_gate   = torch.sigmoid(F.linear(x, out_gate_w))

    # Apply gates
    left  = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 4️⃣  Triton GEMM: out = Σ_k left[...,i,k,:] * right[...,j,k,:]
    #    (batched mat‑mul per hidden channel)
    # ------------------------------------------------------------------
    # reshape to [B, H, N, N] then merge B×H as the batch dimension of the kernel
    left_perm  = left.permute(0, 3, 1, 2).contiguous()   # [B,H,N,N]
    right_perm = right.permute(0, 3, 1, 2).contiguous()
    batch_h = bs * hidden_dim
    left_flat  = left_perm.view(batch_h, N, N)           # [B*H, N, N]
    right_flat = right_perm.view(batch_h, N, N)

    out_flat = torch.empty_like(left_flat)               # fp16 output placeholder

    # Strides (int64 for Triton)
    stride_am = left_flat.stride(1)      # row stride = N
    stride_ak = left_flat.stride(2)      # col stride = 1
    stride_bk = right_flat.stride(2)     # 1  (after transposition we treat this as row stride)
    stride_bn = right_flat.stride(1)     # N
    stride_cm = out_flat.stride(1)       # N
    stride_cn = out_flat.stride(2)       # 1

    # Block sizes – tuned for H100 (fits in shared memory & registers)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(N, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
            batch_h)

    triton_trimul_gemm[grid](
        a_ptr=left_flat,
        b_ptr=right_flat,
        c_ptr=out_flat,
        M=N, N=N, K=N,
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        stride_bn=stride_bn,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # reshape back to [B,N,N,hidden_dim] (still fp16)
    out = out_flat.view(bs, hidden_dim, N, N).permute(0, 2, 3, 1)   # [B,N,N,H]

    # ------------------------------------------------------------------
    # 5️⃣  Final LayerNorm + out‑gate + linear projection back to dim
    # ------------------------------------------------------------------
    out = F.layer_norm(out,
                       (hidden_dim,),
                       weight=to_out_norm_w,
                       bias=to_out_norm_b,
                       eps=eps)                                   # fp16

    out = out * out_gate                                       # fp16, broadcast

    out = F.linear(out, to_out_w)                              # fp16 → [B,N,N,dim]

    # Return in float32 (as the reference implementation does)
    return out.to(torch.float32)