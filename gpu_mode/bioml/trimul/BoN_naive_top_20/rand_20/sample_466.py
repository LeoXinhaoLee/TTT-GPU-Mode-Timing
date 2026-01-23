"""
TriMul (outgoing) – Triton‑accelerated implementation.

The module performs the following steps (as in the AlphaFold3 reference):
1. Layer‑norm on the input tensor.
2. Two linear projections (left/right) from `dim` → `hidden_dim`.
3. Optional masking of the projected tensors.
4. Gating (sigmoid) on the original normalized tensor to modulate left/right.
5. Core “tri‑multiplicative” contraction:
       out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
   This is a batched matrix‑multiplication of shape
       (B·hidden_dim, N, N) × (B·hidden_dim, N, N)ᵀ,
   implemented in a custom Triton kernel (`batched_matmul_transpose_kernel`).
6. Layer‑norm on the hidden dimension, multiplication by an output gate,
   and a final linear projection back to `dim`.

Only the heavy N³ contraction (step 5) is executed on the GPU via Triton;
all other linear / norm / sigmoid operations use PyTorch for simplicity.
"""

import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: batched matmul of A (M×K) with Bᵀ (K×N) → C (M×N)
# ----------------------------------------------------------------------
@triton.jit
def batched_matmul_transpose_kernel(
    a_ptr, b_ptr, c_ptr,                     # pointers
    M, N, K,                                 # dimensions (int)
    stride_a_batch, stride_am, stride_ak,    # A strides
    stride_b_batch, stride_bjn, stride_bkn,  # B strides (B is N×K, we need Bᵀ)
    stride_c_batch, stride_cm, stride_cn,    # C strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B.T for many independent batches.
    A shape per batch : (M, K)
    B shape per batch : (N, K)   (we read it as Bᵀ with shape (K, N))
    C shape per batch : (M, N)
    """
    pid_batch = tl.program_id(2)   # batch*hidden_dim dimension
    pid_m = tl.program_id(1)       # output‑row block index
    pid_n = tl.program_id(0)       # output‑col block index

    # ------------------------------------------------------------------
    # Offsets for the current tile
    # ------------------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Base pointers for this batch
    a_batch = a_ptr + pid_batch * stride_a_batch
    b_batch = b_ptr + pid_batch * stride_b_batch
    c_batch = c_ptr + pid_batch * stride_c_batch

    # Accumulator in fp32 (numerically stable)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Loop over K dimension in blocks
    # ------------------------------------------------------------------
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_base = k * BLOCK_K
        offs_k = k_base + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load A tile (M × K) ------------------------------------
        a_ptrs = a_batch \
            + offs_m[:, None] * stride_am \
            + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0)

        # ---- Load Bᵀ tile (K × N) -----------------------------------
        # B is stored as (N, K); Bᵀ[k, n] = B[n, k]
        b_ptrs = b_batch \
            + offs_k[:, None] * stride_bkn \
            + offs_n[None, :] * stride_bjn
        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0)

        # ---- Multiply‑accumulate ------------------------------------
        acc += tl.dot(a, b)          # (M×K)·(K×N) → (M×N)

    # ------------------------------------------------------------------
    # Store the result tile
    # ------------------------------------------------------------------
    c_ptrs = c_batch \
        + offs_m[:, None] * stride_cm \
        + offs_n[None, :] * stride_cn
    tl.store(c_ptrs,
             acc,
             mask=mask_m[:, None] & mask_n[None, :])

# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul module.

    Arguments
    ---------
    data : tuple
        (input_tensor, mask_tensor, weight_dict, config_dict)

    Returns
    -------
    torch.Tensor
        Tensor of shape [bs, seq_len, seq_len, dim]
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask_tensor, weights, config = data
    device = input_tensor.device
    dtype = input_tensor.dtype

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)

    # ------------------------------------------------------------------
    # 1. Input layer‑norm
    # ------------------------------------------------------------------
    x = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
    )

    # ------------------------------------------------------------------
    # 2. Linear projections (left / right)
    # ------------------------------------------------------------------
    left = torch.nn.functional.linear(x, weights["left_proj.weight"])
    right = torch.nn.functional.linear(x, weights["right_proj.weight"])

    # ------------------------------------------------------------------
    # 3. Optional mask (broadcast over hidden dim)
    # ------------------------------------------------------------------
    if not nomask:
        mask = mask_tensor.to(dtype).unsqueeze(-1)   # [B,N,N,1]
        left = left * mask
        right = right * mask

    # ------------------------------------------------------------------
    # 4. Gating (sigmoid)
    # ------------------------------------------------------------------
    left_gate = torch.nn.functional.linear(x, weights["left_gate.weight"]).sigmoid()
    right_gate = torch.nn.functional.linear(x, weights["right_gate.weight"]).sigmoid()
    out_gate = torch.nn.functional.linear(x, weights["out_gate.weight"]).sigmoid()

    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5. Core contraction via Triton
    # ------------------------------------------------------------------
    B, N, _, _ = left.shape          # B = batch size
    # Move hidden_dim to the batch dimension for the kernel
    left_pm = left.permute(0, 3, 1, 2).contiguous()   # [B, H, N, N]
    right_pm = right.permute(0, 3, 1, 2).contiguous()

    # Flatten batch*hidden_dim -> independent GEMM batch
    batch_h = B * hidden_dim
    left_flat = left_pm.view(batch_h, N, N)
    right_flat = right_pm.view(batch_h, N, N)

    # Output buffer
    out_flat = torch.empty_like(left_flat)

    # Kernel launch configuration
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (N + BLOCK_N - 1) // BLOCK_N,     # blocks along N (output cols)
        (N + BLOCK_M - 1) // BLOCK_M,     # blocks along M (output rows)
        batch_h,                           # independent batch dimension
    )

    # Launch the Triton kernel
    batched_matmul_transpose_kernel[grid](
        # pointers
        left_flat, right_flat, out_flat,
        # sizes
        N, N, N,
        # strides for A (left)
        left_flat.stride(0), left_flat.stride(1), left_flat.stride(2),
        # strides for B (right) – we need (batch, N, K) layout
        right_flat.stride(0), right_flat.stride(1), right_flat.stride(2),
        # strides for C (output)
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        # compile‑time block sizes
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Restore original layout: [B, N, N, hidden_dim]
    out = out_flat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------------
    # 6. Hidden‑dim layer‑norm, output gate, and final projection
    # ------------------------------------------------------------------
    out = torch.nn.functional.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
    )
    out = out * out_gate
    out = torch.nn.functional.linear(out, weights["to_out.weight"])

    return out