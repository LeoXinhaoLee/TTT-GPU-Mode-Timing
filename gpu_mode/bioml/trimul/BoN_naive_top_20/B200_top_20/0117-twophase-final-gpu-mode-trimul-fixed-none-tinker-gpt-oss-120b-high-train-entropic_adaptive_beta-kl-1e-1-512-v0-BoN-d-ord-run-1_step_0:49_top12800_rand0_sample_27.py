"""
TriMul “outgoing” kernel (AlphaFold3 TriMul variant).

Algorithm
---------
1. Layer‑norm the input   x ∈ [B, N, N, D]  (D = dim).
2. Linear projections (no bias) to a hidden dimension H:
       left  = x·W_left   (B,N,N,H)
       right = x·W_right  (B,N,N,H)
3. Compute three sigmoidal gates from the same normalized input:
       left_gate  = σ(x·W_lgate)
       right_gate = σ(x·W_rgate)
       out_gate   = σ(x·W_outgate)
4. Fuse mask (if present) and the per‑pair gates:
       left  = left * left_gate  * mask
       right = right* right_gate * mask
5. Core operation – pairwise multiplicative update:
       out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]
   This is a batched matrix multiplication of (B·H) independent
   (N×N) × (N×N)ᵀ products.  It is implemented with a custom
   Triton kernel that blocks the computation (M‑tile × N‑tile × K‑tile)
   and accumulates in FP32 for numerical stability.
6. Layer‑norm over the hidden dimension, apply the output gate and a final
   linear projection back to the original dimension D.
7. Return the result in FP32.

The kernel works on FP16 tensors (except the final accumulation which is FP32)
and therefore reduces memory bandwidth while keeping the final output
precision identical to the reference implementation.
"""

import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel for batched (B·H) matrix multiplication:
#   C = A @ Bᵀ where
#   A : (tasks, M, K)   – left tensor (i,k)   per hidden channel
#   B : (tasks, K, N)   – right tensor (k,j) per hidden channel
#   C : (tasks, M, N)   – result
# ----------------------------------------------------------------------
@triton.jit
def batch_matmul_kernel(
    a_ptr, b_ptr, c_ptr,               # pointers
    M, N, K,                           # matrix dimensions
    stride_am, stride_ai, stride_ak,    # strides for A (task, i, k)
    stride_bm, stride_bk, stride_bn,    # strides for B (task, k, n)
    stride_cm, stride_ci, stride_cn,    # strides for C (task, i, n)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # --------------------------------------------------------------
    # Program ID layout:
    #   pid0 encodes (task_id, tile_i)  ->  pid0 = task_id * num_tile_m + tile_i
    #   pid1 encodes tile_j
    # --------------------------------------------------------------
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    num_tile_m = tl.cdiv(M, BLOCK_M)

    task_id = pid0 // num_tile_m
    tile_i = pid0 % num_tile_m

    # Global row / column indices for this block
    i = tile_i * BLOCK_M + tl.arange(0, BLOCK_M)
    j = pid1   * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_i = i < M
    mask_j = j < N

    # Offsets for the current batch (task)
    a_batch_off = task_id * stride_am
    b_batch_off = task_id * stride_bm
    c_batch_off = task_id * stride_cm

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --------------------------------------------------------------
    # K‑loop
    # --------------------------------------------------------------
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        ks = k_offset + tl.arange(0, BLOCK_K)
        mask_k = ks < K

        # Load A tile: (BLOCK_M, BLOCK_K)
        a_ptrs = a_ptr + a_batch_off + i[:, None] * stride_ai + ks[None, :] * stride_ak
        a = tl.load(a_ptrs,
                    mask=mask_i[:, None] & mask_k[None, :],
                    other=0.0)

        # Load B tile: (BLOCK_K, BLOCK_N)
        b_ptrs = b_ptr + b_batch_off + ks[:, None] * stride_bk + j[None, :] * stride_bn
        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_j[None, :],
                    other=0.0)

        # Multiply‑accumulate (fp16 * fp16 → fp32)
        acc += tl.dot(a, b)

    # --------------------------------------------------------------
    # Write result
    # --------------------------------------------------------------
    c_ptrs = c_ptr + c_batch_off + i[:, None] * stride_ci + j[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16),
             mask=mask_i[:, None] & mask_j[None, :])


# ----------------------------------------------------------------------
# Entry‑point used by the evaluation harness
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.

    Arguments
    ---------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor [B, N, N, D]   (float32)
        - mask         : torch.Tensor [B, N, N] or None (bool / float)
        - weights      : dict of model parameters (torch.Tensor)
        - config       : dict, must contain "hidden_dim" (int)

    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, N, N, D] (float32)
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    B, N, _, D = input_tensor.shape
    H = config["hidden_dim"]                     # hidden dimension

    # ------------------------------------------------------------------
    # LayerNorm on the last dimension
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias   = weights["norm.bias"]
    x = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=(D,),
        weight=norm_weight,
        bias=norm_bias,
        eps=1e-6,
    )
    # Perform remaining ops in FP16 to save memory / bandwidth
    x = x.to(torch.float16)

    # ------------------------------------------------------------------
    # Linear projections (no bias)
    # ------------------------------------------------------------------
    left_proj_w   = weights["left_proj.weight"].to(torch.float16)
    right_proj_w  = weights["right_proj.weight"].to(torch.float16)
    left_gate_w   = weights["left_gate.weight"].to(torch.float16)
    right_gate_w  = weights["right_gate.weight"].to(torch.float16)
    out_gate_w    = weights["out_gate.weight"].to(torch.float16)
    to_out_norm_w = weights["to_out_norm.weight"].to(torch.float16)
    to_out_norm_b = weights["to_out_norm.bias"].to(torch.float16)
    to_out_w      = weights["to_out.weight"].to(torch.float16)

    left  = torch.nn.functional.linear(x, left_proj_w)          # (B, N, N, H)
    right = torch.nn.functional.linear(x, right_proj_w)         # (B, N, N, H)

    left_gate  = torch.nn.functional.linear(x, left_gate_w).sigmoid()
    right_gate = torch.nn.functional.linear(x, right_gate_w).sigmoid()
    out_gate   = torch.nn.functional.linear(x, out_gate_w).sigmoid()

    # ------------------------------------------------------------------
    # Optional mask handling
    # ------------------------------------------------------------------
    use_mask = (mask is not None) and (not config.get("nomask", True))
    if use_mask:
        # mask: [B, N, N] -> [B, N, N, 1]
        mask_f = mask.to(torch.float16).unsqueeze(-1)
        left  = left * mask_f
        right = right * mask_f

    # ------------------------------------------------------------------
    # Fuse per‑pair gates
    # ------------------------------------------------------------------
    left  = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # Prepare tensors for the batched matmul kernel
    # ------------------------------------------------------------------
    # left:  (B, N, N, H) -> (B, H, N, N)  (i,k) layout
    left_perm   = left.permute(0, 3, 1, 2).contiguous()
    # right: (B, N, N, H) -> (B, H, N, N) but we need (k,j) → transpose last two axes
    right_perm  = right.permute(0, 3, 2, 1).contiguous()

    # Collapse the batch & hidden dimensions so each GEMM can be launched independently
    left_flat   = left_perm.view(-1, N, N)      # (B*H, M=N, K=N)
    right_flat  = right_perm.view(-1, N, N)     # (B*H, K=N, N)
    out_flat    = torch.empty_like(left_flat)   # (B*H, N, N)

    # ------------------------------------------------------------------
    # Strides (in element units) required by the kernel
    # ------------------------------------------------------------------
    stride_am, stride_ai, stride_ak = left_flat.stride()    # (task, i, k)
    stride_bm, stride_bk, stride_bn = right_flat.stride()   # (task, k, n)
    stride_cm, stride_ci, stride_cn = out_flat.stride()     # (task, i, n)

    # ------------------------------------------------------------------
    # Triton launch configuration
    # ------------------------------------------------------------------
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    num_tasks   = left_flat.shape[0]                     # B * H
    num_tile_m  = (N + BLOCK_M - 1) // BLOCK_M
    num_tile_n  = (N + BLOCK_N - 1) // BLOCK_N

    grid = (num_tasks * num_tile_m, num_tile_n)

    # ------------------------------------------------------------------
    # Call the kernel
    # ------------------------------------------------------------------
    batch_matmul_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        N,          # M
        N,          # N (output)
        N,          # K
        stride_am, stride_ai, stride_ak,
        stride_bm, stride_bk, stride_bn,
        stride_cm, stride_ci, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # ------------------------------------------------------------------
    # Reshape back to [B, N, N, H]
    # ------------------------------------------------------------------
    out = out_flat.view(B, H, N, N).permute(0, 2, 3, 1)   # (B, N, N, H)

    # ------------------------------------------------------------------
    # Post‑processing: LayerNorm, output gate, final projection
    # ------------------------------------------------------------------
    out_norm = torch.nn.functional.layer_norm(
        out,
        normalized_shape=(H,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=1e-6,
    )
    out = out_norm * out_gate
    out = torch.nn.functional.linear(out, to_out_w)

    # Cast back to FP32 as required by the spec
    return out.to(torch.float32)