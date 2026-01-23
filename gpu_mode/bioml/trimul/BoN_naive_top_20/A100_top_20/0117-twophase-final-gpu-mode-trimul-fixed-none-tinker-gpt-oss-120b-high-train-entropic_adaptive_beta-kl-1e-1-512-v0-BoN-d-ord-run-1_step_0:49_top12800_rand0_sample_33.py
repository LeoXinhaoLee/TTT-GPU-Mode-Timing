"""
TriMul (outgoing) fused kernel for AlphaFold‑3.

Algorithm
---------
1. Layer‑norm the input tensor `[B, N, N, C]` (`C = dim`).
2. Compute left/right projections and three gates with linear layers.
   All heavy linear ops are performed in FP16 for speed.
3. Apply the masks (if present) and the element‑wise gates.
4. Core operation: for each hidden channel `h` we need
        out[b,i,j,h] = Σₖ left[b,i,k,h] * right[b,j,k,h] .
   This is a batched matrix multiplication `L_h @ R_hᵀ` where
   `L_h` and `R_h` have shape `[N, N]`.  We fuse the entire batch
   and all hidden channels into a single Triton kernel that computes
   many independent GEMMs in parallel (batch = B * hidden_dim).
5. Layer‑norm on the hidden dimension, apply the output gate and a final
   linear projection back to `dim`.
6. Return the result in FP32 (the model expects FP32 output).

The Triton kernel works on FP16 data and accumulates in FP32.
All strides are passed explicitly to avoid any re‑ordering.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr,               # [batch, M, K]   (left)
    B_ptr,               # [batch, K, N]   (rightᵀ)
    C_ptr,               # [batch, M, N]   (output)
    M, N, K,
    stride_am, stride_ak,               # strides for A (M,K)
    stride_bk, stride_bn,               # strides for B (K,N)
    stride_cm, stride_cn,               # strides for C (M,N)
    stride_abatch, stride_bbatch, stride_cbatch,   # batch strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes many independent GEMMs:
        C[b] = A[b] @ B[b]   for all b in [0, batch).
    A, B and C are FP16 tensors; accumulation is done in FP32.
    """

    pid = tl.program_id(0)

    # Number of tiles for each matrix dimension
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)

    # Decode the program id into batch and tile indices
    batch = pid // (num_blocks_m * num_blocks_n)          # which GEMM
    pid = pid % (num_blocks_m * num_blocks_n)            # which tile inside GEMM
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    # Offsets of the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator (FP32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Pointers for the current tile
        a_ptrs = (A_ptr + batch * stride_abatch
                  + offs_m[:, None] * stride_am
                  + offs_k[None, :] * stride_ak)

        b_ptrs = (B_ptr + batch * stride_bbatch
                  + offs_k[:, None] * stride_bk
                  + offs_n[None, :] * stride_bn)

        # Load tiles, cast to FP32 for accumulation
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0).to(tl.float32)

        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0).to(tl.float32)

        # Matrix multiply‑accumulate
        acc += tl.dot(a, b)

    # Write the result back in FP16
    c_ptrs = (C_ptr + batch * stride_cbatch
              + offs_m[:, None] * stride_cm
              + offs_n[None, :] * stride_cn)

    tl.store(c_ptrs, acc.to(tl.float16),
             mask=mask_m[:, None] & mask_n[None, :])


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul module.

    Parameters
    ----------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor   [B, N, N, dim]   (FP32)
        - mask         : torch.Tensor   [B, N, N]       (bool/float) or None
        - weights      : dict of torch.Tensor with the model parameters
        - config       : dict with keys ``dim``, ``hidden_dim`` and optional ``nomask``

    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, N, N, dim] (FP32)
    """
    # ------------------------------------------------------------------
    # unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)
    device = input_tensor.device
    eps = 1e-5

    # ------------------------------------------------------------------
    # unpack weights (all are stored in FP32; we will cast to FP16 where needed)
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias   = weights["norm.bias"]

    left_proj_weight  = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]

    left_gate_weight  = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight   = weights["out_gate.weight"]

    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias   = weights["to_out_norm.bias"]
    to_out_weight      = weights["to_out.weight"]   # shape [dim, hidden_dim]

    # ------------------------------------------------------------------
    # 1) Input LayerNorm (over the last dim)
    # ------------------------------------------------------------------
    x = F.layer_norm(input_tensor,
                     normalized_shape=(dim,),
                     weight=norm_weight,
                     bias=norm_bias,
                     eps=eps)                         # [B,N,N,dim] FP32

    # Cast to FP16 for the heavy linear ops
    x_h = x.to(torch.float16)

    # ------------------------------------------------------------------
    # 2) Linear projections (no bias)
    # ------------------------------------------------------------------
    left  = F.linear(x_h, left_proj_weight.to(torch.float16))   # [B,N,N,hidden_dim]
    right = F.linear(x_h, right_proj_weight.to(torch.float16))

    # ------------------------------------------------------------------
    # 3) Optional mask
    # ------------------------------------------------------------------
    if (not nomask) and mask is not None:
        # mask: [B,N,N] -> [B,N,N,1]
        mask_f = mask.to(torch.float16).unsqueeze(-1)
        left  = left * mask_f
        right = right * mask_f

    # ------------------------------------------------------------------
    # 4) Gates (sigmoid after a linear projection)
    # ------------------------------------------------------------------
    left_gate  = torch.sigmoid(F.linear(x_h, left_gate_weight.to(torch.float16)))
    right_gate = torch.sigmoid(F.linear(x_h, right_gate_weight.to(torch.float16)))
    out_gate   = torch.sigmoid(F.linear(x_h, out_gate_weight.to(torch.float16)))

    left  = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5) Core TriMul: batched GEMM for every hidden channel
    #    out[b,i,j,h] = Σₖ left[b,i,k,h] * right[b,j,k,h]
    # ------------------------------------------------------------------
    #   shape preparation
    left_perm  = left.permute(0, 3, 1, 2).contiguous()   # [B, H, N, N]
    right_perm = right.permute(0, 3, 2, 1).contiguous()  # [B, H, N, N] (k, j)

    B_batch, H, N, _ = left_perm.shape               # B_batch == batch size
    total = B_batch * H                               # number of independent GEMMs

    # flatten batch and hidden dimension → [total, N, N]
    A = left_perm.view(total, N, N)
    B_mat = right_perm.view(total, N, N)
    C = torch.empty_like(A)

    # strides (in elements) for the three tensors
    stride_a_batch = A.stride(0)
    stride_a_m     = A.stride(1)
    stride_a_k     = A.stride(2)

    stride_b_batch = B_mat.stride(0)
    stride_b_k     = B_mat.stride(1)
    stride_b_n     = B_mat.stride(2)

    stride_c_batch = C.stride(0)
    stride_c_m     = C.stride(1)
    stride_c_n     = C.stride(2)

    # Tunable block sizes – chosen for H100 Tensor‑cores
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # launch grid: one program per (batch * tile_m * tile_n)
    grid_m = (N + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (total * grid_m * grid_n,)

    batched_matmul_kernel[grid](
        A,
        B_mat,
        C,
        N,          # M
        N,          # N
        N,          # K
        stride_am=stride_a_m,
        stride_ak=stride_a_k,
        stride_bk=stride_b_k,
        stride_bn=stride_b_n,
        stride_cm=stride_c_m,
        stride_cn=stride_c_n,
        stride_abatch=stride_a_batch,
        stride_bbatch=stride_b_batch,
        stride_cbatch=stride_c_batch,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # reshape back to [B, N, N, hidden_dim]
    out_hidden = C.view(B_batch, H, N, N).permute(0, 2, 3, 1)   # [B, N, N, hidden_dim]

    # ------------------------------------------------------------------
    # 6) Output LayerNorm, output gate and final linear projection
    # ------------------------------------------------------------------
    out = F.layer_norm(out_hidden,
                       normalized_shape=(hidden_dim,),
                       weight=to_out_norm_weight.to(out_hidden.dtype),
                       bias=to_out_norm_bias.to(out_hidden.dtype),
                       eps=eps)

    out = out * out_gate                     # element‑wise gate
    out = F.linear(out, to_out_weight.to(out.dtype))  # back to `dim`

    # Return FP32 as expected by the model
    return out.to(torch.float32)