"""
custom_kernel for the TriMul (outgoing) operator.

Algorithm
---------
1. Layer‑norm the input tensor.
2. Linear projections (left/right) and their gates (left_gate/right_gate/out_gate).
3. Optional masking of left/right.
4. Apply gates to the projected tensors.
5. Compute the “outgoing” pairwise update:
       out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]
   This is a batched matrix multiplication where each (b,h) pair
   multiplies a (N×N) matrix with the transpose of another (N×N) matrix.
   The heavy N³ work is performed by a custom Triton kernel that
   processes many (b,h) batches in parallel.
6. Layer‑norm over the hidden dimension, multiply by out_gate and a final
   linear projection back to the original channel dimension.

The kernel uses float32 accumulation (inputs are cast to float32 for safety)
and a 2‑level tiling (128×128 output tiles, 32‑wide reduction).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _batched_gemm_kernel(
    a_ptr,               # LEFT  : (batch, M, K)
    b_ptr,               # RIGHT : (batch, K, N)   (already transposed)
    c_ptr,               # OUT   : (batch, M, N)
    stride_ab, stride_am, stride_ak,   # strides for A
    stride_bb, stride_bk, stride_bn,   # strides for B
    stride_cb, stride_cm, stride_cn,   # strides for C
    M, N, K,                         # matrix sizes (all = seq_len)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched GEMM:  C[b] = A[b] @ B[b]   where
        A : (M, K)
        B : (K, N)   (note B is already the transpose of the original right tensor)
    """
    pid_m = tl.program_id(0)   # tile row
    pid_n = tl.program_id(1)   # tile col
    pid_b = tl.program_id(2)   # batch ( = batch * hidden_dim )

    # ----------- compute tile start indices -------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ------------ batch offsets -------------------------
    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    c_batch_ptr = c_ptr + pid_b * stride_cb

    # ------------ accumulator ---------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------ k‑loop (reduction) -------------------
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # mask out‑of‑range elements
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(
            a_batch_ptr
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )
        b = tl.load(
            b_batch_ptr
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn,
            mask=b_mask,
            other=0.0,
        )
        # dot: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, b)

    # ------------ write back ----------------------------
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        c_batch_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn,
        acc,
        mask=c_mask,
    )


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.
    Arguments
    ---------
    data : tuple
        (input_tensor, mask_tensor, weights_dict, config_dict)

    Returns
    -------
    torch.Tensor
        Tensor of shape [B, N, N, dim] (float32)
    """
    # -------------------------------------------------------
    # unpack inputs
    # -------------------------------------------------------
    input_tensor, mask_tensor, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    # -------------------------------------------------------
    # fetch weights (all stored on the same device as input)
    # -------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    # -------------------------------------------------------
    # 1. LayerNorm over the channel dimension
    # -------------------------------------------------------
    x = F.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=norm_weight,
        bias=norm_bias,
    )  # (B,N,N,dim)

    # -------------------------------------------------------
    # 2. Linear projections (no bias)
    # -------------------------------------------------------
    left = F.linear(x, left_proj_weight)   # (B,N,N,hidden_dim)
    right = F.linear(x, right_proj_weight) # (B,N,N,hidden_dim)

    # -------------------------------------------------------
    # 3. Optional mask
    # -------------------------------------------------------
    if not nomask:
        # mask shape: (B,N,N) -> (B,N,N,1)
        mask_f = mask_tensor.to(x.dtype).unsqueeze(-1)
        left = left * mask_f
        right = right * mask_f

    # -------------------------------------------------------
    # 4. Gating
    # -------------------------------------------------------
    left_gate = F.linear(x, left_gate_weight).sigmoid()
    right_gate = F.linear(x, right_gate_weight).sigmoid()
    out_gate = F.linear(x, out_gate_weight).sigmoid()

    left = left * left_gate
    right = right * right_gate

    # -------------------------------------------------------
    # 5. Cast to float32 for the heavy GEMM kernel
    # -------------------------------------------------------
    left = left.to(torch.float32)
    right = right.to(torch.float32)

    # -------------------------------------------------------
    # 6. Prepare tensors for the batched GEMM
    #    left  : (B, hidden_dim, N, N)   (i,k)
    #    right : (B, hidden_dim, N, N)   (j,k)  -> we need (k,j)
    # -------------------------------------------------------
    B, N, _, _ = left.shape

    left = left.permute(0, 3, 1, 2).contiguous()          # (B, H, N, N)
    right = right.permute(0, 3, 2, 1).contiguous()        # (B, H, N, N)  (k,j)

    # flatten batch*hidden_dim dimension
    batch_hidden = B * hidden_dim
    left_flat = left.view(batch_hidden, N, N)
    right_flat = right.view(batch_hidden, N, N)

    # output tensor
    out_flat = torch.empty(
        (batch_hidden, N, N), dtype=torch.float32, device=input_tensor.device
    )

    # -------------------------------------------------------
    # 7. Strides for the Triton kernel
    # -------------------------------------------------------
    stride_ab = left_flat.stride(0)
    stride_am = left_flat.stride(1)
    stride_ak = left_flat.stride(2)

    stride_bb = right_flat.stride(0)
    stride_bk = right_flat.stride(1)
    stride_bn = right_flat.stride(2)

    stride_cb = out_flat.stride(0)
    stride_cm = out_flat.stride(1)
    stride_cn = out_flat.stride(2)

    # -------------------------------------------------------
    # 8. Launch the batched GEMM kernel
    # -------------------------------------------------------
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        batch_hidden,
    )

    _batched_gemm_kernel[grid](
        a_ptr=left_flat,
        b_ptr=right_flat,
        c_ptr=out_flat,
        stride_ab=stride_ab,
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bb=stride_bb,
        stride_bk=stride_bk,
        stride_bn=stride_bn,
        stride_cb=stride_cb,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        M=N,
        N=N,
        K=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=2,
    )

    # -------------------------------------------------------
    # 9. Reshape back to (B, N, N, hidden_dim)
    # -------------------------------------------------------
    out = out_flat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1)  # (B,N,N,hidden_dim)

    # -------------------------------------------------------
    # 10. Final LayerNorm, out_gate and projection
    # -------------------------------------------------------
    out = F.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=to_out_norm_weight,
        bias=to_out_norm_bias,
    )
    out = out * out_gate
    out = F.linear(out, to_out_weight)   # (B,N,N,dim)

    return out