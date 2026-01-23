"""
TriMul "outgoing" forward pass implemented with a fused Triton kernel.

Algorithm
---------
1. Layer‑norm the input tensor `x` (shape B×N×N×C) using the provided
   `norm.weight` / `norm.bias`.
2. Linear projections (no bias):
       left  = x_norm @ left_proj.weight.T    → shape B×N×N×H
       right = x_norm @ right_proj.weight.T   → shape B×N×N×H
3. Optional mask (config["nomask"] == False):
       left  *= mask[...,None]
       right *= mask[...,None]
4. Gating (sigmoid on linear transforms of `x_norm`):
       left_gate  = sigmoid(x_norm @ left_gate.weight.T)
       right_gate = sigmoid(x_norm @ right_gate.weight.T)
       out_gate   = sigmoid(x_norm @ out_gate.weight.T)
   Apply the input‑side gates:
       left  = left  * left_gate
       right = right * right_gate
5. Compute the pairwise multiplicative update
       out[b,i,j,:] = Σ_k left[b,i,k,:] * right[b,j,k,:]
   This is a batched matrix multiply `C = A @ B^T` for each hidden
   dimension.  The heavy work is done by a custom Triton kernel that
   fuses the reduction across `k`.  The tensors are cast to float16 for
   the kernel, the accumulator is float32, and the result is cast back
   to float32 after the kernel.
6. Apply a second Layer‑norm (`to_out_norm`), multiply by `out_gate`,
   and a final linear projection (`to_out.weight`) to obtain the output
   shape B×N×N×C.

The Triton kernel processes each (batch * hidden_dim) slice independently
and tiles the N×N output with BLOCK_M × BLOCK_N tiles.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _trmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_az, stride_am, stride_ak,
    stride_bz, stride_bn, stride_bk,
    stride_cz, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Batched GEMM:   C = A @ Bᵀ   where A,B ∈ (Z, M/N, K)"""

    pid_z = tl.program_id(2)          # batch*hidden index
    pid_m = tl.program_id(0)          # tile row   (i)
    pid_n = tl.program_id(1)          # tile col   (j)

    # tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # boundary masks
    mask_m = offs_m < M
    mask_n = offs_n < N

    # accumulator in fp32 for accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A tile: (BLOCK_M, BLOCK_K)
        a_ptrs = a_ptr + pid_z * stride_az \
                       + offs_m[:, None] * stride_am \
                       + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs,
                    mask=(mask_m[:, None] & mask_k[None, :]),
                    other=0.0)
        a = a.to(tl.float32)

        # Bᵀ tile: (BLOCK_K, BLOCK_N)
        # B is stored as (Z, N, K); we treat it as (Z, K, N) here.
        b_ptrs = b_ptr + pid_z * stride_bz \
                       + offs_k[:, None] * stride_bk \
                       + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs,
                    mask=(mask_k[:, None] & mask_n[None, :]),
                    other=0.0)
        b = b.to(tl.float32)

        # fused dot product
        acc += tl.dot(a, b)   # (BLOCK_M, BLOCK_N)

    # write back
    c_ptrs = c_ptr + pid_z * stride_cz \
                     + offs_m[:, None] * stride_cm \
                     + offs_n[None, :] * stride_cn
    tl.store(c_ptrs,
             acc.to(tl.float16),
             mask=(mask_m[:, None] & mask_n[None, :]))


def custom_kernel(data):
    """
    Forward pass of the TriMul “outgoing” operator.

    Parameters
    ----------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor [B, N, N, C]  (float32)
        - mask         : torch.Tensor [B, N, N]  (bool or float) or None
        - weights      : dict of torch.Tensor containing the model parameters
        - config       : dict containing at least 'dim' and 'hidden_dim' and
                         optionally 'nomask' (bool)

    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, N, N, C] (float32)
    """
    # Unpack
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    device = input_tensor.device
    dtype = input_tensor.dtype

    # ---------- 1. LayerNorm ----------
    x_norm = F.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
    )   # [B, N, N, C]

    # ---------- 2. Projections ----------
    left = F.linear(x_norm, weights["left_proj.weight"])   # no bias
    right = F.linear(x_norm, weights["right_proj.weight"])

    # ---------- 3. Optional mask ----------
    if not config.get("nomask", False) and mask is not None:
        # mask shape [B,N,N] → [B,N,N,1] for broadcasting
        mask_exp = mask.unsqueeze(-1).to(left.dtype)
        left = left * mask_exp
        right = right * mask_exp

    # ---------- 4. Gating ----------
    left_gate = torch.sigmoid(F.linear(x_norm, weights["left_gate.weight"]))
    right_gate = torch.sigmoid(F.linear(x_norm, weights["right_gate.weight"]))
    out_gate = torch.sigmoid(F.linear(x_norm, weights["out_gate.weight"]))

    left = left * left_gate
    right = right * right_gate

    # ---------- 5. Multiplicative reduction via Triton ----------
    # Cast to float16 for the kernel (memory + speed)
    left_h = left.to(torch.float16)
    right_h = right.to(torch.float16)

    B, N, _, _ = left_h.shape

    # Rearrange to (Z, M, K) where Z = B * hidden_dim, M = N, K = N
    left_perm = left_h.permute(0, 3, 1, 2).contiguous()   # (B, H, N, N)
    right_perm = right_h.permute(0, 3, 1, 2).contiguous()  # (B, H, N, N)

    Z = B * hidden_dim
    left_mat = left_perm.view(Z, N, N)   # (Z, M, K)
    right_mat = right_perm.view(Z, N, N)  # (Z, N, K) – will be used as Bᵀ inside kernel

    out_mat = torch.empty_like(left_mat, dtype=torch.float16, device=device)

    # Strides for Triton (Z, M, K) layout
    stride_az = left_mat.stride(0)
    stride_am = left_mat.stride(1)
    stride_ak = left_mat.stride(2)

    stride_bz = right_mat.stride(0)
    stride_bn = right_mat.stride(1)   # stride for the “N” dimension (j)
    stride_bk = right_mat.stride(2)   # stride for the “K” dimension (k)

    stride_cz = out_mat.stride(0)
    stride_cm = out_mat.stride(1)
    stride_cn = out_mat.stride(2)

    # Tiling parameters – chosen to fit H100 shared‑memory limits
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        triton.cdiv(N, BLOCK_M),   # i‑tiles
        triton.cdiv(N, BLOCK_N),   # j‑tiles
        Z,                         # one program per (batch * hidden)
    )

    _trmul_kernel[grid](
        left_mat,
        right_mat,
        out_mat,
        N, N, N,
        stride_az, stride_am, stride_ak,
        stride_bz, stride_bn, stride_bk,
        stride_cz, stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Re‑shape back to (B, N, N, H) and cast to float32 for the rest
    out = out_mat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()
    out = out.to(torch.float32)

    # ---------- 6. Output LayerNorm, gating and final linear ----------
    out = F.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
    )
    out = out * out_gate
    out = F.linear(out, weights["to_out.weight"])

    return out