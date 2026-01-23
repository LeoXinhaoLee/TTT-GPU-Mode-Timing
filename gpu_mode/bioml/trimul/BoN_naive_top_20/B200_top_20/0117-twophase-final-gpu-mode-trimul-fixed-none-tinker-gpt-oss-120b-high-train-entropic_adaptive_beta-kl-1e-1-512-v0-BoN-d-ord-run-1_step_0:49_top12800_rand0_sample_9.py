"""
TriMul “outgoing” forward pass (AlphaFold3 style) implemented with Triton.

Algorithm
---------
1. Layer‑norm the input tensor over the last channel (dim).
2. Linear projections (no bias) to a hidden dimension H:
   left = x·W_left ,   right = x·W_right
3. Optional binary mask (shape [B,N,N]) is applied to both projections.
4. Gated projections:
   left  = left  * sigmoid(x·W_left_gate)
   right = right * sigmoid(x·W_right_gate)
5. Core operation:  out[b,i,j,:] = Σ_k left[b,i,k,:] * right[b,j,k,:]
   This is a batch of H independent matrix‑multiplications:
   for each hidden channel h   out_h = left_h @ right_hᵀ .
   Implemented as a Triton kernel that processes a batch of
   (B·H) matrices of size (N×N) in FP16 with FP32 accumulation.
6. Layer‑norm over the hidden dimension, apply the output gate
   (sigmoid(x·W_out_gate)), and project back to the original dim
   with a final linear layer (no bias).
7. Return the result in the original dtype (float32).

The heavy N³ work is performed by the Triton kernel; all other
operations use standard PyTorch (vectorised) code.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_ah,   # A: (batch, M, K)
    stride_bk, stride_bn, stride_bh,   # B: (batch, K, N)
    stride_cm, stride_cn, stride_ch,   # C: (batch, M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B  for a batch of matrices.
    A : (batch, M, K)   B : (batch, K, N)   C : (batch, M, N)
    All tensors are stored in row‑major order.
    """
    pid_m = tl.program_id(0)          # row block index
    pid_n = tl.program_id(1)          # column block index
    pid_b = tl.program_id(2)          # batch index (B * H)

    # -------------------------------------------------------------
    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # -------------------------------------------------------------
    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of tiles in the reduction dimension
    num_tiles = tl.cdiv(K, BLOCK_K)

    for k in range(num_tiles):
        cur_k = k * BLOCK_K
        offs_k = cur_k + tl.arange(0, BLOCK_K)

        # Load a tile of A and B, guarding OOB accesses
        a = tl.load(
            a_ptr
            + pid_b * stride_ah
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + pid_b * stride_bh
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & mask_n[None, :],
            other=0.0,
        )
        # Dot‑product accumulation
        acc += tl.dot(a, b)

    # -------------------------------------------------------------
    # Write‑back
    c = acc.to(tl.float16)
    c_ptrs = (
        c_ptr
        + pid_b * stride_ch
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


def custom_kernel(data):
    """
    TriMul forward pass (outgoing version) with a Triton kernel for the
    N³ reduction.

    Args:
        data: tuple (input_tensor, mask, weights, config)
            - input_tensor : torch.Tensor of shape [B, N, N, dim]
            - mask         : torch.Tensor of shape [B, N, N] (may be None)
            - weights      : dict of model weights (see docstring)
            - config       : dict with keys "dim", "hidden_dim",
                             optional "nomask" (bool, default True)

    Returns:
        torch.Tensor of shape [B, N, N, dim] (same dtype as input_tensor)
    """
    # -----------------------------------------------------------------
    # Unpack arguments
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-5
    nomask = config.get("nomask", True)

    device = input_tensor.device
    dtype = input_tensor.dtype  # expected float32

    # -----------------------------------------------------------------
    # 1) Input layer‑norm
    norm_w = weights["norm.weight"]
    norm_b = weights["norm.bias"]
    x_norm = F.layer_norm(input_tensor,
                          (dim,),
                          weight=norm_w,
                          bias=norm_b,
                          eps=eps)                # [B,N,N,dim]  fp32

    # Convert to half‑precision for the heavy part
    x_h = x_norm.to(torch.float16)

    # -----------------------------------------------------------------
    # 2) Linear projections (no bias)
    w_left = weights["left_proj.weight"].to(torch.float16)
    w_right = weights["right_proj.weight"].to(torch.float16)
    left = F.linear(x_h, w_left)                     # [B,N,N,hidden_dim] fp16
    right = F.linear(x_h, w_right)

    # -----------------------------------------------------------------
    # 3) Optional mask (broadcast on hidden_dim)
    if not nomask:
        # mask can be bool or numeric; cast to half and add channel dim
        mask_h = mask.to(torch.float16).unsqueeze(-1)  # [B,N,N,1]
        left = left * mask_h
        right = right * mask_h

    # -----------------------------------------------------------------
    # 4) Gated projections
    w_left_gate = weights["left_gate.weight"].to(torch.float16)
    w_right_gate = weights["right_gate.weight"].to(torch.float16)
    w_out_gate = weights["out_gate.weight"].to(torch.float16)

    left_gate = torch.sigmoid(F.linear(x_h, w_left_gate))
    right_gate = torch.sigmoid(F.linear(x_h, w_right_gate))
    out_gate = torch.sigmoid(F.linear(x_h, w_out_gate))

    left = left * left_gate
    right = right * right_gate

    # -----------------------------------------------------------------
    # 5) Batched matrix multiplication via Triton
    B, N, _, H = left.shape                     # H == hidden_dim
    batch_sz = B * H

    # Permute to (batch, hidden, N, N) then reshape to (B*H, N, N)
    left_perm = left.permute(0, 3, 1, 2).contiguous()      # (B,H,N,N)
    right_perm = right.permute(0, 3, 2, 1).contiguous()    # (B,H,N,N)  (j,k) -> (k,j)

    left_flat = left_perm.view(batch_sz, N, N)   # (B*H, N, N)  fp16
    right_flat = right_perm.view(batch_sz, N, N) # (B*H, N, N)  fp16
    out_flat = torch.empty_like(left_flat)       # (B*H, N, N) fp16

    # Triton block sizes – tuned for H100 FP16 matmul
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        batch_sz,
    )

    _batched_matmul_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        N,            # M
        N,            # N
        N,            # K
        left_flat.stride(1),   # stride_am (row stride)
        left_flat.stride(2),   # stride_ak (col stride)
        left_flat.stride(0),   # stride_ah (batch stride)
        right_flat.stride(1),  # stride_bk (row stride in B)
        right_flat.stride(2),  # stride_bn (col stride in B)
        right_flat.stride(0),  # stride_bh (batch stride)
        out_flat.stride(1),    # stride_cm
        out_flat.stride(2),    # stride_cn
        out_flat.stride(0),    # stride_ch
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Restore original layout: (B,N,N,hidden_dim)
    out = out_flat.view(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # -----------------------------------------------------------------
    # 6) Output layer‑norm, output‑gate, final projection
    to_out_norm_w = weights["to_out_norm.weight"].to(torch.float16)
    to_out_norm_b = weights["to_out_norm.bias"].to(torch.float16)
    out_norm = F.layer_norm(out,
                            (hidden_dim,),
                            weight=to_out_norm_w,
                            bias=to_out_norm_b,
                            eps=eps)                     # fp16

    out = out_norm * out_gate.to(torch.float16)

    to_out_w = weights["to_out.weight"].to(torch.float16)
    out = F.linear(out, to_out_w)                         # final [B,N,N,dim] fp16

    # Back to the original dtype (float32) for downstream usage
    return out.to(dtype)