"""
TriMul (outgoing) – Triton implementation
=========================================
The forward pass computes for each batch `b`, channel `d` and sequence pair `(i,j)`

    out[b, i, j, d] = Σ_k left[b, i, k, d] * right[b, j, k, d]

where `left` and `right` are linear projections of a layer‑normed input, optionally
masked and gated.  The O(N³) contraction is performed by a fused GEMM kernel in
half‑precision (fp16) while accumulating in fp32 for numerical stability.
All remaining lightweight operations (layer‑norms, linear layers, sigmoids, masking)
are executed in PyTorch.
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel for the heavy O(N³) contraction (outgoing variant)
# ----------------------------------------------------------------------
@triton.jit
def _triton_trimul_outgoing_kernel(
    left_ptr, right_ptr, out_ptr,
    B, H, N,
    stride_l_b, stride_l_h, stride_l_i, stride_l_k,
    stride_r_b, stride_r_h, stride_r_j, stride_r_k,
    stride_o_b, stride_o_h, stride_o_i, stride_o_j,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute  out_{b,i,j,h} = Σ_k left_{b,i,k,h} * right_{b,j,k,h}
    for a single (b,h) pair.  The kernel is tiled with blocks of size
    (BLOCK_M, BLOCK_N, BLOCK_K) and accumulates in fp32.
    """
    pid_bh = tl.program_id(2)                     # index over the (batch, hidden) pair
    b = pid_bh // H                               # batch id
    h = pid_bh % H                                # hidden‑channel id

    pid_i = tl.program_id(0)                      # tile row index (i dimension)
    pid_j = tl.program_id(1)                      # tile column index (j dimension)

    # Offsets for the current tile
    offs_i = pid_i * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)

    i_valid = offs_i < N
    j_valid = offs_j < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the reduction dimension k
    for k in range(0, tl.cdiv(N, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_valid = offs_k < N

        # Pointers to the needed sub‑tiles
        left_ptrs = (
            left_ptr
            + b * stride_l_b
            + h * stride_l_h
            + offs_i[:, None] * stride_l_i
            + offs_k[None, :] * stride_l_k
        )
        right_ptrs = (
            right_ptr
            + b * stride_r_b
            + h * stride_r_h
            + offs_j[None, :] * stride_r_j
            + offs_k[:, None] * stride_r_k
        )

        # Load tiles (zero‑pad out‑of‑bounds entries)
        left_tile = tl.load(
            left_ptrs,
            mask=i_valid[:, None] & k_valid[None, :],
            other=0.0,
        )
        right_tile = tl.load(
            right_ptrs,
            mask=k_valid[:, None] & j_valid[None, :],
            other=0.0,
        )

        # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) → (BLOCK_M, BLOCK_N)
        acc += tl.dot(left_tile, right_tile)

    # Write the result back (store as fp16)
    out_ptrs = (
        out_ptr
        + b * stride_o_b
        + h * stride_o_h
        + offs_i[:, None] * stride_o_i
        + offs_j[None, :] * stride_o_j
    )
    tl.store(
        out_ptrs,
        acc.to(tl.float16),
        mask=i_valid[:, None] & j_valid[None, :],
    )


# ----------------------------------------------------------------------
# Public entry point used by the grader
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Outgoing TriMul forward pass.
    Arguments:
        data = (input_tensor, mask_tensor, weights_dict, config_dict)

    Returns:
        Tensor of shape [B, N, N, dim] (float32).
    """
    # ------------------------------------------------------------------
    # unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask_tensor, weights, config = data
    device = input_tensor.device
    B, N, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]
    eps = 1e-6
    nomask = config.get("nomask", False)

    # ------------------------------------------------------------------
    # 1) Input layer‑norm (float32 for stability)
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
    )  # shape [B,N,N,dim], float32

    # ------------------------------------------------------------------
    # 2) Cast to half‑precision for the heavy parts
    # ------------------------------------------------------------------
    x_h = x.to(torch.float16)

    # ------------------------------------------------------------------
    # 3) Linear projections (no bias)
    # ------------------------------------------------------------------
    left = F.linear(x_h, weights["left_proj.weight"].to(torch.float16))
    right = F.linear(x_h, weights["right_proj.weight"].to(torch.float16))

    # ------------------------------------------------------------------
    # 4) Optional mask (broadcasted over the hidden dimension)
    # ------------------------------------------------------------------
    if not nomask:
        # mask shape: [B, N, N] → [B, N, N, 1]
        mask_f = mask_tensor.to(torch.float16).unsqueeze(-1)
        left = left * mask_f
        right = right * mask_f

    # ------------------------------------------------------------------
    # 5) Gating values (sigmoid after a linear)
    # ------------------------------------------------------------------
    left_gate = torch.sigmoid(
        F.linear(x_h, weights["left_gate.weight"].to(torch.float16))
    )
    right_gate = torch.sigmoid(
        F.linear(x_h, weights["right_gate.weight"].to(torch.float16))
    )
    out_gate = torch.sigmoid(
        F.linear(x_h, weights["out_gate.weight"].to(torch.float16))
    )

    # ------------------------------------------------------------------
    # 6) Apply gates
    # ------------------------------------------------------------------
    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 7) Rearrange tensors for the kernel: [B, hidden, N, N]
    # ------------------------------------------------------------------
    left_perm = left.permute(0, 3, 1, 2).contiguous()
    right_perm = right.permute(0, 3, 1, 2).contiguous()

    # Output buffer (same dtype/shape as inputs to the kernel)
    out_perm = torch.empty_like(left_perm, device=device)

    # ------------------------------------------------------------------
    # 8) Triton GEMM kernel launch
    # ------------------------------------------------------------------
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        B * hidden_dim,
    )

    _triton_trimul_outgoing_kernel[grid](
        left_perm,
        right_perm,
        out_perm,
        B,
        hidden_dim,
        N,
        # strides for left
        left_perm.stride(0),
        left_perm.stride(1),
        left_perm.stride(2),
        left_perm.stride(3),
        # strides for right
        right_perm.stride(0),
        right_perm.stride(1),
        right_perm.stride(2),
        right_perm.stride(3),
        # strides for output
        out_perm.stride(0),
        out_perm.stride(1),
        out_perm.stride(2),
        out_perm.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,          # good default for fp16 matmul on H100
    )

    # ------------------------------------------------------------------
    # 9) Restore original layout: [B, N, N, hidden]
    # ------------------------------------------------------------------
    out = out_perm.permute(0, 2, 3, 1).contiguous()  # fp16

    # ------------------------------------------------------------------
    # 10) Post‑processing (LayerNorm, out‑gate, final linear)
    # ------------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(torch.float16)
    to_out_norm_b = weights["to_out_norm.bias"].to(torch.float16)
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=eps,
    )
    out = out * out_gate                      # element‑wise gate

    # Final linear back to original `dim`
    out = F.linear(out, weights["to_out.weight"].to(out.dtype))

    # Return in float32 as expected by the model
    return out.to(torch.float32)