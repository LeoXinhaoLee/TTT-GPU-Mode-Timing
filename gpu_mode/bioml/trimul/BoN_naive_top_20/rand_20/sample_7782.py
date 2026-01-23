"""
TriMul (outgoing) implementation with a fused Triton kernel for mask & gate fusion.

Algorithm
---------
1. Layer‑norm the input tensor (shape [B,N,N,D]).
2. Linear projections (left/right) and gating linear layers (no bias).
3. Fuse mask (optional) and gating via a custom Triton kernel:
     left  = left  * left_gate  * mask
     right = right * right_gate * mask
   The kernel works on tiles of (i, j, hidden) and supports arbitrary
   batch size B (≤2), sequence length N (≤1024) and hidden dimension H (≤128).
4. Compute the pairwise interaction:
        out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
   expressed as a batched matrix‑multiplication:
        out = (left.permute(0,3,1,2) @ right.permute(0,3,1,2).transpose(-2,-1))
        .permute(0,2,3,1)
5. Apply a second LayerNorm, the output gate (sigmoid), and the final linear
   projection back to the original dimension.

The heavy O(N³·H) work is performed by the cuBLAS batched GEMM, while the
mask‑and‑gate fusion is accelerated by Triton.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def apply_mask_gate_kernel(
    left_ptr, right_ptr,
    left_gate_ptr, right_gate_ptr,
    mask_ptr,
    # strides for left
    stride_l_b, stride_l_i, stride_l_j, stride_l_h,
    # strides for right
    stride_r_b, stride_r_i, stride_r_j, stride_r_h,
    # strides for left_gate
    stride_lg_b, stride_lg_i, stride_lg_j, stride_lg_h,
    # strides for right_gate
    stride_rg_b, stride_rg_i, stride_rg_j, stride_rg_h,
    # strides for mask (no hidden dim)
    stride_m_b, stride_m_i, stride_m_j,
    B, N, H,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_H: tl.constexpr
):
    """
    Fuse mask (optional) and gating:
        left  = left  * left_gate  * mask
        right = right * right_gate * mask
    The kernel processes a tile of size (BLOCK_I, BLOCK_J, BLOCK_H) for a
    specific batch index.
    """
    pid = tl.program_id(0)      # combined batch * tile_i * tile_j
    pid_h = tl.program_id(1)    # hidden channel tile

    # Number of tiles in i/j dimensions
    tiles_i = tl.cdiv(N, BLOCK_I)
    tiles_j = tl.cdiv(N, BLOCK_J)

    # Decode batch & tile coordinates
    batch = pid // (tiles_i * tiles_j)
    tile_idx = pid % (tiles_i * tiles_j)
    tile_i = tile_idx // tiles_j
    tile_j = tile_idx % tiles_j

    # Compute starting offsets for this tile
    i0 = tile_i * BLOCK_I
    j0 = tile_j * BLOCK_J
    h0 = pid_h * BLOCK_H

    # Offsets within the tile
    offs_i = i0 + tl.arange(0, BLOCK_I)
    offs_j = j0 + tl.arange(0, BLOCK_J)
    offs_h = h0 + tl.arange(0, BLOCK_H)

    mask_i = offs_i < N
    mask_j = offs_j < N
    mask_h = offs_h < H

    # Compute base pointer for this batch
    base_l = batch * stride_l_b
    base_r = batch * stride_r_b
    base_lg = batch * stride_lg_b
    base_rg = batch * stride_rg_b
    base_m = batch * stride_m_b

    # Pointers to the required blocks
    left_ptrs = left_ptr + base_l \
        + offs_i[:, None, None] * stride_l_i \
        + offs_j[None, :, None] * stride_l_j \
        + offs_h[None, None, :] * stride_l_h

    right_ptrs = right_ptr + base_r \
        + offs_i[:, None, None] * stride_r_i \
        + offs_j[None, :, None] * stride_r_j \
        + offs_h[None, None, :] * stride_r_h

    left_gate_ptrs = left_gate_ptr + base_lg \
        + offs_i[:, None, None] * stride_lg_i \
        + offs_j[None, :, None] * stride_lg_j \
        + offs_h[None, None, :] * stride_lg_h

    right_gate_ptrs = right_gate_ptr + base_rg \
        + offs_i[:, None, None] * stride_rg_i \
        + offs_j[None, :, None] * stride_rg_j \
        + offs_h[None, None, :] * stride_rg_h

    mask_ptrs = mask_ptr + base_m \
        + offs_i[:, None] * stride_m_i \
        + offs_j[None, :] * stride_m_j

    # Load data (masked loads avoid out‑of‑bounds reads)
    left = tl.load(left_ptrs,
        mask=mask_i[:, None, None] & mask_j[None, :, None] & mask_h[None, None, :],
        other=0.0)
    right = tl.load(right_ptrs,
        mask=mask_i[:, None, None] & mask_j[None, :, None] & mask_h[None, None, :],
        other=0.0)
    left_gate = tl.load(left_gate_ptrs,
        mask=mask_i[:, None, None] & mask_j[None, :, None] & mask_h[None, None, :],
        other=0.0)
    right_gate = tl.load(right_gate_ptrs,
        mask=mask_i[:, None, None] & mask_j[None, :, None] & mask_h[None, None, :],
        other=0.0)

    # Load mask (scalar per (i,j), broadcast over hidden)
    mask_val = tl.load(mask_ptrs,
        mask=mask_i[:, None] & mask_j[None, :],
        other=0.0)
    mask_val = mask_val[:, :, None]  # shape (I,J,1) for broadcasting

    # Apply mask and gates
    left_out = left * left_gate * mask_val
    right_out = right * right_gate * mask_val

    # Store the fused result
    tl.store(left_ptrs,
        left_out,
        mask=mask_i[:, None, None] & mask_j[None, :, None] & mask_h[None, None, :])
    tl.store(right_ptrs,
        right_out,
        mask=mask_i[:, None, None] & mask_j[None, :, None] & mask_h[None, None, :])


def custom_kernel(data):
    """
    Triton‑accelerated forward pass of the outgoing TriMul operator.
    Arguments
    ----------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor, shape [B, N, N, D]
        - mask          : torch.Tensor, shape [B, N, N] (binary or float)
        - weights       : dict of model weight tensors
        - config        : dict containing "dim", "hidden_dim", "nomask" etc.
    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, N, N, D] (float32).
    """
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    # ----------------------------------------------------------------------
    # Load weights
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # 1️⃣ LayerNorm on the input (last dimension)
    # ----------------------------------------------------------------------
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=norm_weight,
        bias=norm_bias,
    )  # shape [B,N,N,dim]

    # ----------------------------------------------------------------------
    # 2️⃣ Linear projections (no bias)
    # ----------------------------------------------------------------------
    left = F.linear(x, left_proj_weight)   # [B,N,N,hidden_dim]
    right = F.linear(x, right_proj_weight)

    # ----------------------------------------------------------------------
    # 3️⃣ Gating (sigmoid) + optional mask
    # ----------------------------------------------------------------------
    left_gate = torch.sigmoid(F.linear(x, left_gate_weight))
    right_gate = torch.sigmoid(F.linear(x, right_gate_weight))
    out_gate = torch.sigmoid(F.linear(x, out_gate_weight))

    # Ensure contiguous layout for Triton
    left = left.contiguous()
    right = right.contiguous()
    left_gate = left_gate.contiguous()
    right_gate = right_gate.contiguous()

    # Prepare mask: if nomask == True we treat mask as all‑ones
    if nomask:
        mask_t = torch.ones_like(mask, dtype=left.dtype, device=left.device)
    else:
        mask_t = mask.to(dtype=left.dtype)

    # ----------------------------------------------------------------------
    # 3️⃣ Triton fusion: mask & gate * left/right
    # ----------------------------------------------------------------------
    B, N_seq, _, _ = left.shape
    BLOCK_I = 32
    BLOCK_J = 32
    BLOCK_H = 32

    # Strides (in elements)
    s_l = left.stride()
    s_r = right.stride()
    s_lg = left_gate.stride()
    s_rg = right_gate.stride()
    s_m = mask_t.stride()

    # Grid dimensions
    grid_i = B * triton.cdiv(N_seq, BLOCK_I) * triton.cdiv(N_seq, BLOCK_J)
    grid_h = triton.cdiv(hidden_dim, BLOCK_H)

    apply_mask_gate_kernel[(grid_i, grid_h)](
        left, right,
        left_gate, right_gate,
        mask_t,
        # left strides
        s_l[0], s_l[1], s_l[2], s_l[3],
        # right strides
        s_r[0], s_r[1], s_r[2], s_r[3],
        # left_gate strides
        s_lg[0], s_lg[1], s_lg[2], s_lg[3],
        # right_gate strides
        s_rg[0], s_rg[1], s_rg[2], s_rg[3],
        # mask strides (no hidden dim)
        s_m[0], s_m[1], s_m[2],
        B, N_seq, hidden_dim,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_H=BLOCK_H,
        num_warps=4,
    )

    # ----------------------------------------------------------------------
    # 4️⃣ Pairwise multiplicative update (batched GEMM)
    #    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    # ----------------------------------------------------------------------
    # reshape to (B, hidden, N, N) for cuBLAS
    left_mat = left.permute(0, 3, 1, 2)      # [B, H, N, N]
    right_mat = right.permute(0, 3, 1, 2)    # [B, H, N, N]

    # batched matrix multiplication: (B,H,N,N) @ (B,H,N,N)^T -> (B,H,N,N)
    out_mat = torch.matmul(left_mat, right_mat.transpose(-2, -1))
    out = out_mat.permute(0, 2, 3, 1)        # [B, N, N, H]

    # ----------------------------------------------------------------------
    # 5️⃣ Output LayerNorm, out_gate and final projection
    # ----------------------------------------------------------------------
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weight=to_out_norm_weight,
        bias=to_out_norm_bias,
    )
    out = out * out_gate
    out = F.linear(out, to_out_weight)  # final shape [B, N, N, dim]

    return out