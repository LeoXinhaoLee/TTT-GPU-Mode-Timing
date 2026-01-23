"""
TriMul forward (outgoing) implementation with a fused Triton kernel for the
mask‑gate stage. The heavy N³ pairwise reduction is performed by a highly
optimized batched GEMM (torch.bmm) after the mask & gate fusion.
All major tensor ops (LayerNorm, Linear, Sigmoid) are done in PyTorch,
while the element‑wise mask × gate × projection fusion is carried out by
a custom Triton kernel.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mask_gate_fuse_kernel(
    lp,                     # left_proj    [B, N, N, H]
    rp,                     # right_proj   [B, N, N, H]
    lg,                     # left_gate    [B, N, N, H]
    rg,                     # right_gate   [B, N, N, H]
    mask,                   # mask         [B, N, N]   (broadcasted over H)
    lo,                     # output left  [B, N, N, H]
    ro,                     # output right [B, N, N, H]
    B, N, H,
    # strides of the tensors (in elements, not bytes)
    stride_lp_b, stride_lp_i, stride_lp_j, stride_lp_h,
    stride_rp_b, stride_rp_i, stride_rp_j, stride_rp_h,
    stride_lg_b, stride_lg_i, stride_lg_j, stride_lg_h,
    stride_rg_b, stride_rg_i, stride_rg_j, stride_rg_h,
    stride_mask_b, stride_mask_i, stride_mask_j,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse mask, left‑gate and right‑gate into the projected tensors."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * N * N * H
    active = offs < total

    # ----- index decomposition -------------------------------------------------
    # each element repeats the mask H times → first remove the H factor
    mask_idx = offs // H          # flat index over (B,N,N)
    h_idx = offs % H               # hidden channel

    b_idx = mask_idx // (N * N)
    rem = mask_idx % (N * N)
    i_idx = rem // N
    j_idx = rem % N

    # ----- compute linear offsets ---------------------------------------------
    lp_off  = b_idx * stride_lp_b + i_idx * stride_lp_i + j_idx * stride_lp_j + h_idx * stride_lp_h
    rp_off  = b_idx * stride_rp_b + i_idx * stride_rp_i + j_idx * stride_rp_j + h_idx * stride_rp_h
    lg_off  = b_idx * stride_lg_b + i_idx * stride_lg_i + j_idx * stride_lg_j + h_idx * stride_lg_h
    rg_off  = b_idx * stride_rg_b + i_idx * stride_rg_i + j_idx * stride_rg_j + h_idx * stride_rg_h
    mask_off = b_idx * stride_mask_b + i_idx * stride_mask_i + j_idx * stride_mask_j

    # ----- load ---------------------------------------------------------------
    lp_val = tl.load(lp + lp_off,    mask=active, other=0.0)
    rp_val = tl.load(rp + rp_off,    mask=active, other=0.0)
    lg_val = tl.load(lg + lg_off,    mask=active, other=0.0)
    rg_val = tl.load(rg + rg_off,    mask=active, other=0.0)
    m_val  = tl.load(mask + mask_off, mask=active, other=0.0)

    # ----- fuse ---------------------------------------------------------------
    lo_val = lp_val * m_val * lg_val
    ro_val = rp_val * m_val * rg_val

    # ----- store --------------------------------------------------------------
    tl.store(lo + lp_off, lo_val, mask=active)
    tl.store(ro + rp_off, ro_val, mask=active)


def custom_kernel(data):
    """
    Triton‑accelerated forward pass of the outgoing TriMul module.
    Arguments:
        data = (x, mask, weights, config)
        x          : [B, N, N, C]   (float32)
        mask       : [B, N, N]      (bool/float, optional)
        weights    : dict of tensors (see keys in the skeleton)
        config     : dict with "dim" and "hidden_dim"
    Returns:
        out : [B, N, N, dim] (float32)
    """
    # --------------------------------------------------------------------- unpack
    x, mask, weights, config = data
    dim = config["dim"]
    hidden = config["hidden_dim"]
    device = x.device

    # --------------------------------------------------------------------- defaults
    if mask is None:
        mask = torch.ones(x.shape[:3], dtype=torch.float32, device=device)

    # --------------------------------------------------------------------- layer‑norm on the input (float32 -> FP16 later)
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    x_norm = F.layer_norm(
        x,
        normalized_shape=(dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=1e-5,
    )  # [B,N,N,dim]  float32

    # --------------------------------------------------------------------- half precision for the bulk of the work
    x_h = x_norm.to(torch.float16)

    # --------------------------------------------------------------------- linear projections (no bias)
    left_proj = F.linear(x_h, weights["left_proj.weight"].to(torch.float16))
    right_proj = F.linear(x_h, weights["right_proj.weight"].to(torch.float16))

    # --------------------------------------------------------------------- gates (with sigmoid)
    left_gate = torch.sigmoid(F.linear(x_h, weights["left_gate.weight"].to(torch.float16)))
    right_gate = torch.sigmoid(F.linear(x_h, weights["right_gate.weight"].to(torch.float16)))
    out_gate = torch.sigmoid(F.linear(x_h, weights["out_gate.weight"].to(torch.float16)))

    # --------------------------------------------------------------------- mask (broadcast to hidden)
    mask_h = mask.to(torch.float16)

    # --------------------------------------------------------------------- fused mask × gate × projection via Triton
    B, N, _, _ = x.shape
    left = torch.empty_like(left_proj)
    right = torch.empty_like(right_proj)

    total_elem = B * N * N * hidden
    BLOCK = 1024
    grid = (triton.cdiv(total_elem, BLOCK),)

    _mask_gate_fuse_kernel[
        grid
    ](
        left_proj,
        right_proj,
        left_gate,
        right_gate,
        mask_h,
        left,
        right,
        B,
        N,
        hidden,
        # strides (in element units)
        left_proj.stride(0),
        left_proj.stride(1),
        left_proj.stride(2),
        left_proj.stride(3),
        right_proj.stride(0),
        right_proj.stride(1),
        right_proj.stride(2),
        right_proj.stride(3),
        left_gate.stride(0),
        left_gate.stride(1),
        left_gate.stride(2),
        left_gate.stride(3),
        right_gate.stride(0),
        right_gate.stride(1),
        right_gate.stride(2),
        right_gate.stride(3),
        mask_h.stride(0),
        mask_h.stride(1),
        mask_h.stride(2),
        BLOCK_SIZE=BLOCK,
    )

    # --------------------------------------------------------------------- batched GEMM for the N³ reduction
    # Permute to [B, H, N, N] and flatten batch & hidden for torch.bmm
    left_perm = left.permute(0, 3, 1, 2).contiguous()   # [B, H, N, N]
    right_perm = right.permute(0, 3, 1, 2).contiguous()  # [B, H, N, N]

    bh = B * hidden
    left_mat = left_perm.view(bh, N, N)                # [B*H, N, N]
    right_mat = right_perm.view(bh, N, N)              # [B*H, N, N]

    # matmul: (i,k) @ (j,k)^T => sum_k left[i,k] * right[j,k]
    out_mat = torch.bmm(left_mat, right_mat.transpose(1, 2))  # [B*H, N, N]

    # reshape back to [B, N, N, H]
    out = (
        out_mat.view(B, hidden, N, N)
        .permute(0, 2, 3, 1)
        .contiguous()
    )  # [B, N, N, H]  (still FP16)

    # --------------------------------------------------------------------- output LayerNorm (still half‑precision)
    out = F.layer_norm(
        out,
        normalized_shape=(hidden,),
        weight=weights["to_out_norm.weight"].to(torch.float16),
        bias=weights["to_out_norm.bias"].to(torch.float16),
        eps=1e-5,
    )

    # --------------------------------------------------------------------- apply out‑gate
    out = out * out_gate

    # --------------------------------------------------------------------- final linear back to dim
    out = F.linear(
        out,
        weights["to_out.weight"].to(torch.float16)
    )  # [B, N, N, dim]  FP16

    # --------------------------------------------------------------------- guarantee float32 output
    return out.to(torch.float32)