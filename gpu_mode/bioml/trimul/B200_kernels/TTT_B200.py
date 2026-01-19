# -*- coding: utf-8 -*-
"""
Optimized Triton implementation of the **outgoing** TriMul operator
(AlphaFold‑3 style).  Only the forward pass is required.

Signature
---------
custom_kernel(data) -> torch.Tensor

where
    data = (input_tensor, mask, weights, config)

    input_tensor : torch.Tensor[bs, N, N, dim]   (float32)
    mask         : torch.Tensor[bs, N, N]       (bool/float) or None
    weights      : dict of torch.Tensor containing all model weights
    config       : dict with keys:
                   - "dim"         (int)
                   - "hidden_dim"  (int)
                   - "nomask"      (bool, optional, default True)

Returns
-------
torch.Tensor[bs, N, N, dim]  (float32)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# 1️⃣  LayerNorm on the input (fp32 → fp16)
# ----------------------------------------------------------------------
@triton.jit
def _layernorm_fp32_to_fp16_kernel(
    x_ptr,          # float32*   (B·N·N·dim) flattened
    out_ptr,        # float16*   same shape
    weight_ptr,     # float32*   (dim,)   γ
    bias_ptr,       # float32*   (dim,)   β
    total_cells: tl.int32,   # B·N·N
    eps: tl.float32,
    # compile‑time constants
    BLOCK_CELL: tl.constexpr,   # #cells processed per program
    BLOCK_D:    tl.constexpr,   # #channels per tile
    DIM:        tl.constexpr,   # model dimension (dim)
):
    pid = tl.program_id(0)

    # ------------------------------------------------------------------
    # Tile over the (B·N·N) positions (“cells”)
    # ------------------------------------------------------------------
    cell_off = pid * BLOCK_CELL + tl.arange(0, BLOCK_CELL)          # (BLOCK_CELL,)
    cell_mask = cell_off < total_cells

    # ------------------------------------------------------------------
    # PASS‑1 : compute per‑cell mean & variance (fp32 accumulation)
    # ------------------------------------------------------------------
    sum_  = tl.zeros((BLOCK_CELL,), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_CELL,), dtype=tl.float32)

    n_blocks = tl.cdiv(DIM, BLOCK_D)
    for blk in range(0, n_blocks):
        d_start = blk * BLOCK_D
        d_off   = d_start + tl.arange(0, BLOCK_D)
        d_mask  = d_off < DIM

        # flat index of the tile: cell * DIM + d
        idx = cell_off[:, None] * DIM + d_off[None, :]               # (BLOCK_CELL, BLOCK_D)

        vals = tl.load(x_ptr + idx,
                       mask=cell_mask[:, None] & d_mask[None, :],
                       other=0.0)                                     # fp32

        sum_  += tl.sum(vals, axis=1)
        sumsq += tl.sum(vals * vals, axis=1)

    mean = sum_ / DIM
    var  = sumsq / DIM - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)                               # (BLOCK_CELL,)

    # ------------------------------------------------------------------
    # PASS‑2 : apply γ, β and store as FP16
    # ------------------------------------------------------------------
    for blk in range(0, n_blocks):
        d_start = blk * BLOCK_D
        d_off   = d_start + tl.arange(0, BLOCK_D)
        d_mask  = d_off < DIM

        idx = cell_off[:, None] * DIM + d_off[None, :]               # (BLOCK_CELL, BLOCK_D)

        vals = tl.load(x_ptr + idx,
                       mask=cell_mask[:, None] & d_mask[None, :],
                       other=0.0)                                     # fp32

        normed = (vals - mean[:, None]) * inv_std[:, None]

        w = tl.load(weight_ptr + d_off, mask=d_mask, other=0.0)     # fp32
        b = tl.load(bias_ptr   + d_off, mask=d_mask, other=0.0)     # fp32

        out = (normed * w + b).to(tl.float16)

        tl.store(out_ptr + idx, out,
                 mask=cell_mask[:, None] & d_mask[None, :])


# ----------------------------------------------------------------------
# 2️⃣  Fused projection, gating, optional mask and out‑gate
# ----------------------------------------------------------------------
@triton.jit
def _fused_proj_gate_mask_outgate_kernel(
    fused_ptr,          # fp16*   (total_cells, 5*hidden_dim) – fused linear output
    mask_ptr,           # fp16*   (total_cells)                – optional mask (0/1)
    left_out_ptr,       # fp16*   (B, hidden_dim, N, N)       – left side
    right_out_ptr,      # fp16*   (B, hidden_dim, N, N)       – right side
    out_gate_ptr,       # fp16*   (B, N, N, hidden_dim)       – out‑gate
    total_cells: tl.int32,
    hidden_dim: tl.constexpr,
    seq_len: tl.constexpr,
    # compile‑time constants
    BLOCK_CELL: tl.constexpr,
    BLOCK_H:    tl.constexpr,
    HAS_MASK:   tl.constexpr,
):
    pid_cell = tl.program_id(0)      # over B·N·N (cells)
    pid_h    = tl.program_id(1)      # over hidden dimension

    cell_off = pid_cell * BLOCK_CELL + tl.arange(0, BLOCK_CELL)   # (BLOCK_CELL,)
    h_off    = pid_h    * BLOCK_H    + tl.arange(0, BLOCK_H)      # (BLOCK_H,)

    cell_mask = cell_off < total_cells
    h_mask    = h_off    < hidden_dim
    active    = cell_mask[:, None] & h_mask[None, :]               # (BLOCK_CELL, BLOCK_H)

    # --------------------------------------------------------------
    # Layout of fused linear output: [total_cells, 5*hidden_dim]
    # --------------------------------------------------------------
    stride_fused = hidden_dim * 5
    base = cell_off[:, None] * stride_fused                         # (BLOCK_CELL, 1)

    # Load the five slices (proj, gate, out_gate) – each (BLOCK_CELL, BLOCK_H)
    left_raw       = tl.load(fused_ptr + base + h_off[None, :],                mask=active, other=0.0)
    left_gate_raw  = tl.load(fused_ptr + base + hidden_dim + h_off[None, :],   mask=active, other=0.0)
    right_raw      = tl.load(fused_ptr + base + 2*hidden_dim + h_off[None, :], mask=active, other=0.0)
    right_gate_raw = tl.load(fused_ptr + base + 3*hidden_dim + h_off[None, :], mask=active, other=0.0)
    out_gate_raw   = tl.load(fused_ptr + base + 4*hidden_dim + h_off[None, :], mask=active, other=0.0)

    # ------------------- sigmoid (FP32 for stability) -------------------
    left_gate  = (1.0 / (1.0 + tl.exp(-left_gate_raw .to(tl.float32)))).to(tl.float16)
    right_gate = (1.0 / (1.0 + tl.exp(-right_gate_raw .to(tl.float32)))).to(tl.float16)
    out_gate   = (1.0 / (1.0 + tl.exp(-out_gate_raw .to(tl.float32)))).to(tl.float16)

    # ------------------- apply gates ------------------------------------
    left  = left_raw  * left_gate
    right = right_raw * right_gate

    # ------------------- optional mask (broadcast over H) ---------------
    if HAS_MASK:
        m = tl.load(mask_ptr + cell_off,
                    mask=cell_mask,
                    other=tl.float16(1.0))[:, None]               # (BLOCK_CELL,1)
        left  *= m
        right *= m

    # ------------------- write left / right (B, H, N, N) ---------------
    cells_per_batch = seq_len * seq_len                 # N²
    b_idx  = cell_off // cells_per_batch
    ij_off = cell_off - b_idx * cells_per_batch          # i·N + j

    # flatten index for (B, H, N, N) = ((b*H + h) * N²) + pos
    out_idx = (b_idx[:, None] * hidden_dim + h_off[None, :]) * cells_per_batch + ij_off[:, None]

    tl.store(left_out_ptr  + out_idx, left,  mask=active)
    tl.store(right_out_ptr + out_idx, right, mask=active)

    # ------------------- write out‑gate (B, N, N, H) ------------------
    out_gate_idx = ((b_idx * cells_per_batch) + ij_off)[:, None] * hidden_dim + h_off[None, :]
    tl.store(out_gate_ptr + out_gate_idx, out_gate, mask=active)


# ----------------------------------------------------------------------
# 3️⃣  Final hidden‑dim LayerNorm + out‑gate + projection (fused)
# ----------------------------------------------------------------------
@triton.jit
def _final_norm_proj_kernel(
    mat_ptr,               # fp16*   (B, hidden_dim, N, N)   raw GEMM output
    out_gate_ptr,          # fp16*   (B, N, N, hidden_dim)   out‑gate
    proj_weight_ptr,       # fp16*   (dim, hidden_dim)       final linear weight
    ln_weight_ptr,         # fp32*   (hidden_dim,)           γ (to_out_norm)
    ln_bias_ptr,           # fp32*   (hidden_dim,)           β (to_out_norm)
    out_ptr,               # fp32*   (B, N, N, dim)         final output
    total_positions: tl.int32,   # B·N·N
    seq_len: tl.int32,           # N
    hidden_dim: tl.constexpr,
    dim: tl.constexpr,
    eps: tl.float32,
    # compile‑time constants
    BLOCK_POS: tl.constexpr,
    BLOCK_D_OUT: tl.constexpr,
):
    pid = tl.program_id(0)                # over positions
    pos_off = pid * BLOCK_POS + tl.arange(0, BLOCK_POS)          # (BLOCK_POS,)
    pos_mask = pos_off < total_positions

    seq_sq = seq_len * seq_len

    b_idx = pos_off // seq_sq
    pos_in_batch = pos_off - b_idx * seq_sq                       # i·N + j

    # ------------------------------------------------------------------
    # Load hidden‑dim vector from the raw GEMM output (FP16 → FP32)
    # ------------------------------------------------------------------
    h_range = tl.arange(0, hidden_dim)

    idx_hidden = b_idx[:, None] * hidden_dim * seq_sq + \
                 h_range[None, :] * seq_sq + \
                 pos_in_batch[:, None]                                 # (BLOCK_POS, hidden_dim)

    hidden_fp16 = tl.load(mat_ptr + idx_hidden,
                          mask=pos_mask[:, None],
                          other=0.0)                                 # fp16
    hidden = hidden_fp16.to(tl.float32)                             # fp32 for LN

    # ------------------------------------------------------------------
    # LayerNorm over hidden dimension (γ,β are fp32)
    # ------------------------------------------------------------------
    sum_  = tl.sum(hidden, axis=1)
    sumsq = tl.sum(hidden * hidden, axis=1)
    mean = sum_ / hidden_dim
    var  = sumsq / hidden_dim - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)                             # (BLOCK_POS,)

    normed = (hidden - mean[:, None]) * inv_std[:, None]           # (BLOCK_POS, hidden_dim)

    # load LN γ/β (fp32)
    ln_w = tl.load(ln_weight_ptr + h_range,
                   mask=h_range < hidden_dim,
                   other=0.0)                                       # fp32
    ln_b = tl.load(ln_bias_ptr   + h_range,
                   mask=h_range < hidden_dim,
                   other=0.0)                                       # fp32

    normed = normed * ln_w + ln_b                                 # (BLOCK_POS, hidden_dim)

    # ------------------------------------------------------------------
    # Multiply by out‑gate (layout B,N,N,hidden_dim)
    # ------------------------------------------------------------------
    idx_gate = ((b_idx * seq_sq) + pos_in_batch)[:, None] * hidden_dim + h_range[None, :]
    gate_fp16 = tl.load(out_gate_ptr + idx_gate,
                        mask=pos_mask[:, None],
                        other=0.0)                                 # fp16
    gate = gate_fp16.to(tl.float32)

    gated = normed * gate                                          # (BLOCK_POS, hidden_dim)

    # ------------------------------------------------------------------
    # Final projection: out = proj_weight @ gated   (dim × hidden_dim)
    # ------------------------------------------------------------------
    d_range = tl.arange(0, BLOCK_D_OUT)
    for d_blk in range(0, tl.cdiv(dim, BLOCK_D_OUT)):
        d_start = d_blk * BLOCK_D_OUT
        d_off   = d_start + d_range                     # (BLOCK_D_OUT,)
        d_mask  = d_off < dim

        # weight slice: (BLOCK_D_OUT, hidden_dim) row‑major → index = d * hidden_dim + h
        w_idx = d_off[:, None] * hidden_dim + h_range[None, :]   # (BLOCK_D_OUT, hidden_dim)
        w = tl.load(proj_weight_ptr + w_idx,
                    mask=d_mask[:, None] & (h_range[None, :] < hidden_dim),
                    other=0.0)                                   # fp16

        # dot: (BLOCK_POS, hidden_dim) * (hidden_dim, BLOCK_D_OUT) -> (BLOCK_POS, BLOCK_D_OUT)
        out_block_fp16 = tl.dot(gated.to(tl.float16), w.T)       # (BLOCK_POS, BLOCK_D_OUT)
        out_block = out_block_fp16.to(tl.float32)

        out_idx = ((b_idx * seq_sq) + pos_in_batch)[:, None] * dim + d_off[None, :]
        tl.store(out_ptr + out_idx,
                 out_block,
                 mask=pos_mask[:, None] & d_mask[None, :])


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the *outgoing* TriMul operator (no gradients).

    Parameters
    ----------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor  [B, N, N, dim] (float32)
        - mask         : torch.Tensor  [B, N, N]    (float32 / bool) or None
        - weights      : dict of torch.Tensors containing model parameters
        - config       : dict with keys "dim", "hidden_dim", optional "nomask"

    Returns
    -------
    torch.Tensor
        [B, N, N, dim] (float32)
    """
    # --------------------------------------------------------------
    # unpack
    # --------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    B, N, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)

    # --------------------------------------------------------------
    # 1️⃣ Input LayerNorm (fp32 → fp16)
    # --------------------------------------------------------------
    total_cells = B * N * N
    x_norm_fp16 = torch.empty((B, N, N, dim), dtype=torch.float16, device=device)

    BLOCK_CELL = 64   # #cells handled per program
    BLOCK_D    = 64   # #channels handled per tile

    _layernorm_fp32_to_fp16_kernel[
        (triton.cdiv(total_cells, BLOCK_CELL),)
    ](
        input_tensor.view(-1).contiguous(),
        x_norm_fp16.view(-1).contiguous(),
        weights["norm.weight"],
        weights["norm.bias"],
        total_cells,
        1e-5,
        BLOCK_CELL=BLOCK_CELL,
        BLOCK_D=BLOCK_D,
        DIM=dim,
    )

    # --------------------------------------------------------------
    # 2️⃣ Fused projection, gating, optional mask & out‑gate
    # --------------------------------------------------------------
    # Stack the five linear weight matrices (shape: (5*hidden_dim, dim))
    fused_w = torch.cat(
        [
            weights["left_proj.weight"],
            weights["left_gate.weight"],
            weights["right_proj.weight"],
            weights["right_gate.weight"],
            weights["out_gate.weight"],
        ],
        dim=0,
    ).to(torch.float16).contiguous()

    # Linear on the normalized input (still fp16) → (total_cells, 5*hidden_dim)
    fused_out = F.linear(
        x_norm_fp16.view(total_cells, dim),   # (total_cells, dim)  fp16
        fused_w,                               # (5*hidden_dim, dim) fp16
    )                                          # → (total_cells, 5*hidden_dim)

    # Storage for the three tensors produced by the fused kernel
    left_out  = torch.empty((B, hidden_dim, N, N), dtype=torch.float16, device=device)
    right_out = torch.empty((B, hidden_dim, N, N), dtype=torch.float16, device=device)
    out_gate  = torch.empty((B, N, N, hidden_dim), dtype=torch.float16, device=device)

    # Optional mask handling
    if mask is not None and not nomask:
        mask_fp16 = mask.to(torch.float16).contiguous().view(-1)   # (total_cells,)
        mask_ptr = mask_fp16
        HAS_MASK = 1
    else:
        # dummy pointer – never dereferenced when HAS_MASK == 0
        mask_ptr = left_out.view(-1)
        HAS_MASK = 0

    BLOCK_H = 64
    grid = (
        triton.cdiv(total_cells, BLOCK_CELL),   # programs over cells
        triton.cdiv(hidden_dim, BLOCK_H),       # programs over hidden dimension
    )
    _fused_proj_gate_mask_outgate_kernel[grid](
        fused_out.view(-1),
        mask_ptr,
        left_out.view(-1),
        right_out.view(-1),
        out_gate.view(-1),
        total_cells,
        hidden_dim,
        N,
        BLOCK_CELL=BLOCK_CELL,
        BLOCK_H=BLOCK_H,
        HAS_MASK=HAS_MASK,
    )

    # --------------------------------------------------------------
    # 3️⃣ Batched GEMM   left @ rightᵀ   (B·hidden_dim, N, N) → (B·hidden_dim, N, N)
    # --------------------------------------------------------------
    left_mat  = left_out.reshape(B * hidden_dim, N, N)      # (B*H, N, N)
    right_mat = right_out.reshape(B * hidden_dim, N, N)     # (B*H, N, N)

    out_mat = torch.bmm(left_mat, right_mat.transpose(1, 2))   # (B*H, N, N)
    out_raw = out_mat.view(B, hidden_dim, N, N)                # (B, H, N, N)  fp16

    # --------------------------------------------------------------
    # 4️⃣ Final hidden‑dim LayerNorm + out‑gate + projection (fused)
    # --------------------------------------------------------------
    out = torch.empty((B, N, N, dim), dtype=torch.float32, device=device)

    BLOCK_POS   = 64   # positions per program
    BLOCK_D_OUT = 32   # output‑dim tile size

    total_positions = B * N * N
    _final_norm_proj_kernel[
        (triton.cdiv(total_positions, BLOCK_POS),)
    ](
        out_raw.view(-1),                                      # mat_ptr (fp16)
        out_gate.view(-1),                                     # out_gate_ptr (fp16)
        weights["to_out.weight"].to(torch.float16).contiguous().view(-1),  # proj weight
        weights["to_out_norm.weight"],                         # LN γ (fp32)
        weights["to_out_norm.bias"],                           # LN β (fp32)
        out.view(-1),                                          # final output (fp32)
        total_positions,
        N,
        hidden_dim=hidden_dim,
        dim=dim,
        eps=1e-5,
        BLOCK_POS=BLOCK_POS,
        BLOCK_D_OUT=BLOCK_D_OUT,
    )

    return out


# ----------------------------------------------------------------------
# Optional sanity‑check (executed only when the file is run directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal correctness test against the reference implementation
    torch.manual_seed(0)
    bs, N, dim, hidden_dim = 1, 16, 128, 128
    x = torch.randn(bs, N, N, dim, device="cuda", dtype=torch.float32)
    mask = (torch.rand(bs, N, N, device="cuda") > 0.5).float()

    # Build a reference TriMul module
    class RefTriMul(torch.nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.norm = torch.nn.LayerNorm(dim)
            self.left_proj  = torch.nn.Linear(dim, hidden_dim, bias=False)
            self.right_proj = torch.nn.Linear(dim, hidden_dim, bias=False)
            self.left_gate   = torch.nn.Linear(dim, hidden_dim, bias=False)
            self.right_gate  = torch.nn.Linear(dim, hidden_dim, bias=False)
            self.out_gate    = torch.nn.Linear(dim, hidden_dim, bias=False)
            self.to_out_norm = torch.nn.LayerNorm(hidden_dim)
            self.to_out      = torch.nn.Linear(hidden_dim, dim, bias=False)

        def forward(self, x, mask):
            x = self.norm(x)
            left  = self.left_proj(x) * mask.unsqueeze(-1)
            right = self.right_proj(x) * mask.unsqueeze(-1)
            left_gate  = self.left_gate(x).sigmoid()
            right_gate = self.right_gate(x).sigmoid()
            out_gate   = self.out_gate(x).sigmoid()
            left  = left * left_gate
            right = right * right_gate
            out = torch.einsum('...ikd,...jkd->...ijd', left, right)
            out = self.to_out_norm(out)
            out = out * out_gate
            return self.to_out(out)

    ref = RefTriMul(dim, hidden_dim).cuda()
    # collect all weights into a dict compatible with `custom_kernel`
    w = {
        "norm.weight":          ref.norm.weight,
        "norm.bias":            ref.norm.bias,
        "left_proj.weight":     ref.left_proj.weight,
        "right_proj.weight":    ref.right_proj.weight,
        "left_gate.weight":     ref.left_gate.weight,
        "right_gate.weight":    ref.right_gate.weight,
        "out_gate.weight":      ref.out_gate.weight,
        "to_out_norm.weight":   ref.to_out_norm.weight,
        "to_out_norm.bias":     ref.to_out_norm.bias,
        "to_out.weight":        ref.to_out.weight,
    }
    cfg = {"dim": dim, "hidden_dim": hidden_dim, "nomask": False}

    out_ref = ref(x, mask)
    out_tri = custom_kernel((x, mask, w, cfg))

    # sanity check – the two results should be very close
    torch.testing.assert_allclose(out_tri, out_ref, rtol=1e-3, atol=1e-3)
    print("Sanity check passed!")