# --------------------------------------------------------------
# Triton‑based implementation of the outgoing TriMul operator
# --------------------------------------------------------------
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# 1️⃣  LayerNorm (float32 → float16)
# ------------------------------------------------------------------
@triton.jit
def _layernorm_fp32_to_fp16_kernel(
    x_ptr,            # float32* (B·N·N·C) flattened
    out_ptr,          # float16* (same shape)
    weight_ptr,       # float32* (C,)
    bias_ptr,         # float32* (C,)
    total_cells: tl.int32,   # B·N·N
    eps: tl.float32,
    # compile‑time constants
    BLOCK_CELL: tl.constexpr,   # how many cells each program processes
    BLOCK_D:    tl.constexpr,   # channel tile size
    DIM:        tl.constexpr,   # C
):
    pid = tl.program_id(0)

    # ----- cell (position) handling -----
    cell_off = pid * BLOCK_CELL + tl.arange(0, BLOCK_CELL)   # (BLOCK_CELL,)
    cell_mask = cell_off < total_cells

    # ----- compute mean / variance over the channel dim -----
    sum_  = tl.zeros((BLOCK_CELL,), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_CELL,), dtype=tl.float32)

    n_dblocks = tl.cdiv(DIM, BLOCK_D)
    for db in range(0, n_dblocks):
        d_start = db * BLOCK_D
        d_off   = d_start + tl.arange(0, BLOCK_D)
        d_mask  = d_off < DIM

        # flat index = cell * C + channel
        idx = cell_off[:, None] * DIM + d_off[None, :]               # (BLOCK_CELL, BLOCK_D)
        vals = tl.load(
            x_ptr + idx,
            mask=cell_mask[:, None] & d_mask[None, :],
            other=0.0,
        )   # fp32
        sum_  += tl.sum(vals, axis=1)
        sumsq += tl.sum(vals * vals, axis=1)

    mean = sum_ / DIM
    var  = sumsq / DIM - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)   # (BLOCK_CELL,)

    # ----- affine + store (cast to fp16) -----
    for db in range(0, n_dblocks):
        d_start = db * BLOCK_D
        d_off   = d_start + tl.arange(0, BLOCK_D)
        d_mask  = d_off < DIM

        idx = cell_off[:, None] * DIM + d_off[None, :]  # (BLOCK_CELL, BLOCK_D)
        vals = tl.load(
            x_ptr + idx,
            mask=cell_mask[:, None] & d_mask[None, :],
            other=0.0,
        )   # fp32

        normed = (vals - mean[:, None]) * inv_std[:, None]

        w = tl.load(weight_ptr + d_off, mask=d_mask, other=0.0)   # fp32
        b = tl.load(bias_ptr   + d_off, mask=d_mask, other=0.0)   # fp32

        out = (normed * w + b).to(tl.float16)

        tl.store(out_ptr + idx, out,
                 mask=cell_mask[:, None] & d_mask[None, :])


# ------------------------------------------------------------------
# 2️⃣  Projection + gating (+ optional mask)
# ------------------------------------------------------------------
@triton.jit
def _proj_gate_mask_kernel(
    x_norm_ptr,                # fp16* (total_cells, C)
    left_out_ptr,              # fp16* (B, H, N, N)
    right_out_ptr,             # fp16* (B, H, N, N)
    out_gate_ptr,              # fp16* (B, N, N, H)
    mask_ptr,                  # fp16* (total_cells,) – optional
    # weight pointers (C, H)
    left_proj_w_ptr,
    left_gate_w_ptr,
    right_proj_w_ptr,
    right_gate_w_ptr,
    out_gate_w_ptr,
    total_cells: tl.int32,
    hidden_dim: tl.int32,
    dim: tl.int32,
    seq_len: tl.int32,
    # compile‑time constants
    BLOCK_CELL: tl.constexpr,
    BLOCK_H:    tl.constexpr,
    BLOCK_D:    tl.constexpr,
    HAS_MASK:   tl.constexpr,
):
    # ----- grid over (cells, hidden dim) -----
    pid_cell = tl.program_id(0)
    pid_h    = tl.program_id(1)

    cell_off = pid_cell * BLOCK_CELL + tl.arange(0, BLOCK_CELL)   # (BLOCK_CELL,)
    h_off    = pid_h    * BLOCK_H    + tl.arange(0, BLOCK_H)      # (BLOCK_H,)

    cell_mask = cell_off < total_cells
    h_mask    = h_off    < hidden_dim
    active    = cell_mask[:, None] & h_mask[None, :]               # (BLOCK_CELL, BLOCK_H)

    # ----- accumulation buffers (fp32) -----
    left_acc      = tl.zeros((BLOCK_CELL, BLOCK_H), dtype=tl.float32)
    left_gate_acc = tl.zeros((BLOCK_CELL, BLOCK_H), dtype=tl.float32)
    right_acc     = tl.zeros((BLOCK_CELL, BLOCK_H), dtype=tl.float32)
    right_gate_acc= tl.zeros((BLOCK_CELL, BLOCK_H), dtype=tl.float32)
    out_gate_acc  = tl.zeros((BLOCK_CELL, BLOCK_H), dtype=tl.float32)

    n_dim_blocks = tl.cdiv(dim, BLOCK_D)
    for db in range(0, n_dim_blocks):
        d_start = db * BLOCK_D
        d_off   = d_start + tl.arange(0, BLOCK_D)
        d_mask  = d_off < dim

        # ----- load normalized input (fp16) -----
        x = tl.load(
            x_norm_ptr + cell_off[:, None] * dim + d_off[None, :],
            mask=cell_mask[:, None] & d_mask[None, :],
            other=0.0,
        )   # (BLOCK_CELL, BLOCK_D) fp16

        # ----- load weight slices (fp16) -----
        left_proj_w   = tl.load(
            left_proj_w_ptr + d_off[:, None] * hidden_dim + h_off[None, :],
            mask=d_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        left_gate_w   = tl.load(
            left_gate_w_ptr + d_off[:, None] * hidden_dim + h_off[None, :],
            mask=d_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        right_proj_w  = tl.load(
            right_proj_w_ptr + d_off[:, None] * hidden_dim + h_off[None, :],
            mask=d_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        right_gate_w  = tl.load(
            right_gate_w_ptr + d_off[:, None] * hidden_dim + h_off[None, :],
            mask=d_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        out_gate_w    = tl.load(
            out_gate_w_ptr + d_off[:, None] * hidden_dim + h_off[None, :],
            mask=d_mask[:, None] & h_mask[None, :],
            other=0.0,
        )

        # ----- mat‑vec (fp32 accumulation) -----
        left_acc      += tl.dot(x, left_proj_w).to(tl.float32)
        left_gate_acc += tl.dot(x, left_gate_w).to(tl.float32)
        right_acc     += tl.dot(x, right_proj_w).to(tl.float32)
        right_gate_acc+= tl.dot(x, right_gate_w).to(tl.float32)
        out_gate_acc  += tl.dot(x, out_gate_w).to(tl.float32)

    # ----- apply sigmoid (stay in fp16) -----
    left_gate  = (1.0 / (1.0 + tl.exp(-left_gate_acc))).to(tl.float16)
    right_gate = (1.0 / (1.0 + tl.exp(-right_gate_acc))).to(tl.float16)
    out_gate   = (1.0 / (1.0 + tl.exp(-out_gate_acc))).to(tl.float16)

    left  = left_acc.to(tl.float16)   * left_gate
    right = right_acc.to(tl.float16)  * right_gate

    # ----- optional mask (broadcast on hidden dim) -----
    if HAS_MASK:
        m = tl.load(mask_ptr + cell_off,
                    mask=cell_mask,
                    other=tl.float16(1.0))[:, None]   # (BLOCK_CELL,1)
        left  *= m
        right *= m

    # ----- write left / right (layout B, H, N, N) -----
    cells_per_batch = seq_len * seq_len               # N²
    b_idx  = cell_off // cells_per_batch              # (BLOCK_CELL,)
    ij_off = cell_off - b_idx * cells_per_batch       # (BLOCK_CELL,)

    # offset = ((b * H + h) * N²) + (i*N + j)
    out_off = ((b_idx[:, None] * hidden_dim + h_off[None, :]) * cells_per_batch) + ij_off[:, None]

    tl.store(left_out_ptr  + out_off, left,  mask=active)
    tl.store(right_out_ptr + out_off, right, mask=active)

    # ----- write out_gate (layout B, N, N, H) -----
    out_gate_idx = (cell_off[:, None] * hidden_dim) + h_off[None, :]
    tl.store(out_gate_ptr + out_gate_idx, out_gate, mask=active)


# ------------------------------------------------------------------
# 3️⃣  Final LayerNorm + out‑gate + output projection
# ------------------------------------------------------------------
@triton.jit
def _final_norm_proj_kernel(
    mat_ptr,                # fp16* (B, H, N, N) – result of batched bmm
    out_gate_ptr,           # fp16* (B, N, N, H)
    proj_weight_ptr,        # fp16* (C, H)
    ln_weight_ptr,          # fp32* (H,)
    ln_bias_ptr,            # fp32* (H,)
    out_ptr,                # fp32* (B, N, N, C)
    total_positions: tl.int32,   # B·N·N
    seq_len: tl.int32,           # N
    hidden_dim: tl.constexpr,
    dim: tl.constexpr,
    eps: tl.float32,
    # compile‑time constants
    BLOCK_POS: tl.constexpr,
    BLOCK_D_OUT: tl.constexpr,
):
    pid = tl.program_id(0)                         # over positions (B·N·N)
    pos_off = pid * BLOCK_POS + tl.arange(0, BLOCK_POS)   # (BLOCK_POS,)
    pos_mask = pos_off < total_positions

    seq_sq = seq_len * seq_len

    # ----- decode batch / (i,j) from flat position -----
    b_idx = pos_off // seq_sq
    pos_in_batch = pos_off - b_idx * seq_sq          # i*N + j

    # ----- load hidden vector (fp16 → fp32) -----
    h_range = tl.arange(0, hidden_dim)               # (H,)
    # layout (B, H, N, N): ((b * H + h) * N²) + (i·N + j)
    idx_hidden = b_idx[:, None] * hidden_dim * seq_sq + \
                 h_range[None, :] * seq_sq + \
                 pos_in_batch[:, None]               # (BLOCK_POS, H)

    hidden_fp16 = tl.load(
        mat_ptr + idx_hidden,
        mask=pos_mask[:, None] & (h_range[None, :] < hidden_dim),
        other=0.0,
    )   # fp16
    hidden = hidden_fp16.to(tl.float32)               # fp32

    # ----- LayerNorm over hidden dimension -----
    sum_  = tl.sum(hidden, axis=1)
    sumsq = tl.sum(hidden * hidden, axis=1)
    mean = sum_ / hidden_dim
    var  = sumsq / hidden_dim - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    normed = (hidden - mean[:, None]) * inv_std[:, None]   # (BLOCK_POS, H)

    # γ, β (fp32)
    ln_w = tl.load(ln_weight_ptr + h_range,
                    mask=h_range < hidden_dim,
                    other=0.0)
    ln_b = tl.load(ln_bias_ptr + h_range,
                    mask=h_range < hidden_dim,
                    other=0.0)

    normed = normed * ln_w + ln_b                         # (BLOCK_POS, H)

    # ----- apply out‑gate -----
    idx_gate = ((b_idx * seq_sq) + pos_in_batch)[:, None] * hidden_dim + h_range[None, :]
    gate_fp16 = tl.load(
        out_gate_ptr + idx_gate,
        mask=pos_mask[:, None] & (h_range[None, :] < hidden_dim),
        other=0.0,
    )   # fp16
    gate = gate_fp16.to(tl.float32)

    gated = normed * gate                                 # (BLOCK_POS, H)

    # ----- final linear projection (fp16 mat‑mul) -----
    gated_fp16 = gated.to(tl.float16)                     # fp16 for tensor cores

    d_range = tl.arange(0, BLOCK_D_OUT)
    d_mask  = d_range < dim

    for d_blk in range(0, tl.cdiv(dim, BLOCK_D_OUT)):
        d_start = d_blk * BLOCK_D_OUT
        d_off   = d_start + d_range                     # (BLOCK_D_OUT,)

        # weight slice (C, H) → offset = d*H + h
        w_idx = d_off[:, None] * hidden_dim + h_range[None, :]   # (BLOCK_D_OUT, H)
        w = tl.load(
            proj_weight_ptr + w_idx,
            mask=d_mask[:, None] & (h_range[None, :] < hidden_dim),
            other=0.0,
        )   # fp16

        # (BLOCK_POS, H) @ (H, BLOCK_D_OUT) → (BLOCK_POS, BLOCK_D_OUT)
        out_block_fp16 = tl.dot(gated_fp16, w.T).to(tl.float16)
        out_block = out_block_fp16.to(tl.float32)

        # write to layout (B,N,N,C)
        out_idx = ((b_idx * seq_sq) + pos_in_batch)[:, None] * dim + d_off[None, :]
        tl.store(
            out_ptr + out_idx,
            out_block,
            mask=pos_mask[:, None] & d_mask[None, :],
        )


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
        - input_tensor : torch.Tensor  [B, N, N, C] (float32)
        - mask         : torch.Tensor  [B, N, N]   (float32/bool) or None
        - weights      : dict of torch.Tensors containing model parameters
        - config       : dict with keys "dim", "hidden_dim", optional "nomask"

    Returns
    -------
    torch.Tensor : [B, N, N, C] (float32)
    """
    # --------------------------------------------------------------
    # Unpack inputs
    # --------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    B, N, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)

    # --------------------------------------------------------------
    # 1️⃣ First LayerNorm (float32 → float16)
    # --------------------------------------------------------------
    total_cells = B * N * N
    x_norm_fp16 = torch.empty((B, N, N, dim), dtype=torch.float16, device=device)

    BLOCK_CELL = 64   # number of positions processed per program
    BLOCK_D    = 64   # channel tile size in LN

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
    # 2️⃣ Projection + gating (+ optional mask)
    # --------------------------------------------------------------
    left_out  = torch.empty((B, hidden_dim, N, N), dtype=torch.float16, device=device)
    right_out = torch.empty((B, hidden_dim, N, N), dtype=torch.float16, device=device)
    out_gate  = torch.empty((B, N, N, hidden_dim), dtype=torch.float16, device=device)

    # weight layout: (C, H) – convenient for the kernel
    left_proj_w_T  = weights["left_proj.weight"].t().contiguous().to(torch.float16)
    left_gate_w_T  = weights["left_gate.weight"].t().contiguous().to(torch.float16)
    right_proj_w_T = weights["right_proj.weight"].t().contiguous().to(torch.float16)
    right_gate_w_T = weights["right_gate.weight"].t().contiguous().to(torch.float16)
    out_gate_w_T   = weights["out_gate.weight"].t().contiguous().to(torch.float16)

    if (mask is not None) and (not nomask):
        # mask is broadcasted over the hidden dimension
        mask_fp16 = mask.to(torch.float16).contiguous().view(-1)   # (total_cells,)
        mask_ptr = mask_fp16
        HAS_MASK = 1
    else:
        # dummy pointer – never dereferenced because HAS_MASK=0
        mask_ptr = left_out.view(-1)
        HAS_MASK = 0

    BLOCK_H = 64          # tile over hidden dim (covers hidden_dim=128, also works for 384)
    BLOCK_D_PROJ = 32     # dim tile for projection kernel

    grid = (
        triton.cdiv(total_cells, BLOCK_CELL),   # over cells (B·N·N)
        triton.cdiv(hidden_dim, BLOCK_H),       # over hidden dim
    )
    _proj_gate_mask_kernel[grid](
        x_norm_fp16.view(-1),
        left_out.view(-1),
        right_out.view(-1),
        out_gate.view(-1),
        mask_ptr,
        left_proj_w_T.view(-1),
        left_gate_w_T.view(-1),
        right_proj_w_T.view(-1),
        right_gate_w_T.view(-1),
        out_gate_w_T.view(-1),
        total_cells,
        hidden_dim,
        dim,
        N,
        BLOCK_CELL=BLOCK_CELL,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D_PROJ,
        HAS_MASK=HAS_MASK,
    )

    # --------------------------------------------------------------
    # 3️⃣ Batched GEMM  left @ rightᵀ  (B·hidden_dim, N, N)
    # --------------------------------------------------------------
    left_mat  = left_out.reshape(B * hidden_dim, N, N)          # fp16
    right_mat = right_out.reshape(B * hidden_dim, N, N)         # fp16

    out_mat = torch.bmm(left_mat, right_mat.transpose(1, 2))   # fp16, shape (B*hidden_dim, N, N)
    out_raw = out_mat.view(B, hidden_dim, N, N)                # (B, H, N, N)

    # --------------------------------------------------------------
    # 4️⃣ Final LayerNorm + out‑gate + output projection
    # --------------------------------------------------------------
    out = torch.empty((B, N, N, dim), dtype=torch.float32, device=device)

    BLOCK_POS   = 64    # positions per program
    BLOCK_D_OUT = 64    # output‑dim tile size

    _final_norm_proj_kernel[
        (triton.cdiv(total_cells, BLOCK_POS),)
    ](
        out_raw.view(-1),   # mat_ptr (fp16)
        out_gate.view(-1),  # out_gate_ptr (fp16)
        weights["to_out.weight"].contiguous().to(torch.float16).view(-1),   # proj_weight_ptr
        weights["to_out_norm.weight"],   # LN γ (fp32)
        weights["to_out_norm.bias"],     # LN β (fp32)
        out.view(-1),                    # final output (fp32)
        total_cells,
        N,
        hidden_dim=hidden_dim,
        dim=dim,
        eps=1e-5,
        BLOCK_POS=BLOCK_POS,
        BLOCK_D_OUT=BLOCK_D_OUT,
    )

    return out