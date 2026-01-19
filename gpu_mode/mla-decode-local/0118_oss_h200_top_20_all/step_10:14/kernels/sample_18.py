### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# Helper utilities (rotate‑half, RoPE table, Triton soft‑max)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


_rope_cache: dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_rope_tbl(dim: int, max_seq_len: int, device: torch.device):
    """
    Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    Cached lazily.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                          dtype=torch.float32,
                                          device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                     # (max_seq_len, 1)
        idx = pos * theta[None, :]                                          # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                 # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# Triton softmax – used only for the generic (NoPE) path
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,      # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur,
                      mask=mask,
                      other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur,
                      mask=mask,
                      other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur,
                 tl.cast(exp_val, tl.bfloat16),
                 mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur,
                      mask=mask,
                      other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur,
                 tl.cast(norm, tl.bfloat16),
                 mask=mask)


def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax on a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
    # pick a sensible block size
    if n_cols <= 32:
        BLOCK_SIZE = 32
    elif n_cols <= 64:
        BLOCK_SIZE = 64
    elif n_cols <= 128:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    out = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_kernel[grid](
        out,
        x,
        out.stride(0),
        x.stride(0),
        N=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------------
# Fast path (dnope == 0) – combines the two down‑projections into a single GEMM
# ----------------------------------------------------------------------
def _forward_fast_combined(
    x: torch.Tensor,                     # (bs, sl, d)
    kv_data: torch.Tensor,               # (bs, max_seq_len, dkv+drope)
    cur_len: int,                        # already‑cached length
    w_comb: torch.Tensor,                # (dq+dkv+drope, d)
    wUQ: torch.Tensor,                   # ((dnope+drope)*nh, dq)   -> here dnope==0 → (nh*drope, dq)
    wUKV: torch.Tensor,                  # ((dnope+dv)*nh, dkv)    -> (nh*dv, dkv)
    wO: torch.Tensor,                    # (d, nh*dv)
    cos_tbl: torch.Tensor,
    sin_tbl: torch.Tensor,
    nh: int, dnope: int, drope: int,
    dkv: int, dv: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fast‑path where dnope == 0.
    Returns (output, updated_kv, new_len).
    """
    bs, sl, d = x.shape
    total_down = w_comb.shape[0]                 # dq + dkv + drope
    dq = w_comb.shape[0] - dkv - drope           # recover dq

    # --------------------------------------------------------------
    # 1) Combined down‑projection (Q‑down + KV‑down)
    # --------------------------------------------------------------
    # (bs, sl, total_down) = x @ w_comb.T
    down = F.linear(x, w_comb)                   # bf16

    # split
    q_lora     = down[..., :dq]                                 # (bs, sl, dq)
    kv_latent  = down[..., dq:dq + dkv]                         # (bs, sl, dkv)
    kv_rope_raw = down[..., dq + dkv:]                          # (bs, sl, drope)

    # --------------------------------------------------------------
    # 2) Write KV‑cache (rotate rope part on‑the‑fly)
    # --------------------------------------------------------------
    new_len = cur_len + sl
    pos_range = torch.arange(cur_len, new_len, device=x.device, dtype=torch.long)   # (sl,)
    cos_pos = cos_tbl[pos_range]                              # (sl, drope)
    sin_pos = sin_tbl[pos_range]                              # (sl, drope)

    # rotate‑half + rope
    kv_rope_rot = kv_rope_raw * cos_pos[None, :, :] + \
                  _rotate_half(kv_rope_raw) * sin_pos[None, :, :]      # (bs, sl, drope)

    kv_new = torch.cat([kv_latent, kv_rope_rot], dim=-1)          # (bs, sl, dkv+drope)
    kv_data[:, cur_len:new_len, :] = kv_new
    # --------------------------------------------------------------
    # 3) Up‑project queries (rope part only)
    # --------------------------------------------------------------
    # (bs, sl, nh*drope) = q_lora @ wUQ.T
    q_up = F.linear(q_lora, wUQ)                                 # (bs, sl, nh*drope)
    q_up = q_up.view(bs, sl, nh, drope).permute(0, 2, 1, 3)      # (bs, nh, sl, drope)

    # query rope – we follow the original implementation (only last token gets RoPE)
    q_pos = new_len - 1                                            # scalar position of the new token
    cos_q = cos_tbl[q_pos].view(1, 1, 1, drope)                  # (1,1,1,drope)
    sin_q = sin_tbl[q_pos].view(1, 1, 1, drope)
    q = q_up * cos_q + _rotate_half(q_up) * sin_q                # (bs, nh, sl, drope)

    # --------------------------------------------------------------
    # 4) Keys / Values from KV‑cache (already rotated for keys)
    # --------------------------------------------------------------
    kv_cached = kv_data[:, :new_len, :]                                   # (bs, new_len, dkv+drope)

    k = kv_cached[..., dkv:]                         # (bs, new_len, drope)
    k = k[:, None, :, :].expand(-1, nh, -1, -1)      # (bs, nh, new_len, drope)

    v = kv_cached[..., :dkv]                         # (bs, new_len, dkv)
    v = v[:, None, :, :].expand(-1, nh, -1, -1)      # (bs, nh, new_len, dkv)

    # --------------------------------------------------------------
    # 5) Scaled dot‑product attention (flash‑attention via PyTorch)
    # --------------------------------------------------------------
    scale_qk = 1.0 / math.sqrt(drope)                # dnope == 0
    attn_out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale_qk,
    )                                                # (bs, nh, sl, dkv)

    # --------------------------------------------------------------
    # 6) Project aggregated latent vectors to the value space
    # --------------------------------------------------------------
    # wUKV : ((dnope+dv) * nh, dkv)  ; with dnope==0  → (nh*dv, dkv)
    wV = wUKV.view(nh, dv, dkv)                     # (nh, dv, dkv)

    attn_out_flat = attn_out.reshape(-1, dkv)        # (bs*nh*sl, dkv)
    y_head = F.linear(attn_out_flat, wV.reshape(-1, dkv))   # (bs*nh*sl, dv)
    y_head = y_head.view(bs, nh, sl, dv)                     # (bs, nh, sl, dv)

    # --------------------------------------------------------------
    # 7) Final output projection
    # --------------------------------------------------------------
    y_head = y_head.permute(0, 2, 1, 3).contiguous()   # (bs, sl, nh, dv)
    y_head = y_head.view(bs, sl, nh * dv)              # (bs, sl, nh*dv)
    out = F.linear(y_head, wO)                         # (bs, sl, d)

    return out, kv_data, new_len


# ----------------------------------------------------------------------
# General (NoPE) path – fallback to the reference implementation
# ----------------------------------------------------------------------
def _forward_general(
    x: torch.Tensor,
    kv_data: torch.Tensor,
    cur_len: int,
    wDQ: torch.Tensor, wDKV: torch.Tensor,
    wUQ: torch.Tensor, wUKV: torch.Tensor,
    wO: torch.Tensor,
    cos_tbl: torch.Tensor, sin_tbl: torch.Tensor,
    nh: int, dnope: int, drope: int,
    dkv: int, dv: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fully‑featured path supporting a non‑zero NoPE dimension.
    Mirrors the reference implementation (no Triton acceleration) and
    is used only when ``dnope > 0`` – the common benchmark configuration
    never takes this branch.
    """
    bs, sl, d = x.shape

    # 1) Down‑project
    q_lora = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora = F.linear(x, wDKV)                   # (bs, sl, dkv + drope)

    # 2) Update KV‑cache (store raw rope part – we will apply RoPE later)
    kv_latent_new = kv_lora[..., :dkv]            # (bs, sl, dkv)
    kv_rope_raw   = kv_lora[..., dkv:]            # (bs, sl, drope)
    kv_new = torch.cat([kv_latent_new, kv_rope_raw], dim=-1)     # (bs, sl, dkv+drope)
    kv_data[:, cur_len:cur_len + sl, :] = kv_new
    new_len = cur_len + sl

    # 3) Up‑project queries (NoPE + RoPE)
    q_up = F.linear(q_lora.squeeze(1), wUQ)       # (bs, nh*(dnope+drope))
    q_up = q_up.view(bs, nh, dnope + drope)      # (bs, nh, dnope+drope)
    q_nope, q_rope = torch.split(q_up, [dnope, drope], dim=-1)

    # 4) KV split
    kv_all = kv_data[:, :new_len, :]               # (bs, new_len, dkv+drope)
    kv_nope = kv_all[..., :dkv]                    # (bs, new_len, dkv)
    kv_rope = kv_all[..., dkv:]                    # (bs, new_len, drope)

    # 5) RoPE for queries
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos].view(1, 1, drope)
    sin_q = sin_tbl[q_pos].view(1, 1, drope)
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q

    # 6) RoPE for keys (all positions)
    cos_k = cos_tbl[:new_len].unsqueeze(0)               # (1, new_len, drope)
    sin_k = sin_tbl[:new_len].unsqueeze(0)
    k_rope = kv_rope * cos_k + _rotate_half(kv_rope) * sin_k   # (bs, new_len, drope)

    # 7) NoPE projections
    # wUKV : ((dnope + dv) * nh, dkv)  → split into K‑proj and V‑proj per head
    wUKV_view = wUKV.view(nh, dnope + dv, dkv)               # (nh, dnope+dv, dkv)
    wK = wUKV_view[:, :dnope, :] if dnope > 0 else None      # (nh, dnope, dkv)
    wV_T = wUKV_view[:, dnope:, :].permute(0, 2, 1)         # (nh, dkv, dv)

    # 8) Query NoPE → latent space (if present)
    if dnope > 0:
        q_nope_latent = torch.einsum('b h d, h d k -> b h k',
                                      q_nope, wK)            # (bs, nh, dkv)
    else:
        q_nope_latent = torch.zeros((bs, nh, dkv),
                                    dtype=torch.bfloat16,
                                    device=x.device)

    # 9) Scores
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))          # (bs, nh, new_len)
    scores_nope = torch.matmul(q_nope_latent, kv_nope.transpose(-2, -1))  # (bs, nh, new_len)
    scale = 1.0 / math.sqrt(dnope + drope)
    scores = (scores_rope + scores_nope) * scale

    # 10) Soft‑max (Triton implementation for bf16)
    scores_flat = scores.reshape(bs * nh, new_len)
    attn_flat = _triton_softmax(scores_flat)
    attn = attn_flat.view(bs, nh, new_len)          # (bs, nh, new_len)

    # 11) Weighted sum over latent vectors
    latent_agg = torch.matmul(attn, kv_nope)        # (bs, nh, dkv)

    # 12) Project to value space (per‑head linear)
    y_head = torch.einsum('b h k, h k v -> b h v',
                          latent_agg, wV_T)       # (bs, nh, dv)

    # 13) Output projection
    y_head_flat = y_head.reshape(bs, nh * dv)       # (bs, nh*dv)
    out = F.linear(y_head_flat, wO)                 # (bs, d)
    out = out.unsqueeze(1)                          # (bs, 1, d)

    return out, kv_data, new_len


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised MLA forward.  The function updates the KVCache in‑place
    and returns the attention output together with the (updated) KV‑cache tensor.
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # 0️⃣  Extract configuration & weights
    # ------------------------------------------------------------------
    bs = config.batch_size
    sl = config.seq_len               # normally 1 for decode
    nh = config.n_heads
    d  = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim
    msl = config.max_seq_len

    wDQ  = config.Q_proj_down_weight          # (dq, d)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, d)
    wUQ  = config.Q_proj_up_weight            # ((dnope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight           # ((dnope + dv) * nh, dkv)
    wO   = config.wo_weight                   # (d, nh * dv)

    # ------------------------------------------------------------------
    # 1️⃣  Prepare RoPE tables (cached globally)
    # ------------------------------------------------------------------
    cos_tbl, sin_tbl = _get_rope_tbl(drope, msl, x.device)

    # ------------------------------------------------------------------
    # 2️⃣  Dispatch: fast path when NoPE dimension is zero
    # ------------------------------------------------------------------
    cur_len = kv_cache.seq_len
    if dnope == 0:
        # combine the two down‑projections once and cache the result
        if not hasattr(config, "_combined_down_weight"):
            # (dq + dkv + drope, d)
            config._combined_down_weight = torch.cat([wDQ, wDKV], dim=0)
        w_comb = config._combined_down_weight
        out, new_kv, new_len = _forward_fast_combined(
            x, kv_cache.data, cur_len,
            w_comb, wUQ, wUKV, wO,
            cos_tbl, sin_tbl,
            nh, dnope, drope,
            dkv, dv,
        )
    else:
        out, new_kv, new_len = _forward_general(
            x, kv_cache.data, cur_len,
            wDQ, wDKV, wUQ, wUKV, wO,
            cos_tbl, sin_tbl,
            nh, dnope, drope,
            dkv, dv,
        )

    # ------------------------------------------------------------------
    # 3️⃣  Update KV cache state (in‑place) and return
    # ------------------------------------------------------------------
    kv_cache.data = new_kv
    kv_cache.seq_len = new_len
    return out, kv_cache.data