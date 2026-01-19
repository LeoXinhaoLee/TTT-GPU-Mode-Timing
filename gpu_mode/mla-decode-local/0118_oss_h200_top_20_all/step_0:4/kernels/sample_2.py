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
# Helper functions (RoPE tables, rotate‑half, Triton softmax)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Cache (cos, sin) tables of shape (max_seq_len, dim) in bf16."""
    if not hasattr(_get_rope_tables, "_cache"):
        _get_rope_tables._cache = {}
    key = (dim, max_seq_len, device)
    if key not in _get_rope_tables._cache:
        half = dim // 2
        theta = (
            10000.0
            ** (-torch.arange(half, dtype=torch.float32, device=device) / half)
        ).to(torch.bfloat16)                     # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
        idx = pos * theta[None, :]               # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)      # (max_seq_len, dim)
        _get_rope_tables._cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _get_rope_tables._cache[key]

# ----------------------------------------------------------------------
# Triton row‑wise softmax (bf16) – copied from the reference implementation
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape

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
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# Custom kernel – highly optimised MLA forward
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor                # shape (batch, seq_len, dim), bf16
    kv_cache.data : torch.Tensor        # updated KV‑cache (raw down‑projected values)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Unpack configuration
    # --------------------------------------------------------------
    bs   = config.batch_size          # e.g. 128
    sl   = config.seq_len             # always 1 (time dimension)
    d    = config.dim                 # model dimension, e.g. 7168
    nh   = config.n_heads             # number of heads, e.g. 128
    dq   = config.q_lora_rank         # low‑rank dim for Q, e.g. 1536
    dkv  = config.kv_lora_rank        # low‑rank dim for KV, e.g. 512
    d_nope = config.qk_nope_head_dim  # dimension without RoPE, e.g. 64
    d_rope = config.qk_rope_head_dim  # RoPE dimension, e.g. 64
    dv   = config.v_head_dim          # value dimension per head, e.g. 128
    msl  = config.max_seq_len         # e.g. 8192

    # --------------------------------------------------------------
    # Weight tensors (already on device, bf16)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, d)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, d)
    wUQ   = config.Q_proj_up_weight            # ((d_nope + d_rope) * nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
    wO    = config.wo_weight                   # (d, nh * dv)

    # --------------------------------------------------------------
    # 1️⃣ Down‑projection
    # --------------------------------------------------------------
    x2d = x.squeeze(1)                     # (bs, d)

    q_lora = F.linear(x2d, wDQ)            # (bs, dq)
    kv_lora_raw = F.linear(x2d, wDKV)      # (bs, dkv + d_rope)

    # --------------------------------------------------------------
    # 2️⃣ KV‑cache update (raw low‑rank values)
    # --------------------------------------------------------------
    kv_lora_raw = kv_lora_raw.unsqueeze(1)           # (bs, 1, dkv + d_rope)
    kv_lora, kv_len = kv_cache(kv_lora_raw)          # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                            # absolute position of current token

    # --------------------------------------------------------------
    # 3️⃣ Split Q and KV streams
    # --------------------------------------------------------------
    # Q up‑projection and split
    q_up = F.linear(q_lora, wUQ)                     # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)       # (bs, nh, d_total)
    q_nope = q_up[..., :d_nope]                     # (bs, nh, d_nope)
    q_rope_raw = q_up[..., d_nope:]                 # (bs, nh, d_rope)

    # KV split (raw)
    kv_nope_raw = kv_lora[..., :dkv]                # (bs, kv_len, dkv)
    kv_rope_raw = kv_lora[..., dkv:]                # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 4️⃣ RoPE tables (cached)
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ----- Query RoPE (single position) -----
    cos_q = cos_table[query_pos]            # (d_rope,)
    sin_q = sin_table[query_pos]            # (d_rope,)
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, d_rope)

    # ----- Key RoPE (all cached positions) -----
    cos_k = cos_table[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
    sin_k = sin_table[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
    k_rope = kv_rope_raw * cos_k + _rotate_half(kv_rope_raw) * sin_k   # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 5️⃣ Project Q‑no‑PE into latent space (per‑head weight)
    # --------------------------------------------------------------
    # wUKV shape: ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)          # (nh, total_qk, dkv)
    wK = wUKV_view[:, :d_nope, :]                         # (nh, d_nope, dkv)

    # (bs, nh, d_nope) × (nh, d_nope, dkv) → (bs, nh, dkv)
    q_proj_dkv = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # --------------------------------------------------------------
    # 6️⃣ Compute attention scores (latent + RoPE)
    # --------------------------------------------------------------
    # latent part
    scores_nope = torch.bmm(q_proj_dkv, kv_nope_raw.transpose(1, 2))   # (bs, nh, kv_len)

    # RoPE part
    scores_rope = torch.bmm(q_rope, k_rope.transpose(1, 2))           # (bs, nh, kv_len)

    # combine & scale
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                     # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 7️⃣ Softmax (Triton) → attention weights
    # --------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)          # (bs*nh, kv_len)
    attn_flat = _triton_softmax(scores_flat)            # (bs*nh, kv_len)
    attn = attn_flat.view(bs, nh, kv_len)               # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 8️⃣ Weighted sum of the latent KV vectors (no‑PE part)
    # --------------------------------------------------------------
    agg = torch.bmm(attn, kv_nope_raw)                  # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 9️⃣ Project aggregated latent vectors to values (per‑head weight)
    # --------------------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                       # (nh, dv, dkv)
    v = torch.einsum('bhd,hvd->bhv', agg, wV)          # (bs, nh, dv)

    # --------------------------------------------------------------
    # 🔟 Merge heads & final linear projection
    # --------------------------------------------------------------
    y = v.reshape(bs, nh * dv)                         # (bs, nh*dv)
    y = y.unsqueeze(1)                                 # (bs, 1, nh*dv)
    output = F.linear(y, wO)                           # (bs, 1, dim)

    # --------------------------------------------------------------
    # Return output and the (updated) raw KV‑cache tensor
    # --------------------------------------------------------------
    return output, kv_cache.data