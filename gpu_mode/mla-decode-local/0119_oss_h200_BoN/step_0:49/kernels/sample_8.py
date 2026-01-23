# --------------------------------------------------------------
#  Triton‑accelerated MLA forward
# --------------------------------------------------------------
### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from reference import KVCache, Config          # must be imported exactly like this
### END OF IMPORT STATEMENTS BLOCK ###

# --------------------------------------------------------------
#  1️⃣  RoPE utilities (cached cosine / sine tables)
# --------------------------------------------------------------
_rope_table_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_table_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                     # (max_seq_len,1)
        idx = pos * theta[None, :]                         # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)               # (max_seq_len, dim)
        _rope_table_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_table_cache[key]

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

# --------------------------------------------------------------
#  2️⃣  Triton soft‑max (row‑wise, bf16 → fp32 accumulation)
# --------------------------------------------------------------
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
    col = tl.arange(0, BLOCK_SIZE)
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        v = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(v, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        v = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        e = tl.exp(tl.cast(v, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(e, tl.bfloat16), mask=mask)
        sum_val += e
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        v = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(v, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape

    # pick a reasonable block size (power‑of‑2, capped at 1024)
    if n_cols <= 32:
        BLOCK = 32
    elif n_cols <= 64:
        BLOCK = 64
    elif n_cols <= 128:
        BLOCK = 128
    else:
        BLOCK = 1 << (n_cols - 1).bit_length()
        BLOCK = min(BLOCK, 1024)

    out = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_kernel[grid](
        out,
        x,
        out.stride(0),
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# --------------------------------------------------------------
#  3️⃣  Main kernel
# --------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # (batch, seq_len=1, dim)  bf16
    kv_cache.data : torch.Tensor   # updated cache (batch, max_seq_len, kv_dim)
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    #  Unpack config (all values are Python ints / scalars)
    # ------------------------------------------------------------------
    bs   = config.batch_size
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    #  Weight tensors (already on device, bf16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight      # (dq, dim)
    wDKV  = config.KV_proj_down_weight     # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight        # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight       # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight               # (dim, nh*dv)

    # ------------------------------------------------------------------
    #  1️⃣ Down‑projection
    # ------------------------------------------------------------------
    # x : (bs, 1, dim)
    q_lora = F.linear(x, wDQ)                     # (bs, 1, dq)
    kv_lora_input = F.linear(x, wDKV)              # (bs, 1, dkv+d_rope)

    # ------------------------------------------------------------------
    #  2️⃣ KV‑cache update (in‑place, avoid the .to() inside KVCache)
    # ------------------------------------------------------------------
    # kv_lora_input always has seq_len == 1 here
    start = kv_cache.seq_len
    new_len = start + 1
    kv_cache.data[:, start:new_len, :] = kv_lora_input.to(kv_cache.data.dtype)
    kv_cache.seq_len = new_len

    # Slice the cache to the valid length
    kv_lora = kv_cache.data[:, :new_len, :]          # (bs, kv_len, dkv+d_rope)
    kv_len  = new_len
    query_pos = kv_len - 1                            # absolute position of current token

    # ------------------------------------------------------------------
    #  3️⃣ Up‑project queries
    # ------------------------------------------------------------------
    # squeeze seq‑dim (always 1)
    q_up = F.linear(q_lora.squeeze(1), wUQ)          # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)        # (bs, nh, d_nope+d_rope)
    q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

    # ------------------------------------------------------------------
    #  4️⃣ Split KV into latent + RoPE parts
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]                # (bs, kv_len, dkv)
    k_rope_input   = kv_lora[..., dkv:]               # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    #  5️⃣ RoPE tables (cached)
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ------------------------------------------------------------------
    #  6️⃣ Apply RoPE to queries (single position)
    # ------------------------------------------------------------------
    cos_q = cos_table[query_pos]          # (d_rope,)
    sin_q = sin_table[query_pos]          # (d_rope,)
    # broadcast to (bs, nh, d_rope)
    cos_q_b = cos_q.view(1, 1, -1)
    sin_q_b = sin_q.view(1, 1, -1)
    q_rope = q_rope * cos_q_b + _rotate_half(q_rope) * sin_q_b

    # ------------------------------------------------------------------
    #  7️⃣ Apply RoPE to keys (all cached positions)
    # ------------------------------------------------------------------
    # (kv_len, d_rope) → broadcast to (bs, kv_len, d_rope)
    cos_k = cos_table[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
    sin_k = sin_table[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    #  8️⃣ Split KV‑up projection into weight tensors
    # ------------------------------------------------------------------
    # wUKV : ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)            # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                         # (nh, d_nope, dkv)
    wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)     # (nh, dkv, dv)

    # ------------------------------------------------------------------
    #  9️⃣ Latent part of attention scores
    # ------------------------------------------------------------------
    # q_nope      : (bs, nh, d_nope)
    # wK          : (nh, d_nope, dkv)
    # → q_nope_latent : (bs, nh, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # scores_nope = q_nope_latent @ kv_nope_input^T   → (bs, nh, kv_len)
    scores_nope = torch.einsum('bhk,btk->bht', q_nope_latent, kv_nope_input)

    # ------------------------------------------------------------------
    #  🔟 RoPE part of scores
    # ------------------------------------------------------------------
    # q_rope : (bs, nh, d_rope)      k_rope : (bs, kv_len, d_rope)
    scores_rope = torch.einsum('bhd,btd->bht', q_rope, k_rope)

    # ------------------------------------------------------------------
    #  🧮 Combine + scale
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale            # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    #  🟢 Softmax (rows = bs*nh, columns = kv_len)
    # ------------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)              # (B*H, L)
    attn_flat = _triton_softmax(scores_flat)                # (B*H, L)  bf16
    attn = attn_flat.view(bs, nh, kv_len)                   # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    #  🧩 Weighted sum of latent keys  M = attn @ kv_nope_input
    # ------------------------------------------------------------------
    M = torch.einsum('bht,btk->bhk', attn, kv_nope_input)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    #  📤 Project aggregated latent keys → per‑head values
    # ------------------------------------------------------------------
    # y_head = M @ wV_T   (bs, nh, dkv) @ (nh, dkv, dv) → (bs, nh, dv)
    y_head = torch.einsum('bhk,hkd->bhd', M, wV_T)           # (bs, nh, dv)

    # ------------------------------------------------------------------
    #  📏 Final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                         # (bs, nh*dv)
    y = F.linear(y, wO)                                     # (bs, dim)
    y = y.unsqueeze(1)                                      # (bs, 1, dim) – matches original API

    # ------------------------------------------------------------------
    #  Return output and the (now updated) KV‑cache tensor
    # ------------------------------------------------------------------
    return y, kv_cache.data