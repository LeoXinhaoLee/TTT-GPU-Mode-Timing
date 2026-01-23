# ------------------------------------------------------------
#  IMPORTS (DO NOT MODIFY)
# ------------------------------------------------------------
import os
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from reference import KVCache, Config  # must be imported exactly like this

# ------------------------------------------------------------
#  GLOBAL CACHE FOR RoPE tables (shared across calls)
# ------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )                     # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (T,1)
        idx = pos * theta[None, :]               # (T, half)
        idx = torch.cat([idx, idx], dim=-1)      # (T, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ------------------------------------------------------------
#  TRITON SOFTMAX (copied from the reference, kept unchanged)
# ------------------------------------------------------------
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
    """Row‑wise softmax for a 2‑D bfloat16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
    # choose a power‑of‑2 block size (capped at 1024)
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

# ------------------------------------------------------------
#  MAIN KERNEL
# ------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑Head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # (batch, seq_len, dim)   BF16
    kv_cache : torch.Tensor # updated cache tensor   BF16
    """
    config, x, kv_cache = data                # unpack

    # --------------------------------------------------------
    # 1️⃣  Shape / hyper‑parameter shortcuts
    # --------------------------------------------------------
    bs, sl, dim = x.shape                     # batch, seq_len (usually 1), model_dim
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # --------------------------------------------------------
    # 2️⃣  Extract weight tensors (already BF16, on the correct device)
    # --------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight           # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight          # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                  # (dim, nh*dv)

    # --------------------------------------------------------
    # 3️⃣  Down‑projection (Q and KV)
    # --------------------------------------------------------
    q_lora  = F.linear(x, wDQ)                # (bs, sl, dq)
    kv_lora_in = F.linear(x, wDKV)            # (bs, sl, dkv + d_rope)

    # --------------------------------------------------------
    # 4️⃣  KV‑cache update
    # --------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_in)    # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                     # absolute position of the newest token

    # --------------------------------------------------------
    # 5️⃣  Up‑project Q
    # --------------------------------------------------------
    # (bs, sl, (d_nope+d_rope)*nh) -> (bs, sl, nh, d_nope+d_rope)
    q_up = F.linear(q_lora, wUQ).view(bs, sl, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]               # (bs, sl, nh, d_nope)
    q_rope = q_up[..., d_nope:]               # (bs, sl, nh, d_rope)

    # --------------------------------------------------------
    # 6️⃣  Split the KV cache into latent part and RoPE part
    # --------------------------------------------------------
    kv_nope = kv_lora[..., :dkv]               # (bs, kv_len, dkv)   – latent representation
    k_rope  = kv_lora[..., dkv:]               # (bs, kv_len, d_rope)

    # --------------------------------------------------------
    # 7️⃣  RoPE tables (cos / sin) – cached globally
    # --------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ---- 7a. RoPE for queries (single position) -----------------
    #   q_rope (bs, sl, nh, d_rope)  ->  (bs, sl, nh, d_rope) after rotation
    cos_q = cos_table[query_pos]               # (d_rope,)
    sin_q = sin_table[query_pos]               # (d_rope,)
    # broadcast to (1,1,1,d_rope) so that broadcasting works over all dimensions
    cos_q = cos_q.view(1, 1, 1, d_rope)
    sin_q = sin_q.view(1, 1, 1, d_rope)
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # still (bs, sl, nh, d_rope)

    # ---- 7b. RoPE for keys (all cached positions) ---------------
    #   k_rope (bs, kv_len, d_rope) -> (bs, kv_len, d_rope)
    cos_k = cos_table[:kv_len]                # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                # (kv_len, d_rope)
    cos_k = cos_k.unsqueeze(0)                # (1, kv_len, d_rope)
    sin_k = sin_k.unsqueeze(0)                # (1, kv_len, d_rope)
    k_rope = k_rope * cos_k + _rotate_half(k_rope) * sin_k

    # --------------------------------------------------------
    # 8️⃣  Split the big KV‑up weight into K‑part and V‑part
    # --------------------------------------------------------
    # wUKV shape: ((d_nope+dv)*nh, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)   # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                # (nh, d_nope, dkv)
    wV = wUKV_view[:, d_nope:, :]                # (nh, dv, dkv)

    # --------------------------------------------------------
    # 9️⃣  Project the “no‑PE” query part into the latent space (dkv)
    # --------------------------------------------------------
    if d_nope == 0:
        # empty – just a zero tensor of the right shape
        q_nope_latent = torch.zeros(bs, sl, nh, dkv,
                                    dtype=torch.bfloat16, device=x.device)
    else:
        # Einstein summation: (b s h d_nope) • (h d_nope dkv) → (b s h dkv)
        q_nope_latent = torch.einsum('b s h d, h d k -> b s h k', q_nope, wK)

    # --------------------------------------------------------
    # 10️⃣  Scores (latent + RoPE)
    # --------------------------------------------------------
    #   kv_nope_T : (bs, dkv, kv_len)
    kv_nope_T = kv_nope.transpose(1, 2)        # (bs, dkv, kv_len)

    # latent part: (b s h k) • (b k t) → (b s h t)
    scores_nope = torch.einsum('b s h k, b k t -> b s h t', q_nope_latent, kv_nope_T)

    # rope part: (b s h d) • (b t d) → (b s h t)
    scores_rope = torch.einsum('b s h d, b t d -> b s h t', q_rope, k_rope)

    # total scores, scaled
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale   # (bs, sl, nh, kv_len)

    # --------------------------------------------------------
    # 11️⃣  Soft‑max (Triton) over the cache dimension
    # --------------------------------------------------------
    # flatten the first three axes to feed the 2‑D soft‑max kernel
    scores_flat = scores.reshape(bs * sl * nh, kv_len)           # (B·S·H, kv_len)
    attn_flat = _triton_softmax(scores_flat)                    # same shape
    attn = attn_flat.view(bs, sl, nh, kv_len)                # (bs, sl, nh, kv_len)

    # --------------------------------------------------------
    # 12️⃣  Weighted sum of latent keys (M = attn @ kv_nope)
    # --------------------------------------------------------
    # M : (b s h k) = Σ_t attn_{b s h t} * kv_nope_{b t k}
    M = torch.einsum('b s h t, b t k -> b s h k', attn, kv_nope)   # (bs, sl, nh, dkv)

    # --------------------------------------------------------
    # 13️⃣  Project the aggregated latent keys to values (v‑head)
    # --------------------------------------------------------
    # wV : (nh, dv, dkv)   →   we need (dh, k, v) order for einsum
    wV_T = wV.permute(0, 2, 1)                 # (nh, dkv, dv)
    y_head = torch.einsum('b s h k, h k v -> b s h v', M, wV_T)   # (bs, sl, nh, dv)

    # --------------------------------------------------------
    # 14️⃣  Merge heads and final linear projection
    # --------------------------------------------------------
    y = y_head.reshape(bs, sl, nh * dv)        # (bs, sl, nh*dv)
    output = F.linear(y, wO)                   # (bs, sl, dim)  BF16

    # --------------------------------------------------------
    # 15️⃣  Return
    # --------------------------------------------------------
    return output, kv_cache.data