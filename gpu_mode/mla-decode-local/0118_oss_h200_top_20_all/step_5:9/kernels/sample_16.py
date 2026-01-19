### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import math
import os
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from reference import KVCache, Config          # <- must be imported exactly like this
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
#  Cached RoPE tables (cos / sin) – BF16
# ----------------------------------------------------------------------
_rope_cache: dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        #   theta_i = 10000 ** (-i/half)   (float32 → bf16)
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device)[:, None]          # (max_seq_len,1)
        idx = pos * theta[None, :]                     # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)           # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Helper: rotate‑half (used by RoPE)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
#  Triton softmax (row‑wise, BF16 → FP32 reduction, then back to BF16)
#  This version is *much* faster than torch.softmax for very long rows.
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,                     # pointers
    stride_out, stride_in,               # strides (row stride)
    ROWS: tl.constexpr,                 # number of rows (batch * heads)
    COLS: tl.constexpr,                 # length of the KV cache (dynamic)
    BLOCK: tl.constexpr,                # block size for the reduction
):
    row = tl.program_id(0)                  # which (batch,head) we are processing
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK)                # (BLOCK,)  – constexpr → power‑of‑2
    for start in range(0, COLS, BLOCK):
        cur = start + col
        mask = cur < COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK], 0.0, tl.float32)
    for start in range(0, COLS, BLOCK):
        cur = start + col
        mask = cur < COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, COLS, BLOCK):
        cur = start + col
        mask = cur < COLS
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Row‑wise softmax using the Triton kernel above."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    rows, cols = x.shape
    assert cols == kv_len
    # pick a reasonable block size (next power‑of‑2)
    BLOCK = 128
    out = torch.empty_like(x)
    grid = (rows,)
    _softmax_kernel[grid](
        out,
        x,
        out.stride(0),
        x.stride(0),
        ROWS=rows,
        COLS=cols,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
#  Main kernel – everything is pure Torch except the tiny softmax above.
#  The heavy part is wrapped with torch.compile (fullgraph+dynamic) so
#  that the whole forward pass fuses the many tiny operators.
# ----------------------------------------------------------------------
_compiled_fwd = None          # will hold the torch‑compiled function after first call

def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns (output, updated_kv_cache) where *output* has shape
    [batch_size, seq_len, dim] and *updated_kv_cache* is the
    KVCache.data Tensor (shape [batch, max_seq_len, d_kv+d_rope]).
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    #  Unpack configuration
    # ------------------------------------------------------------------
    bs    = config.batch_size
    sl    = config.seq_len                     # =1 for generation
    d     = config.dim
    nh    = config.n_heads
    dq    = config.q_lora_rank
    dkv   = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv    = config.v_head_dim
    msl   = config.max_seq_len

    # ------------------------------------------------------------------
    #  Weight tensors (already on the correct device & dtype)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, d)
    wDKV  = config.KV_proj_down_weight         # (dkv+d_rope, d)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (d, nh*dv)

    # ------------------------------------------------------------------
    #  1️⃣  KV‑cache update (down‑projection only – kept outside the compiled
    #        graph to keep the in‑place write simple)
    # ------------------------------------------------------------------
    #   kv_lora : (bs, sl, dkv+d_rope)
    kv_lora = F.linear(x, wDKV)                       # (bs, 1, dkv+d_rope)

    cur_len = kv_cache.seq_len
    kv_cache.data[:, cur_len:cur_len + sl, :] = kv_lora
    kv_cache.seq_len = cur_len + sl                   # new length after the token is added
    kv_len = kv_cache.seq_len                         # scalar used later

    # ------------------------------------------------------------------
    #  2️⃣  RoPE tables (cached globally)
    # ------------------------------------------------------------------
    cos_tbl, sin_tbl = _get_rope_tables(d_rope, msl, x.device)

    # ------------------------------------------------------------------
    #  3️⃣  Heavy part – compiled / fused
    # ------------------------------------------------------------------
    global _compiled_fwd
    if _compiled_fwd is None:
        # ------------------------------------------------------------------
        #  Compile a function that does everything **after** the cache has been
        #  updated.  All heavy tensor ops (linear, matmul, softmax, etc.) are
        #  inside this graph so that torch‑compile can fuse them.
        # ------------------------------------------------------------------
        def _fwd(x,                # (bs, 1, d)
                 kv_cache_data,    # (bs, max_seq_len, dkv+d_rope)  – already contains the new token
                 kv_len,           # int  – current length of the cache
                 wDQ, wUQ, wUKV, wO,
                 cos_tbl, sin_tbl,
                 n_heads, d_nope, d_rope, dkv, dv):
            """
            Inside the compiled region we only use tensor operations that
            support JIT‑fusion.  The function returns a tensor of shape
            (bs, 1, d) and does NOT touch the KV‑cache any more.
            """
            bs = x.shape[0]                     # batch size (== config.batch_size)

            # ------------------------------------------------------------------
            #   Q : down‑proj → up‑proj → split
            # ------------------------------------------------------------------
            q_lora = F.linear(x, wDQ)                               # (bs,1,dq)
            q_up   = F.linear(q_lora, wUQ)                           # (bs,1,(d_nope+d_rope)*nh)
            # reshape to (bs, nh, d_nope+d_rope)
            q_up   = q_up.view(bs, 1, n_heads, d_nope + d_rope).squeeze(1)   # (bs, nh, Dq)
            q_nope, q_rope_raw = torch.split(q_up,
                                              [d_nope, d_rope],
                                              dim=-1)                     # (bs,nh,d_nope) , (bs,nh,d_rope)

            # ------------------------------------------------------------------
            #   KV cache : latent part + rope part
            # ------------------------------------------------------------------
            kv_latent   = kv_cache_data[:, :kv_len, :dkv]             # (bs, kv_len, dkv)
            k_rope_raw  = kv_cache_data[:, :kv_len, dkv:]            # (bs, kv_len, d_rope)

            # ------------------------------------------------------------------
            #   RoPE – query (single position) and keys (entire cache)
            # ------------------------------------------------------------------
            # query position = kv_len-1   (absolute position of the just‑generated token)
            q_pos = kv_len - 1
            cos_q = cos_tbl[q_pos]                                 # (d_rope,)
            sin_q = sin_tbl[q_pos]                                 # (d_rope,)

            q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, d_rope)

            # keys – whole cached interval
            cos_k = cos_tbl[:kv_len]                                # (kv_len, d_rope)
            sin_k = sin_tbl[:kv_len]                                # (kv_len, d_rope)
            # broadcast to (bs, 1, kv_len, d_rope) then apply rotation
            cos_kb = cos_k[None, :, :]                              # (1, kv_len, d_rope)
            sin_kb = sin_k[None, :, :]                              # (1, kv_len, d_rope)
            k_rope = k_rope_raw * cos_kb + _rotate_half(k_rope_raw) * sin_kb   # (bs, kv_len, d_rope)

            # ------------------------------------------------------------------
            #   Project the query “no‑PE” part into the latent space (dkv)
            # ------------------------------------------------------------------
            # wUKV shape: ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
            wUKV_view = wUKV.view(n_heads, d_nope + dv, dkv)          # (nh, d_nope+dv, dkv)
            wK = wUKV_view[:, :d_nope, :]                            # (nh, d_nope, dkv)
            #   (bs, nh, d_nope)  ×  (nh, d_nope, dkv)  →  (bs, nh, dkv)
            q_nope_lat = torch.einsum('bhd,hdm->bhm', q_nope, wK)   # (bs, nh, dkv)

            # ------------------------------------------------------------------
            #   Attention scores (two parts)
            # ------------------------------------------------------------------
            #   (bs, nh, dkv)  ×  (bs, kv_len, dkv)^T   → (bs, nh, kv_len)
            scores_nope = torch.matmul(q_nope_lat, kv_latent.transpose(-1, -2))
            #   (bs, nh, d_rope)  ×  (bs, kv_len, d_rope)^T   → (bs, nh, kv_len)
            scores_rope = torch.matmul(q_rope, k_rope.transpose(-1, -2))

            scale = 1.0 / math.sqrt(d_nope + d_rope)
            scores = (scores_nope + scores_rope) * scale           # (bs, nh, kv_len)

            # ------------------------------------------------------------------
            #   Softmax – we replace torch.nn.functional.softmax with the
            #   tiny Triton kernel defined above (much faster for long rows)
            # ------------------------------------------------------------------
            attn = _triton_softmax(scores.reshape(-1, kv_len), kv_len)   # (bs*nh, kv_len)
            attn = attn.view(bs, n_heads, kv_len)                       # (bs, nh, kv_len)

            # ------------------------------------------------------------------
            #   Weighted sum over the *latent* KV vectors
            # ------------------------------------------------------------------
            #   (bs, nh, kv_len)  ×  (bs, kv_len, dkv)   → (bs, nh, dkv)
            M = torch.matmul(attn, kv_latent)                      # (bs, nh, dkv)

            # ------------------------------------------------------------------
            #   Project latent M → values (dv)   –  wV is the second half of wUKV
            # ------------------------------------------------------------------
            wV = wUKV_view[:, d_nope:, :]                         # (nh, dv, dkv)
            # we need (dkv, dv) per head, so transpose
            wV_T = wV.permute(0, 2, 1)                             # (nh, dkv, dv)
            #   (bs, nh, dkv)  ×  (nh, dkv, dv)   →  (bs, nh, dv)
            y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)       # (bs, nh, dv)

            # ------------------------------------------------------------------
            #   Output projection (combine heads)
            # ------------------------------------------------------------------
            y = y_head.reshape(bs, -1)                            # (bs, nh*dv)
            out = F.linear(y, wO)                                 # (bs, d)
            out = out.unsqueeze(1)                                # (bs, 1, d)
            return out

        # compile once – the graph is static except for `kv_len` (dynamic)
        _compiled_fwd = torch.compile(
            _fwd,
            fullgraph=True,
            dynamic=True,          # kv_len changes at runtime
        )

    # ------------------------------------------------------------------
    #  4️⃣  Run compiled forward
    # ------------------------------------------------------------------
    out = _compiled_fwd(
        x,
        kv_cache.data,          # the whole cache (already contains the new token)
        kv_len,
        wDQ, wUQ, wUKV, wO,
        cos_tbl, sin_tbl,
        nh, d_nope, d_rope, dkv, dv,
    )                           # (bs, 1, d)

    # ------------------------------------------------------------------
    #  5️⃣  Return output and the *updated* cache tensor
    # ------------------------------------------------------------------
    return out, kv_cache.data