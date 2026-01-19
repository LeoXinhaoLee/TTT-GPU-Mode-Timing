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
# 1️⃣  Helper utilities (identical to the reference implementation)
# ----------------------------------------------------------------------
_rope_cache: dict = {}
_cached_cos = None          # (max_seq_len, rope_dim)  bfloat16
_cached_sin = None          # same

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Create (or fetch) cached sin/cos tables for RoPE."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)          # (max_seq_len, 1)
        idx = pos * theta[None, :]                               # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                      # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 2️⃣  Triton‐based soft‑max (fallback – used only in the generic path)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # -------------------------------------------------
    # 1️⃣  max‑reduction (row wise)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 2️⃣  exp & sum
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3️⃣  normalisation
    # -------------------------------------------------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur,
                 tl.cast(norm, tl.bfloat16),
                 mask=mask)


def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bfloat16 tensor using Triton."""
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
        N=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# 3️⃣  Fast‑path kernel (d_nope == 0) – everything stays in PyTorch
#     but we cache the two “expensive” mat‑muls (Q↓→Q↑ and KV↓)
# ----------------------------------------------------------------------
#   • Q↓ + Q↑ are fused into a single compiled routine (`_fast_q_proj`).
#   • KV↓ is fused into a single compiled routine (`_fast_kv_proj`).
#   • The final V‑projection + output projection are left as two calls
#     because they are already extremely cheap compared to the attention.
#   • All other ops (RoPE, softmax, attention) are unchanged but we
#     reuse the already cached cosine/sine tables, so no extra work
#     happens per token.
# ----------------------------------------------------------------------
_fast_q_proj = None          # will be set on first call
_fast_kv_proj = None         # will be set on first call

def _build_fast_q_proj(dq: int, nh: int, drope: int):
    """Compile the *down‑* + *up‑* projection for queries."""
    def _inner(x, wDQ, wUQ):
        # x : (bs, dim)                     (float16)
        # wDQ: (dq, dim)   – ↓
        # wUQ: (nh*drope, dq) – ↑
        q_lora = F.linear(x, wDQ)                     # (bs, dq)
        q_up   = F.linear(q_lora, wUQ)                # (bs, nh*drope)
        return q_up.view(x.shape[0], nh, drope)       # (bs, nh, drope)
    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )

def _build_fast_kv_proj(dkv: int, drope: int):
    """Compile the down‑projection for KV (no up‑proj needed)."""
    def _inner(x, wDKV):
        # x : (bs, dim)
        # wDKV : (dkv + drope, dim)
        kv = F.linear(x, wDKV)                        # (bs, dkv+drope)
        return kv
    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )

# ----------------------------------------------------------------------
# 4️⃣  Generic compiled forward (fallback – d_nope > 0)
# ----------------------------------------------------------------------
_compiled_forward = None   # lazily built later

def _build_compiled_forward():
    """Compile the full MLA forward for the general case (d_nope > 0)."""
    def _inner(x: torch.Tensor,
               kv_data: torch.Tensor,
               cur_len: int,
               cos_tbl: torch.Tensor,
               sin_tbl: torch.Tensor,
               wDQ: torch.Tensor,
               wDKV: torch.Tensor,
               wUQ: torch.Tensor,
               wUKV: torch.Tensor,
               wO: torch.Tensor,
               nh: int,
               d_nope: int,
               d_rope: int,
               dkv: int,
               dv: int):
        # -------------------------------------------------
        # 1️⃣  Down‑projection
        # -------------------------------------------------
        q_lora = F.linear(x, wDQ)               # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)            # (bs, 1, dkv+d_rope)

        # -------------------------------------------------
        # 2️⃣  KV‑cache write
        # -------------------------------------------------
        new_len = cur_len + kv_lora0.shape[1]    # always adds 1 token
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]       # (bs, kv_len, dkv+d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        # -------------------------------------------------
        # 3️⃣  Up‑project queries
        # -------------------------------------------------
        q_up = F.linear(q_lora.squeeze(1), wUQ)                 # (bs, nh * (d_nope+d_rope))
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)      # (bs, nh, d_nope+d_rope)

        # -------------------------------------------------
        # 4️⃣  Split Q into No‑PE / RoPE
        # -------------------------------------------------
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        # -------------------------------------------------
        # 5️⃣  KV split / latent projection
        # -------------------------------------------------
        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)  # kv_nope unused
        kv_latent = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None    # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        # -------------------------------------------------
        # 6️⃣  Project “no‑pe” part of query into latent space
        # -------------------------------------------------
        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk',
                                         q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                        dtype=torch.bfloat16,
                                        device=x.device)

        # -------------------------------------------------
        # 7️⃣  RoPE on queries & keys
        # -------------------------------------------------
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)

        # -------------------------------------------------
        # 8️⃣  Scores & soft‑max
        # -------------------------------------------------
        scores_rope = torch.matmul(q_rope_rot,
                                   k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

        scores_flat = scores.reshape(x.shape[0] * nh, kv_len)
        attn_flat = _triton_softmax(scores_flat)
        attn = attn_flat.view(x.shape[0], nh, kv_len)          # (bs, nh, kv_len)

        # -------------------------------------------------
        # 9️⃣  Weighted sum over latent vectors
        # -------------------------------------------------
        latent_agg = torch.matmul(attn, kv_latent)               # (bs, nh, dkv)

        # -------------------------------------------------
        # 🔟  Project to value space (dv)
        # -------------------------------------------------
        y_head = torch.einsum('bhd, hdf -> bhf',
                              latent_agg, wV_T)                  # (bs, nh, dv)

        # -------------------------------------------------
        # 1️⃣1️⃣  Output projection
        # -------------------------------------------------
        y_head_flat = y_head.reshape(x.shape[0], nh * dv)       # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                         # (bs, dim)
        out = out.unsqueeze(1)                                   # (bs, 1, dim)

        return out, kv_data, new_len
    # Compile once
    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )

# ----------------------------------------------------------------------
# 5️⃣  Main entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised MLA forward.
    • d_nope == 0 → fast‑path that re‑uses compiled Q/KV projections.
    • d_nope  > 0 → generic compiled fallback (identical to reference).
    """
    config, x, kv_cache = data

    # -------------------------------------------------
    # 0️⃣  Resolve shape / config shortcuts
    # -------------------------------------------------
    bs = config.batch_size
    sl = config.seq_len                     # always 1 in the generation loop
    nh = config.n_heads
    d = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim          # noqa: N806
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim
    msl = config.max_seq_len

    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+d_rope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # -------------------------------------------------
    # 1️⃣  Ensure RoPE tables are cached (global)
    # -------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < msl:
        _cached_cos, _cached_sin = _get_rope_tables(d_rope, msl, x.device)

    # -------------------------------------------------
    # 2️⃣  Fast‑path when there is **no** “No‑PE” part (the most common config)
    # -------------------------------------------------
    if d_nope == 0:
        # -------------------------------------------------
        # 2.1   Compile‑once the tiny linear pipelines (if not already done)
        # -------------------------------------------------
        global _fast_q_proj, _fast_kv_proj
        if _fast_q_proj is None:
            _fast_q_proj = _build_fast_q_proj(dq, nh, d_rope)
        if _fast_kv_proj is None:
            _fast_kv_proj = _build_fast_kv_proj(dkv, d_rope)

        # -------------------------------------------------
        # 2.2   Collapse the sequence dimension (seq_len == 1)
        # -------------------------------------------------
        x2d = x.squeeze(1)                       # (bs, dim)

        # -------------------------------------------------
        # 2.3   KV down‑projection and cache write (including RoPE rotation)
        # -------------------------------------------------
        kv_lora = _fast_kv_proj(x2d, wDKV)       # (bs, dkv + d_rope)

        # current write position in the cache
        pos = kv_cache.seq_len

        # write latent part (no rotation needed)
        kv_cache.data[:, pos, :dkv] = kv_lora[:, :dkv]

        # rotate RoPE part *once* before storing
        rope_raw = kv_lora[:, dkv:]                # (bs, d_rope)
        cos_pos = _cached_cos[pos]                 # (d_rope,)
        sin_pos = _cached_sin[pos]                 # (d_rope,)
        rope_rot = rope_raw * cos_pos + _rotate_half(rope_raw) * sin_pos
        kv_cache.data[:, pos, dkv:] = rope_rot

        # advance cache length
        kv_cache.seq_len += 1
        cur_len = kv_cache.seq_len                 # = new cache length
        query_pos = cur_len - 1

        # -------------------------------------------------
        # 2.4   Q down‑projection + up‑projection (low‑rank) + RoPE
        # -------------------------------------------------
        q_up = _fast_q_proj(x2d, wDQ, wUQ)         # (bs, nh, d_rope)
        cos_q = _cached_cos[query_pos]             # (d_rope,)
        sin_q = _cached_sin[query_pos]             # (d_rope,)
        q = q_up * cos_q + _rotate_half(q_up) * sin_q   # (bs, nh, d_rope)

        # -------------------------------------------------
        # 2.5   Gather keys / values from cache
        # -------------------------------------------------
        #   k : (bs, cur_len, d_rope)
        #   v : (bs, cur_len, dkv)
        k = kv_cache.data[:, :cur_len, dkv:]       # (bs, cur_len, d_rope)
        v = kv_cache.data[:, :cur_len, :dkv]       # (bs, cur_len, dkv)

        # -------------------------------------------------
        # 2.6   Compute attention (flash‑style)
        # -------------------------------------------------
        #   The built‑in flash‑attention kernel is already optimal for
        #   Q‑length‑1 queries, so we simply forward to it.
        #   Shapes expected by `scaled_dot_product_attention`:
        #       q : (bs, nh, 1, d_rope)
        #       k : (bs, nh, cur_len, d_rope)
        #       v : (bs, nh, cur_len, dkv)
        q_sdp = q.unsqueeze(2)                                 # (bs, nh, 1, d_rope)
        k_sdp = k.unsqueeze(1).expand(-1, nh, -1, -1)         # (bs, nh, cur_len, d_rope)
        v_sdp = v.unsqueeze(1).expand(-1, nh, -1, -1)         # (bs, nh, cur_len, dkv)

        latent_agg = F.scaled_dot_product_attention(
            q_sdp, k_sdp, v_sdp, is_causal=False)          # (bs, nh, 1, dkv)
        latent_agg = latent_agg.squeeze(2)                    # (bs, nh, dkv)

        # -------------------------------------------------
        # 2.7   Project aggregated latents → value space (dv)
        # -------------------------------------------------
        #   wV_T : (nh, dkv, dv)  – pre‑reshaped once for speed
        wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1)        # (nh, dkv, dv)
        y_head = torch.einsum('bhd, hdf -> bhf',
                              latent_agg, wV_T)               # (bs, nh, dv)

        # -------------------------------------------------
        # 2.8   Final output projection
        # -------------------------------------------------
        y_head_flat = y_head.reshape(bs, nh * dv)             # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                         # (bs, dim)
        out = out.unsqueeze(1)                                 # (bs, 1, dim)

        return out, kv_cache.data

    # -----------------------------------------------------------------
    # 3️⃣  General case – d_nope > 0 (fallback to compiled reference)
    # -----------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,                                 # (bs, 1, dim)
        kv_cache.data,                     # (bs, max_seq_len, dkv+d_rope)
        kv_cache.seq_len,                  # current cache length (int)
        _cached_cos,                       # (max_seq_len, d_rope)
        _cached_sin,
        wDQ, wDKV, wUQ, wUKV, wO,
        nh, d_nope, d_rope, dkv, dv
    )

    # -----------------------------------------------------------------
    # 4️⃣  Update KV cache state (in‑place, as the reference expects)
    # -----------------------------------------------------------------
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data