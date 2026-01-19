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
# Global caches & helpers (identical to the reference implementation)
# ----------------------------------------------------------------------
_rope_cache: dict = {}
_cached_cos = None          # (max_seq_len, rope_dim)  bfloat16
_cached_sin = None          # same

_q_fused_cache = {}         # cache for fused Q projection weight
_vproj_cache = {}           # cache for per‑head V‑projection weight (wV_T)

_compiled_fast = None       # torch‑compiled fast‑path (no‑pe case)
_compiled_forward = None    # torch‑compiled generic forward (with no‑pe)

# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _get_fused_q_weight(wUQ: torch.Tensor, wDQ: torch.Tensor,
                        nh: int, d_rope: int) -> torch.Tensor:
    """Return (or compute) the fused Q‑projection weight wUQ @ wDQ."""
    key = (id(wUQ), id(wDQ), nh, d_rope)
    if key not in _q_fused_cache:
        _q_fused_cache[key] = torch.matmul(wUQ, wDQ)      # (nh*d_rope, dim)
    return _q_fused_cache[key]


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Build (or fetch) cosine / sine tables for rotary embeddings."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)          # (max_seq_len,1)
        idx = pos * theta[None, :]                                 # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                        # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# Fast‑path (qk_nope_head_dim == 0) – compiled with torch.compile
# ----------------------------------------------------------------------
def _build_compiled_fast():
    """Compile the fast MLA forward (no‑PE) for maximum performance."""
    def _inner(x: torch.Tensor,
               kv_data: torch.Tensor,
               cur_len: int,
               cos_tbl: torch.Tensor,
               sin_tbl: torch.Tensor,
               wQ_fused: torch.Tensor,
               wDKV: torch.Tensor,
               wV_T: torch.Tensor,
               wO: torch.Tensor,
               nh: int,
               d_rope: int,
               dkv: int,
               dv: int,
               dim: int):
        # 1) flatten input
        bs = x.shape[0]
        x2d = x.squeeze(1)                       # (bs, dim)

        # 2) fused Q‑projection (rope only)
        q_flat = F.linear(x2d, wQ_fused)          # (bs, nh*d_rope)
        q = q_flat.view(bs, nh, d_rope)          # (bs, nh, d_rope)

        # 3) KV down‑projection
        kv_down = F.linear(x2d, wDKV)             # (bs, dkv+d_rope)
        latent_new = kv_down[:, :dkv]              # (bs, dkv)
        rope_raw   = kv_down[:, dkv:]              # (bs, d_rope)

        # 4) rotate rope part for the *new* key position (cur_len)
        cos_pos = cos_tbl[cur_len].view(1, d_rope)    # (1, d_rope)
        sin_pos = sin_tbl[cur_len].view(1, d_rope)
        rope_rot = rope_raw * cos_pos + _rotate_half(rope_raw) * sin_pos

        # 5) write new entry into KV cache (in‑place)
        new_len = cur_len + 1
        kv_new = torch.cat([latent_new, rope_rot], dim=-1)          # (bs, dkv+d_rope)
        kv_data[:, cur_len:new_len, :] = kv_new                    # update cache

        # 6) extract full KV tensors up to new_len
        kv_all   = kv_data[:, :new_len, :]                         # (bs, new_len, dkv+d_rope)
        kv_latent = kv_all[:, :, :dkv]                             # (bs, new_len, dkv)
        k_rope   = kv_all[:, :, dkv:]                              # (bs, new_len, d_rope)

        # 7) rotate query for its absolute position (new_len‑1)
        qpos = new_len - 1
        cos_q = cos_tbl[qpos].view(1, 1, d_rope)                  # (1,1,d_rope)
        sin_q = sin_tbl[qpos].view(1, 1, d_rope)
        q_rot = q * cos_q + _rotate_half(q) * sin_q                # (bs, nh, d_rope)

        # 8) broadcast rope keys and values across heads
        k_rot = k_rope[:, None, :, :].expand(-1, nh, -1, -1)       # (bs, nh, new_len, d_rope)
        v     = kv_latent[:, None, :, :].expand(-1, nh, -1, -1)    # (bs, nh, new_len, dkv)

        # 9) flash‑attention on the rope part, values are the latent vectors
        scale = 1.0 / math.sqrt(d_rope)
        attn_out = F.scaled_dot_product_attention(
            q_rot.unsqueeze(2),          # (bs, nh, 1, d_rope)
            k_rot,                       # (bs, nh, new_len, d_rope)
            v,                           # (bs, nh, new_len, dkv)
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale
        )                               # (bs, nh, 1, dkv)
        latent_agg = attn_out.squeeze(2)                         # (bs, nh, dkv)

        # 10) per‑head V‑projection (batched mat‑mul)
        # (bs, nh, dkv) -> (nh, bs, dkv)
        latent_perm = latent_agg.permute(1, 0, 2).contiguous()
        # wV_T: (nh, dkv, dv)
        y_head = torch.bmm(latent_perm, wV_T)                     # (nh, bs, dv)
        y_head = y_head.permute(1, 0, 2).contiguous()            # (bs, nh, dv)

        # 11) final output projection
        y_head_flat = y_head.view(bs, nh * dv)                    # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                           # (bs, dim)
        out = out.unsqueeze(1)                                    # (bs, 1, dim)

        return out, kv_data, new_len
    # Compile once – the backend will autotune for the H200.
    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )

# ----------------------------------------------------------------------
# Generic compiled forward (fallback when qk_nope_head_dim > 0)
# ----------------------------------------------------------------------
def _build_compiled_forward():
    """Construct the Torch‑Inductor compiled version of the full MLA forward."""
    # (identical to the reference implementation – kept unchanged)
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
        # 1) Down‑projection
        # -------------------------------------------------
        q_lora = F.linear(x, wDQ)               # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)            # (bs, 1, dkv+d_rope)

        # -------------------------------------------------
        # 2) KV‑cache write
        # -------------------------------------------------
        new_len = cur_len + kv_lora0.shape[1]   # always adds 1 token
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]       # (bs, kv_len, dkv+d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        # -------------------------------------------------
        # 3) Up‑project queries
        # -------------------------------------------------
        q_up = F.linear(q_lora.squeeze(1), wUQ)               # (bs, nh*(d_nope+d_rope))
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)    # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        # -------------------------------------------------
        # 4) KV split / latent projection
        # -------------------------------------------------
        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)  # kv_nope unused
        kv_latent = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        # -------------------------------------------------
        # 5) Project “no‑pe” part of query into latent space
        # -------------------------------------------------
        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk',
                                         q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                         dtype=torch.bfloat16,
                                         device=x.device)

        # -------------------------------------------------
        # 6) RoPE on queries & keys
        # -------------------------------------------------
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, nh, kv_len, d_rope)

        # -------------------------------------------------
        # 7) Scores & soft‑max
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
        # 8) Weighted sum over latent vectors
        # -------------------------------------------------
        latent_agg = torch.matmul(attn, kv_latent)               # (bs, nh, dkv)

        # -------------------------------------------------
        # 9) Project to value space
        # -------------------------------------------------
        y_head = torch.einsum('bhd, hdf -> bhf',
                             latent_agg, wV_T)                  # (bs, nh, dv)

        # -------------------------------------------------
        # 10) Output projection
        # -------------------------------------------------
        y_head_flat = y_head.reshape(x.shape[0], nh * dv)       # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                         # (bs, dim)
        out = out.unsqueeze(1)                                   # (bs, 1, dim)

        return out, kv_data, new_len

    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )

# ----------------------------------------------------------------------
# Triton soft‑max (used only by the generic compiled forward)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # max
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

    # exp & sum
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

    # normalize
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
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward pass.
    * When qk_nope_head_dim == 0 we fuse the Q‑projection and use flash‑attention.
      The heavy projection & projection‑to‑value steps are batched‑matmul fused via torch.compile.
    * Otherwise we fall back to the generic compiled implementation.
    """
    config, x, kv_cache = data

    # -----------------------------------------------------------------
    #  Convenience aliases (all Python ints)
    # -----------------------------------------------------------------
    bs   = config.batch_size
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim

    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+d_rope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # -----------------------------------------------------------------
    #  Prepare RoPE tables (global, computed once)
    # -----------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(d_rope,
                                                    config.max_seq_len,
                                                    x.device)

    # -----------------------------------------------------------------
    #  Fast path – qk_nope_head_dim == 0
    # -----------------------------------------------------------------
    if d_nope == 0:
        # fused Q weight
        wQ_fused = _get_fused_q_weight(wUQ, wDQ, nh, d_rope)

        # per‑head V‑projection weight (cached)
        v_key = id(wUKV)
        if v_key not in _vproj_cache:
            # wUKV shape: ((dv)*nh, dkv)
            _vproj_cache[v_key] = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)
        wV_T = _vproj_cache[v_key]

        # compile the fast kernel once
        global _compiled_fast
        if _compiled_fast is None:
            _compiled_fast = _build_compiled_fast()

        out, new_kv_data, new_len = _compiled_fast(
            x,                         # (bs, 1, dim)
            kv_cache.data,             # (bs, max_seq_len, dkv+d_rope)
            kv_cache.seq_len,          # current cache length (int)
            _cached_cos,               # (max_seq_len, d_rope)
            _cached_sin,               # (max_seq_len, d_rope)
            wQ_fused,                  # (nh*d_rope, dim)
            wDKV,                      # (dkv+d_rope, dim)
            wV_T,                      # (nh, dkv, dv)
            wO,                        # (dim, nh*dv)
            nh, d_rope, dkv, dv, d,    # integer params
        )
        # Update cache state
        kv_cache.data = new_kv_data
        kv_cache.seq_len = int(new_len)
        return out, kv_cache.data

    # -----------------------------------------------------------------
    #  General case – fall back to the compiled reference implementation
    # -----------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,                         # (bs, 1, dim)
        kv_cache.data,             # (bs, max_seq_len, dkv+d_rope)
        kv_cache.seq_len,          # current length (int)
        _cached_cos,               # (max_seq_len, d_rope)
        _cached_sin,
        wDQ, wDKV, wUQ, wUKV, wO,
        nh, d_nope, d_rope, dkv, dv
    )

    # Update cache state
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data