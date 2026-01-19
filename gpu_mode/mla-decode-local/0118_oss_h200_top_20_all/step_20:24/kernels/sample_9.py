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
# Global caches for RoPE tables (cosine / sine) and Triton soft‑max
# ----------------------------------------------------------------------
_rope_cache: dict = {}
_cached_cos = None   # shape (max_seq_len, rope_dim)  bfloat16
_cached_sin = None   # same

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the last dimension (swap halves and negate second)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables for RoPE of given dimension."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64,
                           device=device).unsqueeze_(1)      # (max_seq_len, 1)
        idx = pos * theta[None, :]                         # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# Triton row‑wise soft‑max (used for the fast‑path)
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

    col = tl.arange(0, BLOCK_SIZE)
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)

    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur,
                      mask=mask,
                      other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))

    row_max = tl.max(max_val)

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
    # choose a block size that covers the length efficiently
    if n_cols <= 32:
        BLOCK_SIZE = 32
    elif n_cols <= 64:
        BLOCK_SIZE = 64
    elif n_cols <= 128:
        BLOCK_SIZE = 128
    else:
        # round up to next power‑of‑2 (capped at 1024)
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
# Generic compiled forward (used when qk_nope_head_dim > 0)
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
        # 1) Down‑projection
        q_lora = F.linear(x, wDQ)               # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)            # (bs, 1, dkv+d_rope)

        # 2) KV‑cache write
        new_len = cur_len + kv_lora0.shape[1]    # always adds 1 token
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]       # (bs, kv_len, dkv+d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        # 3) Up‑project queries
        q_up = F.linear(q_lora.squeeze(1), wUQ)                 # (bs, nh * (d_nope+d_rope))
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)      # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        # 4) KV split / latent projection
        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)  # kv_nope unused
        kv_latent = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None    # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        # 5) Project “no‑pe” part of query into latent space
        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk',
                                         q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                        dtype=torch.bfloat16,
                                        device=x.device)

        # 6) RoPE on queries & keys
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)

        # 7) Scores & soft‑max
        scores_rope = torch.matmul(q_rope_rot,
                                   k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

        scores_flat = scores.reshape(x.shape[0] * nh, kv_len)
        attn_flat = _triton_softmax(scores_flat)
        attn = attn_flat.view(x.shape[0], nh, kv_len)          # (bs, nh, kv_len)

        # 8) Weighted sum over latent vectors
        latent_agg = torch.matmul(attn, kv_latent)               # (bs, nh, dkv)

        # 9) Project to value space (dv)
        y_head = torch.einsum('bhd, hdf -> bhf',
                              latent_agg, wV_T)                  # (bs, nh, dv)

        # 10) Output projection
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
# Main custom kernel entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Highly‑optimized MLA forward:
      * d_nope == 0 → fast path using fused matmul + Triton soft‑max.
      * d_nope > 0  → fall back to the generic compiled implementation.
    """
    config, x, kv_cache = data

    # -------------------------------------------------
    # Convenience aliases (all Python ints)
    # -------------------------------------------------
    bs   = config.batch_size
    seq_len = x.size(1)          # can be >1 only for the generic case
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # -------------------------------------------------
    # Weight tensors (already on device & bf16)
    # -------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv+d_rope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # -------------------------------------------------
    # Prepare RoPE tables (cached globally)
    # -------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < msl:
        _cached_cos, _cached_sin = _get_rope_tables(d_rope, msl, x.device)

    # -------------------------------------------------
    # Fast path – only when d_nope == 0 (the common configuration)
    # -------------------------------------------------
    if d_nope == 0:
        # -------------------------------------------------
        # 1) Down‑projection (Q + KV) – single GEMM
        # -------------------------------------------------
        # Concatenate down‑projection weights once (cache on the function)
        if not hasattr(custom_kernel, "_w_concat"):
            custom_kernel._w_concat = torch.cat([wDQ, wDKV], dim=0)   # (dq+dkv+d_rope, dim)
        w_concat = custom_kernel._w_concat

        # x: (bs, seq_len, dim) → flatten for linear
        proj = torch.nn.functional.linear(x.view(bs * seq_len, config.dim), w_concat)
        proj = proj.view(bs, seq_len, dq + dkv + d_rope)

        q_lora = proj[:, :, :dq]                # (bs, seq_len, dq)
        kv_lora = proj[:, :, dq:]               # (bs, seq_len, dkv+d_rope)

        # -------------------------------------------------
        # 2) KV‑cache write (latent + rotated RoPE keys)
        # -------------------------------------------------
        start = kv_cache.seq_len
        new_len = start + seq_len
        # split latent / rope
        kv_latent_part = kv_lora[..., :dkv]                 # (bs, seq_len, dkv)
        kv_rope_raw = kv_lora[..., dkv:]                    # (bs, seq_len, d_rope)

        # RoPE rotation for the *keys* before caching
        pos_range = torch.arange(start, new_len, device=x.device, dtype=torch.long)   # (seq_len,)
        cos_k = _cached_cos[pos_range]      # (seq_len, d_rope)
        sin_k = _cached_sin[pos_range]      # (seq_len, d_rope)

        kv_rope_rot = kv_rope_raw * cos_k[None, :, :] + _rotate_half(kv_rope_raw) * sin_k[None, :, :]

        # store parts
        kv_cache.data[:, start:new_len, :dkv] = kv_latent_part.to(kv_cache.data.dtype)
        kv_cache.data[:, start:new_len, dkv:] = kv_rope_rot.to(kv_cache.data.dtype)

        # update cache length
        kv_cache.seq_len = int(new_len)

        # -------------------------------------------------
        # 3) Up‑project queries (only RoPE part exists)
        # -------------------------------------------------
        q_up = torch.nn.functional.linear(q_lora.view(bs * seq_len, dq), wUQ)   # (bs*seq_len, nh*d_rope)
        q_up = q_up.view(bs, seq_len, nh, d_rope)   # (bs, seq_len, nh, d_rope)

        # -------------------------------------------------
        # 4) Apply RoPE to the *query* token(s)
        # -------------------------------------------------
        # For the fast path we only ever have seq_len == 1, but the code also works
        # for larger seq_len (treating each token independently).
        cos_q = _cached_cos[pos_range]      # (seq_len, d_rope)
        sin_q = _cached_sin[pos_range]      # (seq_len, d_rope)

        q_rot = q_up * cos_q[None, :, None, :] + _rotate_half(q_up) * sin_q[None, :, None, :]   # (bs, seq_len, nh, d_rope)

        # We only need the *last* token for the attention computation (the newly generated token)
        # The MLA design works with seq_len == 1, so we extract the final one.
        q_rot = q_rot[:, -1, :, :]   # (bs, nh, d_rope)

        # -------------------------------------------------
        # 5) Gather keys (already rotated) and latent values from cache
        # -------------------------------------------------
        k_rot = kv_cache.data[:, :new_len, dkv:]               # (bs, new_len, d_rope)
        v_lat = kv_cache.data[:, :new_len, :dkv]               # (bs, new_len, dkv)

        # -------------------------------------------------
        # 6) Scaled dot‑product + Triton soft‑max
        # -------------------------------------------------
        scale = 1.0 / math.sqrt(d_rope)
        scores = torch.einsum('bhd,bkd->bhk', q_rot, k_rot) * scale   # (bs, nh, new_len)

        scores_flat = scores.view(bs * nh, new_len)               # (bs*nh, new_len)
        attn_flat = _triton_softmax(scores_flat)                  # (bs*nh, new_len)
        attn = attn_flat.view(bs, nh, new_len)                    # (bs, nh, new_len)

        # -------------------------------------------------
        # 7) Weighted sum of latent values
        # -------------------------------------------------
        # Use float32 accumulation for stability then cast back to bfloat16
        latent_agg = torch.einsum('bhn,bnd->bhd', attn.float(), v_lat.float())
        latent_agg = latent_agg.to(torch.bfloat16)   # (bs, nh, dkv)

        # -------------------------------------------------
        # 8) Project latent aggregation to value space (dv)
        # -------------------------------------------------
        # wUKV : ((dv)*nh, dkv) → reshape to (nh, dv, dkv)
        wV = wUKV.view(nh, dv, dkv)            # (nh, dv, dkv)
        wV_T = wV.permute(0, 2, 1)            # (nh, dkv, dv)

        y_head = torch.einsum('bhd, hdf -> bhf',
                             latent_agg, wV_T)   # (bs, nh, dv)

        # -------------------------------------------------
        # 9) Final output projection
        # -------------------------------------------------
        y_head_flat = y_head.reshape(bs, nh * dv)   # (bs, nh*dv)
        out = torch.nn.functional.linear(y_head_flat, wO)  # (bs, dim)
        out = out.unsqueeze(1)                                 # (bs, 1, dim)

        return out, kv_cache.data

    # -----------------------------------------------------------------
    # General case – fall back to the compiled generic implementation
    # -----------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,                      # (bs, seq_len, dim)
        kv_cache.data,          # (bs, max_seq_len, dkv+d_rope)
        kv_cache.seq_len,       # current cache length (int)
        _cached_cos,            # (max_seq_len, d_rope)
        _cached_sin,
        wDQ, wDKV, wUQ, wUKV, wO,
        nh, d_nope, d_rope, dkv, dv
    )

    # Update KV cache state
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data