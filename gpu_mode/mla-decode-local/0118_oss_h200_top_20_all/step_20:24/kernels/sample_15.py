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
# Global caches (persist across kernel calls)
# ----------------------------------------------------------------------
_cached_cos = None                      # (max_seq_len, rope_dim)  bfloat16
_cached_sin = None                      # (max_seq_len, rope_dim)  bfloat16
_rope_cache: dict = {}                  # (rope_dim, max_seq_len, device) -> (cos, sin)

_combined_q_weight = {}                 # (id(wUQ), id(wDQ)) -> Tensor[nh*drope, dim]

_compiled_forward = None                # fallback compiled implementation (generic case)

# ----------------------------------------------------------------------
# Helper utilities (RoPE tables, rotate‑half, Triton softmax)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) on *device*."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta_i = 10000^{-i/half}
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)               # (max_seq_len, 1)
        idx = pos * theta[None, :]                                   # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                           # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# Triton row‑wise softmax (used by the generic fallback path)
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

    # ------------------------------------------------------------
    # Max (row‑wise)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Exp & sum
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Normalise
    # ------------------------------------------------------------
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
# Generic compiled forward (fallback when d_nope > 0)
# ----------------------------------------------------------------------
def _build_compiled_forward():
    """Compile the full MLA forward for the generic case."""
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
        # 1️⃣  Down‑projection
        q_lora = F.linear(x, wDQ)                # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)             # (bs, 1, dkv+d_rope)

        # 2️⃣  KV‑cache write
        new_len = cur_len + kv_lora0.shape[1]    # always adds 1 token
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]        # (bs, kv_len, dkv+d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        # 3️⃣  Up‑project queries
        q_up = F.linear(q_lora.squeeze(1), wUQ)               # (bs, nh*(d_nope+d_rope))
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)    # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        # 4️⃣  KV split / latent projection
        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)  # kv_nope unused
        kv_latent = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        # 5️⃣  Project “no‑pe” part of query into latent space
        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk',
                                         q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                         dtype=torch.bfloat16,
                                         device=x.device)

        # 6️⃣  RoPE on queries & keys
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)   # (1,1,d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)

        # 7️⃣  Scores & soft‑max
        scores_rope = torch.matmul(q_rope_rot,
                                   k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

        scores_flat = scores.reshape(x.shape[0] * nh, kv_len)
        attn_flat = _triton_softmax(scores_flat)                # (bs*nh, kv_len)
        attn = attn_flat.view(x.shape[0], nh, kv_len)          # (bs, nh, kv_len)

        # 8️⃣  Weighted aggregation of latent values
        latent_agg = torch.einsum('bhl,bld->bhd', attn, kv_latent)               # (bs, nh, dkv)

        # 9️⃣  Project aggregated latents → value space
        y_head = torch.einsum('bhd, hdf -> bhf',
                             latent_agg, wV_T)                  # (bs, nh, dv)

        # 🔟  Output projection
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
# Fast‑path (qk_nope_head_dim == 0) – fused attention using the
# built‑in flash‑attention kernel + a few small fusions.
# ----------------------------------------------------------------------
def _fast_path(
    bs: int,
    nh: int,
    drope: int,
    dkv: int,
    dv: int,
    wDQ: torch.Tensor,
    wDKV: torch.Tensor,
    wUQ: torch.Tensor,
    wUKV: torch.Tensor,
    wO: torch.Tensor,
    x: torch.Tensor,
    kv_cache: KVCache,
    cos_tbl: torch.Tensor,
    sin_tbl: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward when `qk_nope_head_dim == 0`.
    """
    device = x.device
    dtype = x.dtype

    # -------------------------------
    # 1️⃣  Q projection (combined up‑+down‑proj)
    # -------------------------------
    x2 = x.squeeze(1)                     # (bs, dim)

    # cache combined Q weight (up‑proj ∘ down‑proj)
    global _combined_q_weight
    key_q = (id(wUQ), id(wDQ))
    if key_q not in _combined_q_weight:
        with torch.no_grad():
            _combined_q_weight[key_q] = torch.matmul(wUQ, wDQ)   # (nh*drope, dim)
    combined_q_weight = _combined_q_weight[key_q]                # (nh*drope, dim)

    q_proj = F.linear(x2, combined_q_weight)                     # (bs, nh*drope)
    q_proj = q_proj.view(bs, nh, drope)                         # (bs, nh, drope)

    # -------------------------------
    # 2️⃣  KV down‑projection + cache write
    # -------------------------------
    kv_down = F.linear(x2, wDKV)                                 # (bs, dkv + drope)
    latent_new = kv_down[..., :dkv]                              # (bs, dkv)
    rope_raw   = kv_down[..., dkv:]                              # (bs, drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    # rotate the *new* key (position = cur_len)
    cos_k = cos_tbl[cur_len].view(1, drope)                     # (1, drope)
    sin_k = sin_tbl[cur_len].view(1, drope)
    rope_rot = rope_raw * cos_k + _rotate_half(rope_raw) * sin_k   # (bs, drope)

    # write into cache (single contiguous writes)
    kv_cache.data[:, cur_len:new_len, :dkv] = latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # -------------------------------
    # 3️⃣  Gather keys / values from cache
    # -------------------------------
    kv_len = new_len
    k_rope_all = kv_cache.data[:, :kv_len, dkv:]                 # (bs, kv_len, drope)  – already rotated
    v_latent   = kv_cache.data[:, :kv_len, :dkv]                 # (bs, kv_len, dkv)

    # -------------------------------
    # 4️⃣  RoPE on the query (position = kv_len‑1)
    # -------------------------------
    query_pos = kv_len - 1
    cos_q = cos_tbl[query_pos].view(1, 1, drope)                # (1,1,drope)
    sin_q = sin_tbl[query_pos].view(1, 1, drope)
    q_rot = q_proj * cos_q + _rotate_half(q_proj) * sin_q        # (bs, nh, drope)

    # -------------------------------
    # 5️⃣  Flash‑attention (multi‑query form)
    # -------------------------------
    # Q : (bs, nh, 1, drope)
    # K : (bs, 1 , kv_len, drope)   → broadcasted over heads (multi‑query)
    # V : (bs, 1 , kv_len, dkv)    → same broadcasting
    Q = q_rot.unsqueeze(2)                                     # (bs, nh, 1, drope)
    K = k_rope_all.unsqueeze(1)                                 # (bs, 1, kv_len, drope)
    V = v_latent.unsqueeze(1)                                   # (bs, 1, kv_len, dkv)

    scale = 1.0 / math.sqrt(drope)
    # Note: `scaled_dot_product_attention` automatically picks the flash‑attention
    # kernel when the inputs are on CUDA, fp16/bf16 and suitably sized.
    attn_out = F.scaled_dot_product_attention(
        Q, K, V,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )                                                         # (bs, nh, 1, dkv)

    latent_agg = attn_out.squeeze(2)                           # (bs, nh, dkv)

    # -------------------------------
    # 6️⃣  Per‑head value projection (linear, no einsum)
    # -------------------------------
    # wUKV has shape (nh*dv, dkv) – each head’s dv×dkv matrix is concatenated.
    latent_agg_flat = latent_agg.view(bs * nh, dkv)            # (bs*nh, dkv)
    y_head_flat = F.linear(latent_agg_flat, wUKV)              # (bs*nh, dv)
    y_head = y_head_flat.view(bs, nh, dv)                     # (bs, nh, dv)

    # -------------------------------
    # 7️⃣  Output projection
    # -------------------------------
    y_head_comb = y_head.view(bs, nh * dv)                     # (bs, nh*dv)
    out = F.linear(y_head_comb, wO)                            # (bs, dim)
    out = out.unsqueeze(1)                                     # (bs, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward.
    - If `qk_nope_head_dim == 0` we use a highly‑optimized path that fuses
      RoPE, flash‑attention and the final projection.
    - Otherwise we fall back to a generic compiled implementation.
    """
    config, x, kv_cache = data

    # -----------------------------------------------------------------
    # Convenience aliases (Python ints)
    # -----------------------------------------------------------------
    bs   = config.batch_size
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # -----------------------------------------------------------------
    # Weight aliases (already on device, bfloat16)
    # -----------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight           # (dq, dim)
    wDKV = config.KV_proj_down_weight          # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight             # ((dnope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight            # ((dnope+dv)*nh, dkv)
    wO   = config.wo_weight                    # (dim, nh*dv)

    # -------------------------------------------------
    # 1️⃣  RoPE tables (cached globally)
    # -------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < msl:
        _cached_cos, _cached_sin = _get_rope_tables(drope, msl, x.device)

    # -----------------------------------------------------------------
    # Fast‑path when `qk_nope_head_dim == 0`
    # -----------------------------------------------------------------
    if dnope == 0:
        return _fast_path(
            bs, nh, drope, dkv, dv,
            wDQ, wDKV, wUQ, wUKV, wO,
            x, kv_cache,
            _cached_cos, _cached_sin,
        )

    # -----------------------------------------------------------------
    # General case – fall back to the compiled generic implementation
    # -----------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,                      # (bs, 1, dim)
        kv_cache.data,          # (bs, max_seq_len, dkv+drope)
        kv_cache.seq_len,       # current length (int)
        _cached_cos,            # (max_seq_len, rope_dim)
        _cached_sin,
        wDQ, wDKV, wUQ, wUKV, wO,
        nh, dnope, drope, dkv, dv
    )

    # Update cache state
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data