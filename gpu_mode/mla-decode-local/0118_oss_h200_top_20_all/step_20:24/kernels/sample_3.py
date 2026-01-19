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

# --------------------------------------------------------------
# Global caches (rope tables, per‑head value‑projection matrix, compiled kernels)
# --------------------------------------------------------------
_cached_cos = None          # (max_seq_len, rope_dim)  bfloat16
_cached_sin = None          # (max_seq_len, rope_dim) bfloat16
_rope_cache = {}            # reusable cosine / sine tables
_wV_T_cache = {}            # (id(wUKV), nh, dv, dkv) -> Tensor
_compiled_forward = None    # compiled generic forward (fallback when d_nope > 0)

# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Create (or fetch) cosine / sine tables for rotary embeddings."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(
            half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)   # (max_seq_len, 1)
        idx = pos * theta[None, :]                         # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


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

    # -----------------------------------------------------------------
    # max
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # exp & sum
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # normalize
    # -----------------------------------------------------------------
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


def _get_wV_T(wUKV: torch.Tensor, nh: int, dv: int, dkv: int) -> torch.Tensor:
    """Cache the per‑head value‑projection matrix (dkv‑shaped)."""
    key = (id(wUKV), nh, dv, dkv)
    if key not in _wV_T_cache:
        # wUKV shape ((d_nope+dv)*nh, dkv) – for d_nope==0 it is (dv*nh, dkv)
        _wV_T_cache[key] = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()
    return _wV_T_cache[key]


# ----------------------------------------------------------------------
# Fused attention kernel (RoPE + stable softmax + weighted‑sum)
# ----------------------------------------------------------------------
@triton.jit
def _fused_attn_kernel(
    out_ptr,                     # (B*H, Dkv) output
    q_ptr,                       # (B*H, Drope)  query (already RoPE‑rotated)
    k_ptr,                       # (B, L, Drope) key cache (already RoPE‑rotated)
    v_ptr,                       # (B, L, Dkv)   value cache
    # strides
    out_stride_batch, out_stride_dim,
    q_stride_batch, q_stride_dim,
    k_stride_batch, k_stride_len, k_stride_dim,
    v_stride_batch, v_stride_len, v_stride_dim,
    # scalars
    inv_sqrt_d: tl.float32,      # 1/sqrt(Drope)  – scaling factor for Q·K
    kv_len: tl.int32,            # actual length of K/V cache (runtime)
    Drope: tl.constexpr,         # rope dimension
    Dkv: tl.constexpr,           # latent value dimension
    nh: tl.constexpr,            # number of heads
    BLOCK_K: tl.constexpr,       # block size along sequence dimension
    MAX_KV: tl.constexpr,        # compile‑time upper bound (max_seq_len)
):
    pid = tl.program_id(0)                     # pid ∈ [0, B*H)
    batch = pid // nh
    head = pid % nh

    # -----------------------------------------------------------------
    # Load query (single vector) and apply scaling
    # -----------------------------------------------------------------
    q_off = batch * q_stride_batch + head * q_stride_dim
    q = tl.load(q_ptr + q_off + tl.arange(0, Drope) * q_stride_dim)
    q_f = tl.cast(q, tl.float32) * inv_sqrt_d      # scaled query

    # -----------------------------------------------------------------
    # Accumulators: max, sum(exp), weighted‑sum of values
    # -----------------------------------------------------------------
    max_val = tl.full([1], -float('inf'), tl.float32)      # current max
    sum_exp = tl.full([1], 0.0, tl.float32)               # Σ exp
    out_acc = tl.zeros([Dkv], dtype=tl.float32)          # Σ exp * V

    # -----------------------------------------------------------------
    # Iterate over the KV cache in blocks
    # -----------------------------------------------------------------
    for start in range(0, MAX_KV, BLOCK_K):
        cur = start + tl.arange(0, BLOCK_K)                 # (BLOCK_K)
        mask = cur < kv_len                                 # valid rows mask

        # ---------- keys ----------
        k_off = batch * k_stride_batch + start * k_stride_len
        k = tl.load(
            k_ptr + k_off
            + tl.arange(0, BLOCK_K)[:, None] * k_stride_len
            + tl.arange(0, Drope)[None, :] * k_stride_dim,
            mask=mask[:, None],
            other=0.0,
        )
        k_f = tl.cast(k, tl.float32)

        # ---------- scores ----------
        # dot(q, k_i) for every i in block
        scores = tl.sum(q_f[None, :] * k_f, axis=1)          # (BLOCK_K)
        # mask out‑of‑range entries (set to -inf so they never become max)
        scores = tl.where(mask, scores, -float('inf'))
        block_max = tl.max(scores)

        # ---------- stable reduction ----------
        new_max = tl.maximum(max_val[0], block_max)
        # factor to scale previous accumulators to the new max
        scale = tl.exp(max_val[0] - new_max)

        # ---------- exponentials ----------
        exp_scores = tl.exp(scores - new_max)                # (BLOCK_K)
        sum_exp_block = tl.sum(exp_scores)                  # scalar

        # ---------- values ----------
        v_off = batch * v_stride_batch + start * v_stride_len
        v = tl.load(
            v_ptr + v_off
            + tl.arange(0, BLOCK_K)[:, None] * v_stride_len
            + tl.arange(0, Dkv)[None, :] * v_stride_dim,
            mask=mask[:, None],
            other=0.0,
        )
        v_f = tl.cast(v, tl.float32)

        # ---------- weighted sum ----------
        weighted_v = v_f * exp_scores[:, None]               # (BLOCK_K, Dkv)
        sum_v_block = tl.sum(weighted_v, axis=0)            # (Dkv)

        # ---------- update accumulators ----------
        out_acc = out_acc * scale + sum_v_block
        sum_exp = sum_exp * scale + sum_exp_block
        max_val[0] = new_max

    # -----------------------------------------------------------------
    # Normalisation and store
    # -----------------------------------------------------------------
    out = out_acc / sum_exp[0]                              # (Dkv)
    out_off = batch * out_stride_batch + head * out_stride_dim
    tl.store(out_ptr + out_off + tl.arange(0, Dkv) * out_stride_dim,
             tl.cast(out, tl.bfloat16))


def _fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     inv_sqrt_d: float, kv_len: int, max_seq_len: int) -> torch.Tensor:
    """
    q: (B, H, Drope)   already RoPE‑rotated
    k: (B, L, Drope)   cache (already RoPE‑rotated)
    v: (B, L, Dkv)     cache (latent values)
    Returns: (B, H, Dkv) attention‑aggregated latent vectors
    """
    B, H, Drope = q.shape
    Dkv = v.shape[2]

    # reshape for kernel (flatten B*H dimension)
    q_flat = q.reshape(B * H, Drope).contiguous()

    out = torch.empty(B * H, Dkv, dtype=torch.bfloat16, device=q.device)

    grid = (B * H,)                      # one program per (batch, head)
    BLOCK_K = 64                          # tune‑able; 64 works well for rope dim 64
    _fused_attn_kernel[grid](
        out,
        q_flat,
        k,
        v,
        # strides
        out.stride(0), out.stride(1),
        q_flat.stride(0), q_flat.stride(1),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        # scalars
        inv_sqrt_d,
        kv_len,
        Drope,
        Dkv,
        H,
        BLOCK_K=BLOCK_K,
        MAX_KV=max_seq_len,
        num_warps=4,
    )
    return out.view(B, H, Dkv)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward pass.
    * When qk_nope_head_dim == 0 we use a highly‑optimised branch that
      fuses down‑projection, RoPE, attention (via a custom fused kernel
      that performs stable soft‑max and weighted‑sum) and the per‑head
      value projection.
    * Otherwise we fall back to the compiled generic implementation.
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Convenience aliases
    # --------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len                 # always 1 for our use‑case
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+d_rope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # Build / fetch RoPE tables (once per model)
    # --------------------------------------------------------------
    global _cached_cos, _cached_sin
    if (_cached_cos is None) or (_cached_cos.shape[0] < msl):
        _cached_cos, _cached_sin = _get_rope_tables(d_rope, msl, x.device)

    # --------------------------------------------------------------
    # Fast path when there is no “no‑pe” part (d_nope == 0)
    # --------------------------------------------------------------
    if d_nope == 0:
        # ------------------------------------------------------------------
        # 0) Reduce sequence dimension (always 1)
        # ------------------------------------------------------------------
        x_center = x.squeeze(1)                       # (bs, dim)

        # ------------------------------------------------------------------
        # 1) Down‑project Q and KV in a single matmul (two weight blocks concatenated)
        # ------------------------------------------------------------------
        w_down = torch.cat([wDQ, wDKV], dim=0)        # ((dq+dkv+d_rope), dim)
        down = F.linear(x_center, w_down)             # (bs, dq + dkv + d_rope)

        q_lora   = down[:, :dq]                       # (bs, dq)
        kv_down  = down[:, dq:]                       # (bs, dkv + d_rope)

        # ------------------------------------------------------------------
        # 2) Write new token into KV cache (latent + rotated key)
        # ------------------------------------------------------------------
        cur_len = kv_cache.seq_len                     # length BEFORE inserting this token
        new_len = cur_len + 1

        latent_new = kv_down[..., :dkv]                # (bs, dkv)
        rope_raw   = kv_down[..., dkv:]                # (bs, d_rope)

        # Rotate key for current position (cur_len)
        cos_pos = _cached_cos[cur_len].view(1, d_rope)   # (1, d_rope)
        sin_pos = _cached_sin[cur_len].view(1, d_rope)
        rope_rot = rope_raw * cos_pos + _rotate_half(rope_raw) * sin_pos

        # store into cache (latent first, then rotated key)
        kv_cache.data[:, cur_len, :dkv] = latent_new
        kv_cache.data[:, cur_len, dkv:] = rope_rot
        kv_cache.seq_len = new_len                      # advance cache pointer

        # ------------------------------------------------------------------
        # 3) Up‑project Q and apply RoPE (position = new_len‑1)
        # ------------------------------------------------------------------
        q_up = F.linear(q_lora, wUQ)                   # (bs, nh*d_rope)
        q_up = q_up.view(bs, nh, d_rope)               # (bs, nh, d_rope)

        cos_q = _cached_cos[new_len - 1].view(1, 1, d_rope)   # (1,1,d_rope)
        sin_q = _cached_sin[new_len - 1].view(1, 1, d_rope)
        q_rot = q_up * cos_q + _rotate_half(q_up) * sin_q   # (bs, nh, d_rope)

        # ------------------------------------------------------------------
        # 4) Gather KV cache up to the new token
        # ------------------------------------------------------------------
        kv_all = kv_cache.data[:, :new_len, :]               # (bs, new_len, dkv+d_rope)
        v_all = kv_all[..., :dkv]                            # (bs, new_len, dkv)
        k_all = kv_all[..., dkv:]                            # (bs, new_len, d_rope) – already RoPE‑rotated

        # ------------------------------------------------------------------
        # 5) Fused attention → latent aggregation (bs, nh, dkv)
        # ------------------------------------------------------------------
        inv_sqrt = 1.0 / math.sqrt(d_rope)                  # scaling factor 1/√(Drope)
        latent_agg = _fused_attention(q_rot, k_all, v_all,
                                      inv_sqrt, new_len, msl)   # (bs, nh, dkv)

        # ------------------------------------------------------------------
        # 6) Per‑head value projection (cached matrix)
        # ------------------------------------------------------------------
        wV_T = _get_wV_T(wUKV, nh, dv, dkv)                # (nh, dkv, dv)
        y_head = torch.einsum('bhd, hdf -> bhf',
                             latent_agg, wV_T)               # (bs, nh, dv)

        # ------------------------------------------------------------------
        # 7) Output projection
        # ------------------------------------------------------------------
        y_head_flat = y_head.reshape(bs, nh * dv)           # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                     # (bs, dim)
        out = out.unsqueeze(1)                               # (bs, 1, dim)

        return out, kv_cache.data

    # --------------------------------------------------------------
    # General case – fallback to compiled generic implementation
    # --------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        # (The generic implementation from the reference code is kept unchanged)
        # It can be left as‑is because the only performance‑critical path
        # for the provided benchmark uses d_nope == 0.
        # ------------------------------------------------------------------
        def _build_compiled_forward():
            """Construct the Torch‑Inductor compiled version of the full MLA forward."""
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
                kv_lora0 = F.linear(x, wDKV)            # (bs, 1, dkv + d_rope)

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
                k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)

                # -------------------------------------------------
                # 7) Scores & soft‑max
                # -------------------------------------------------
                scores_rope = torch.matmul(q_rope_rot,
                                           k_rope_rot.transpose(-1, -2))      # (bs, nh, kv_len)
                scores_nope = torch.matmul(q_nope_latent,
                                           kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
                scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))
                scores_flat = scores.reshape(x.shape[0] * nh, kv_len)
                attn_flat = _triton_softmax(scores_flat)             # (bs*nh, kv_len)
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
        # compile once
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,                      # (bs, 1, dim)
        kv_cache.data,          # (bs, max_seq_len, dkv+d_rope)
        kv_cache.seq_len,       # current cached length (int)
        _cached_cos,            # (max_seq_len, d_rope)
        _cached_sin,
        wDQ, wDKV, wUQ, wUKV, wO,
        nh, d_nope, d_rope, dkv, dv
    )

    # Update cache state
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data