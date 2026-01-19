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
# Global caches (weights, rope tables, fused matrices, etc.)
# ----------------------------------------------------------------------
_q_fused_cache = {}          # (id(wUQ), id(wDQ)) → (nh·d_rope, dim)
_wV_T_cache = {}            # (id(wUKV), nh, dv, dkv) → (nh, dkv, dv)
_rope_cache = {}            # (rope_dim, max_seq_len, device) → (cos, sin)
_cached_cos = None          # (max_seq_len, rope_dim)   bfloat16
_cached_sin = None          # (max_seq_len, rope_dim)   bfloat16

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _get_fused_q(wUQ: torch.Tensor, wDQ: torch.Tensor) -> torch.Tensor:
    """Fuse Q‑down and Q‑up matrices → (nh·d_rope, dim)."""
    key = (id(wUQ), id(wDQ))
    if key not in _q_fused_cache:
        _q_fused_cache[key] = torch.matmul(wUQ, wDQ)  # (nh·d_rope, dim)   bfloat16
    return _q_fused_cache[key]


def _get_wV_T(wUKV: torch.Tensor, nh: int, dv: int, dkv: int) -> torch.Tensor:
    """Cache the per‑head value‑projection matrix (dkv, dv)."""
    key = (id(wUKV), nh, dv, dkv)
    if key not in _wV_T_cache:
        # wUKV shape: ((d_nope+dv)*nh, dkv) → (dv·nh, dkv) because d_nope == 0
        _wV_T_cache[key] = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()  # (nh, dkv, dv)
    return _wV_T_cache[key]


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables for rotary embeddings of size `dim`."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)   # (max_seq_len, 1)
        idx = pos * theta[None, :]                                                    # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                          # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# Fallback for the generic case (qk_nope_head_dim != 0)
# ----------------------------------------------------------------------
_compiled_fallback = None
def _fallback_forward(
    x: torch.Tensor,
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
    dv: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Thin wrapper around a generic compiled implementation (exactly the reference)."""
    global _compiled_fallback
    if _compiled_fallback is None:
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
            q_up = F.linear(q_lora.squeeze(1), wUQ)               # (bs, nh*d_rope)
            q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)    # (bs, nh, d_nope+d_rope)
            q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

            # -------------------------------------------------
            # 4) KV split / latent projection
            # -------------------------------------------------
            kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)  # kv_nope unused
            kv_latent = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

            wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
            wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
            wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1).contiguous()          # (nh, dkv, dv)

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

        _compiled_fallback = torch.compile(
            _inner,
            backend="inductor",
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
    out, new_kv_data, new_len = _compiled_fallback(
        x,
        kv_data,
        cur_len,
        cos_tbl,
        sin_tbl,
        wDQ,
        wDKV,
        wUQ,
        wUKV,
        wO,
        nh,
        d_nope,
        d_rope,
        dkv,
        dv,
    )
    return out, new_kv_data, new_len


# ----------------------------------------------------------------------
# Fast‑path for the common case: qk_nope_head_dim == 0
# ----------------------------------------------------------------------
_compiled_fast = None
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized MLA forward.  When `qk_nope_head_dim == 0` we use a highly
    optimized compiled implementation that fuses the down‑/up‑projections,
    RoPE rotation, KV‑cache update and per‑head value projection.
    For the generic case we fall back to the reference compiled kernel.
    """
    config, x, kv_cache = data

    # -----------------------------------------------------------------
    # Convenience aliases (plain Python ints)
    # -----------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # may be >1 for pre‑fill
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # -----------------------------------------------------------------
    # Model weights (already on device & bf16)
    # -----------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight        # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight          # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # -----------------------------------------------------------------
    # Prepare RoPE tables (cached globally)
    # -----------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < msl:
        _cached_cos, _cached_sin = _get_rope_tables(drope, msl, x.device)

    # -----------------------------------------------------------------
    # Fast‑path when there is no “no‑PE” head dimension
    # -----------------------------------------------------------------
    if d_nope == 0:
        # Fused Q‑projection weight (cached)
        wQ_fused = _get_fused_q(wUQ, wDQ)               # (nh*drope, dim)

        # Per‑head value‑projection matrix (cached)
        wV_T = _get_wV_T(wUKV, nh, dv, dkv)             # (nh, dkv, dv)

        global _compiled_fast
        if _compiled_fast is None:
            def _inner_fast(x, kv_data, cur_len,
                            cos_tbl, sin_tbl,
                            wQ_fused, wDKV, wV_T, wO,
                            nh, drope, dkv, dv):
                """
                Compiled MLA forward for the d_nope == 0 case.
                Returns (out, updated_kv, new_len).
                """
                bs, seq_len, dim = x.shape

                # -------------------------------------------------
                # 1️⃣ Q projection + RoPE
                # -------------------------------------------------
                q = torch.nn.functional.linear(x, wQ_fused)                 # (bs, S, nh*drope)
                q = q.view(bs, seq_len, nh, drope).permute(0, 2, 1, 3)    # (bs, nh, S, drope)

                cur = int(cur_len)
                new_len = cur + seq_len
                pos = torch.arange(cur, new_len, device=x.device, dtype=torch.int64)  # (S,)

                # RoPE tables for the new positions
                cos_pos = cos_tbl[pos].view(1, 1, seq_len, drope)   # (1,1,S,d)
                sin_pos = sin_tbl[pos].view(1, 1, seq_len, drope)

                # Apply RoPE to queries
                q_rot = q * cos_pos + _rotate_half(q) * sin_pos   # (bs, nh, S, drope)

                # -------------------------------------------------
                # 2️⃣ KV down‑projection + RoPE + KV‑cache update
                # -------------------------------------------------
                kv = torch.nn.functional.linear(x, wDKV)               # (bs, S, dkv+drope)
                kv_latent = kv[..., :dkv]                              # (bs, S, dkv)
                kv_rope_raw = kv[..., dkv:]                            # (bs, S, drope)

                # Rotate the rope part for the new tokens
                cos_s = cos_pos.squeeze(0).squeeze(0)   # (S, drope)
                sin_s = sin_pos.squeeze(0).squeeze(0)   # (S, drope)
                kv_rope = kv_rope_raw * cos_s + _rotate_half(kv_rope_raw) * sin_s   # (bs, S, drope)

                # Write into KV‑cache (in‑place)
                kv_data[:, cur:new_len, :dkv] = kv_latent
                kv_data[:, cur:new_len, dkv:] = kv_rope

                # -------------------------------------------------
                # 3️⃣ Retrieve complete cache and build K/V for attention
                # -------------------------------------------------
                kv_all = kv_data[:, :new_len, :]                       # (bs, new_len, dkv+drope)
                kv_all_latent = kv_all[..., :dkv]                      # (bs, new_len, dkv)
                kv_all_rope = kv_all[..., dkv:]                        # (bs, new_len, drope)

                # Keys – rope part broadcast over heads
                k = kv_all_rope[:, None, :, :].expand(-1, nh, -1, -1)  # (bs, nh, new_len, drope)
                # Values – latent part broadcast over heads
                v = kv_all_latent[:, None, :, :].expand(-1, nh, -1, -1)  # (bs, nh, new_len, dkv)

                # -------------------------------------------------
                # 4️⃣ Scaled‑dot‑product attention (Flash‑Attention)
                # -------------------------------------------------
                scale = 1.0 / math.sqrt(drope)
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q_rot, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale
                )   # (bs, nh, S, dkv)

                # -------------------------------------------------
                # 5️⃣ Per‑head value projection (latent → dv)
                # -------------------------------------------------
                # (bs, nh, S, dkv) -> (bs*nh, S, dkv)
                attn_flat = attn_out.reshape(bs * nh, seq_len, dkv)   # (B*H, S, dkv)
                # Expand wV_T to match batch*head dimension without materialising a huge tensor
                wV_T_exp = wV_T.unsqueeze(0).expand(bs, nh, dkv, dv).reshape(bs * nh, dkv, dv)
                y_head = torch.bmm(attn_flat, wV_T_exp)               # (B*H, S, dv)
                y_head = y_head.view(bs, nh, seq_len, dv)            # (bs, nh, S, dv)

                # -------------------------------------------------
                # 6️⃣ Output projection
                # -------------------------------------------------
                y_flat = y_head.permute(0, 2, 1, 3).reshape(bs, seq_len, nh * dv)  # (bs, S, nh*dv)
                out = torch.nn.functional.linear(y_flat, wO)                         # (bs, S, dim)

                return out, kv_data, new_len

            _compiled_fast = torch.compile(
                _inner_fast,
                backend="inductor",
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
        # -----------------------------------------------------------------
        # Execute compiled fast‑path
        # -----------------------------------------------------------------
        out, new_kv_data, new_len = _compiled_fast(
            x,
            kv_cache.data,
            kv_cache.seq_len,
            _cached_cos,
            _cached_sin,
            wQ_fused,
            wDKV,
            wV_T,
            wO,
            nh,
            drope,
            dkv,
            dv,
        )
        kv_cache.data = new_kv_data
        kv_cache.seq_len = int(new_len)
        return out, kv_cache.data

    # -----------------------------------------------------------------
    # Generic case – fall back to reference implementation
    # -----------------------------------------------------------------
    out, new_kv, new_len = _fallback_forward(
        x,
        kv_cache.data,
        kv_cache.seq_len,
        _cached_cos,
        _cached_sin,
        wDQ,
        wDKV,
        wUQ,
        wUKV,
        wO,
        nh,
        d_nope,
        drope,
        dkv,
        dv,
    )
    kv_cache.data = new_kv
    kv_cache.seq_len = int(new_len)
    return out, kv_cache.data