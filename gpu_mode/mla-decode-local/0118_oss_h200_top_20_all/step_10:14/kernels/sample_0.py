### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# Global caches (rope tables, fused weight, compiled forward)
# ----------------------------------------------------------------------
_rope_cache: dict = {}
_cached_cos: torch.Tensor = None
_cached_sin: torch.Tensor = None
_fused_q_weight_cache: dict = {}
_compiled_forward = None

# ----------------------------------------------------------------------
#  RoPE table construction (cached)
# ----------------------------------------------------------------------
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta: (half,)
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
        idx = pos * theta[None, :]                 # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)        # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Triton softmax kernel (row‑wise)
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

    # -------- max ----------
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

    # -------- exp & sum ----------
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

    # -------- normalize ----------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur,
                      mask=mask,
                      other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur,
                 tl.cast(norm, tl.bfloat16),
                 mask=mask)

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
        N=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
#  Compiled forward for the general (d_nope > 0) case
# ----------------------------------------------------------------------
def _build_compiled_forward():
    """Creates a torch‑compiled version of the MLA forward for d_nope>0."""
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
        # --------------------------------------------------------------
        # 1) Down‑projection
        # --------------------------------------------------------------
        q_lora   = F.linear(x, wDQ)          # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)         # (bs, 1, dkv+d_rope)

        # --------------------------------------------------------------
        # 2) KV‑cache write
        # --------------------------------------------------------------
        new_len = cur_len + kv_lora0.shape[1]          # always adds exactly one token
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]              # (bs, kv_len, dkv+d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        # --------------------------------------------------------------
        # 3) Up‑project queries (general case)
        # --------------------------------------------------------------
        q_up = F.linear(q_lora.squeeze(1), wUQ)                     # (bs, nh*(d_nope+d_rope))
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)          # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        # --------------------------------------------------------------
        #    KV split / up‑project
        # --------------------------------------------------------------
        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)   # kv_nope unused later
        kv_latent = kv_lora[..., :dkv]                                     # (bs, kv_len, dkv)

        # --------------------------------------------------------------
        #    Prepare weight slices for the latent → value projection
        # --------------------------------------------------------------
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        # --------------------------------------------------------------
        #    Project query‑nope into latent space
        # --------------------------------------------------------------
        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk',
                                         q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                         dtype=torch.bfloat16,
                                         device=x.device)

        # --------------------------------------------------------------
        #    RoPE on queries
        # --------------------------------------------------------------
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)   # (1,1,d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        # --------------------------------------------------------------
        #    RoPE on keys (shared across heads)
        # --------------------------------------------------------------
        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)

        # --------------------------------------------------------------
        #    Scores (rope part + nope part)
        # --------------------------------------------------------------
        scores_rope = torch.matmul(q_rope_rot,
                                   k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

        # --------------------------------------------------------------
        #    Softmax (row‑wise, Triton)
        # --------------------------------------------------------------
        scores_flat = scores.reshape(x.shape[0] * nh, kv_len)
        attn_flat = _triton_softmax(scores_flat)
        attn = attn_flat.view(x.shape[0], nh, kv_len)          # (bs, nh, kv_len)

        # --------------------------------------------------------------
        #    Weighted sum over latent vectors
        # --------------------------------------------------------------
        latent_agg = torch.matmul(attn, kv_latent)               # (bs, nh, dkv)

        # --------------------------------------------------------------
        #    Project to value space
        # --------------------------------------------------------------
        y_head = torch.einsum('bhd, hdf -> bhf',
                             latent_agg, wV_T)                  # (bs, nh, dv)

        # --------------------------------------------------------------
        #    Output projection
        # --------------------------------------------------------------
        y_head_flat = y_head.reshape(x.shape[0], nh * dv)       # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                         # (bs, dim)
        out = out.unsqueeze(1)                                   # (bs, 1, dim)

        return out, kv_data, new_len

    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False   # only cur_len varies at runtime
    )

# ----------------------------------------------------------------------
#  Helper – half‑rotation (used only in compiled forward)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
#  Main entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized MLA forward – fast‑path when qk_nope_head_dim == 0,
    otherwise falls back to a torch‑compiled implementation.
    Returns (output, kv_cache.data).
    """
    config, x, kv_cache = data

    # -----------------------------------------------------------------
    #  Local shortcuts (all plain Python ints)
    # -----------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    d  = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim
    msl = config.max_seq_len

    # -----------------------------------------------------------------
    #  Weights (already on device, bfloat16)
    # -----------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+d_rope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # -----------------------------------------------------------------
    #  Ensure RoPE tables are cached
    # -----------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape != (msl, d_rope):
        _cached_cos, _cached_sin = _get_rope_tables(d_rope, msl, x.device)

    # -----------------------------------------------------------------
    #  Fast path – d_nope == 0 (no‑PE component)
    # -----------------------------------------------------------------
    if d_nope == 0:
        # -------------------------------------------------------------
        # 1) Fused Q‑projection weight
        # -------------------------------------------------------------
        global _fused_q_weight_cache
        fused_key = (id(wUQ), id(wDQ), nh, d_rope)
        wQ_fused = _fused_q_weight_cache.get(fused_key)
        if wQ_fused is None:
            wQ_fused = torch.matmul(wUQ, wDQ)               # (nh*d_rope, dim)
            _fused_q_weight_cache[fused_key] = wQ_fused

        # -------------------------------------------------------------
        # 2) Input flatten (seq_len == 1)
        # -------------------------------------------------------------
        x2d = x.squeeze(1)                                 # (bs, dim)

        # -------------------------------------------------------------
        # 3) Queries (linear + reshape)
        # -------------------------------------------------------------
        q = F.linear(x2d, wQ_fused)                        # (bs, nh*d_rope)
        q = q.view(bs, nh, d_rope)                         # (bs, nh, d_rope)

        # -------------------------------------------------------------
        # 4) KV raw projection (latent + rope part)
        # -------------------------------------------------------------
        kv_raw = F.linear(x2d, wDKV)                        # (bs, dkv+d_rope)
        kv_latent_raw = kv_raw[..., :dkv]                   # (bs, dkv)
        k_rope_raw   = kv_raw[..., dkv:]                    # (bs, d_rope)

        # -------------------------------------------------------------
        # 5) Insert new token into KV cache (store with RoPE already applied)
        # -------------------------------------------------------------
        cur_len = kv_cache.seq_len
        pos = cur_len                                        # position of this token

        # ---- RoPE rotation for the new key ----
        half = d_rope // 2
        cos_vec = _cached_cos[pos]                           # (d_rope,)
        sin_vec = _cached_sin[pos]                           # (d_rope,)
        cos_h = cos_vec[:half]
        sin_h = sin_vec[:half]

        k1 = k_rope_raw[..., :half]
        k2 = k_rope_raw[..., half:]
        k_rot = torch.empty_like(k_rope_raw)
        k_rot[..., :half] = k1 * cos_h - k2 * sin_h
        k_rot[..., half:] = k2 * cos_h + k1 * sin_h

        # ---- write latent + rotated key ----
        kv_cache.data[:, cur_len, :dkv] = kv_latent_raw
        kv_cache.data[:, cur_len, dkv:] = k_rot
        kv_cache.seq_len = cur_len + 1
        kv_len = kv_cache.seq_len

        # -------------------------------------------------------------
        # 6) Gather full cache (latent + rope‑rotated keys)
        # -------------------------------------------------------------
        kv_all   = kv_cache.data[:, :kv_len, :]               # (bs, kv_len, dkv+d_rope)
        kv_latent = kv_all[:, :, :dkv]                       # (bs, kv_len, dkv)
        k_rot_all = kv_all[:, :, dkv:]                       # (bs, kv_len, d_rope)

        # -------------------------------------------------------------
        # 7) Apply RoPE to the query (single token)
        # -------------------------------------------------------------
        cos_q = cos_vec.view(1, 1, d_rope)                   # (1,1,d_rope)
        sin_q = sin_vec.view(1, 1, d_rope)

        q1 = q[..., :half]
        q2 = q[..., half:]
        q_rot = torch.empty_like(q)
        q_rot[..., :half] = q1 * cos_q - q2 * sin_q
        q_rot[..., half:] = q2 * cos_q + q1 * sin_q
        q_rot = q_rot.unsqueeze(2)                           # (bs, nh, 1, d_rope)

        # -------------------------------------------------------------
        # 8) Prepare K and V tensors for SDPA
        # -------------------------------------------------------------
        # K: (bs, nh, kv_len, d_rope) – broadcast rope keys across heads
        k = k_rot_all.unsqueeze(1).expand(-1, nh, -1, -1)
        # V: (bs, nh, kv_len, dkv) – broadcast latent values across heads
        v = kv_latent.unsqueeze(1).expand(-1, nh, -1, -1)

        # -------------------------------------------------------------
        # 9) Scaled dot‑product attention (only rope dimensions)
        # -------------------------------------------------------------
        scale = 1.0 / math.sqrt(d_rope)
        latent_agg = F.scaled_dot_product_attention(
            q_rot, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )                                                   # (bs, nh, 1, dkv)
        latent_agg = latent_agg.squeeze(2)                  # (bs, nh, dkv)

        # -------------------------------------------------------------
        # 10) Project latent aggregation to value space (dv per head)
        # -------------------------------------------------------------
        # wV_T shape: (nh, dkv, dv)
        wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1)     # (nh, dkv, dv)
        y_head = torch.einsum('bhd, hdf -> bhf', latent_agg, wV_T)  # (bs, nh, dv)

        # -------------------------------------------------------------
        # 11) Final output projection
        # -------------------------------------------------------------
        y_head_flat = y_head.reshape(bs, nh * dv)           # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                      # (bs, dim)
        out = out.unsqueeze(1)                               # (bs, 1, dim)

        return out, kv_cache.data

    # -----------------------------------------------------------------
    #  General case – fall back to compiled implementation
    # -----------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,                               # (bs, 1, dim)
        kv_cache.data,                   # (bs, max_seq_len, dkv+d_rope)
        kv_cache.seq_len,                # current length (int)
        _cached_cos,                     # (max_seq_len, d_rope)
        _cached_sin,                     # (max_seq_len, d_rope)
        wDQ, wDKV, wUQ, wUKV, wO,
        nh, d_nope, d_rope, dkv, dv
    )

    # Update KVCache state
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data