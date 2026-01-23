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
# Global caches (shared across kernel calls) – never re‑allocated
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None          # (max_seq_len, rope_dim)    bfloat16
_cached_sin: torch.Tensor = None          # (max_seq_len, rope_dim)    bfloat16
_cached_wV_T: torch.Tensor = None         # (n_heads, kv_lora_rank, v_head_dim) bfloat16
_cached_sumV: torch.Tensor = None         # (batch, n_heads, kv_lora_rank)  fp32
# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half (used by RoPE)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Pre‑compute cosine / sine tables for rope (bfloat16)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half,
                                        dtype=torch.float32,
                                        device=device) / half)).to(torch.bfloat16)
    pos = torch.arange(max_seq_len,
                       dtype=torch.int64,
                       device=device).unsqueeze_(1)           # (max_seq_len, 1)
    idx = pos * theta                                          # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                        # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)

# ----------------------------------------------------------------------
# Triton kernel – fused stable‑softmax + aggregation of V‑latents
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_fused_kernel(
    # --------------------------------------------------------------
    # Pointers
    # --------------------------------------------------------------
    Q_ptr,               # (B, H, Dq)                     bf16   <-- also output buffer
    K_ptr,               # (B, L, Dq)                     bf16
    V_ptr,               # (B, L, Dkv)                    bf16
    wV_T_ptr,            # (H, Dkv, Dv)                   bf16
    sumV_ptr,            # (B, H, Dkv)                    fp32    (temporarily stores Σ a·V)
    # --------------------------------------------------------------
    # Strides
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len,  stride_k_dim,
    stride_v_batch, stride_v_len,  stride_v_dim,
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
    stride_sumV_batch, stride_sumV_head, stride_sumV_dim,
    # --------------------------------------------------------------
    # Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    Dq: tl.constexpr,         # rope dim (e.g. 64)
    Dkv: tl.constexpr,        # kv‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value dim (e.g. 128)
    scale: tl.constexpr,      # 1/sqrt(Dq)

    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    # --------------------------------------------------------------
    # Runtime argument
    # --------------------------------------------------------------
    L: tl.int32,               # current KV length (dynamic)
):
    """
    Stable‑softmax attention fused with a *single* pass over the weight
    matrix.   The algorithm proceeds in two phases:

    1️⃣  Scan the KV cache → compute stable softmax, accumulate the
        (unnormalised) Σ a·V into a fp32 buffer `sumV`  (no weight load).

    2️⃣  After the scan finishes, normalise `sumV` and multiply with the
        per‑head output projection matrix `wV_T`. The projection is performed
        block‑wise over the latent dimension.
    """
    pid = tl.program_id(0)                         # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # head‑tile index inside batch
    head_start = tile * HEADS_PER_BLOCK              # first head handled by this program

    # --------------------------------------------------------------
    # 0️⃣  Helpers – head mask & valid‑head predicate
    # --------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_valid = head_start + head_range < H

    # --------------------------------------------------------------
    # 1️⃣  Load queries for this tile (shape: HEADS_PER_BLOCK × Dq)
    # --------------------------------------------------------------
    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)                     # (HEADS_PER_BLOCK, Dq) bf16

    # --------------------------------------------------------------
    # 2️⃣  Initialise stable‑softmax state + final output accumulator
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)
    # accumulator for the *projected* head output (float32)
    v_head_acc = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)

    # --------------------------------------------------------------
    # 3️⃣  Main KV scan
    # --------------------------------------------------------------
    for start_k in tl.range(0, L, BLOCK_K, num_stages=4):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)       # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- load K block -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0,
                          cache_modifier='CA')               # (BLOCK_K, Dq) bf16

        # ----- q·k ---------------------------------------------------------
        prod = tl.dot(q, tl.permute(k_block, (1, 0)))           # (HEADS_PER_BLOCK, BLOCK_K) bf16
        score_f32 = tl.cast(prod, tl.float32) * scale

        # ----- stable‑softmax update ---------------------------------------
        block_max = tl.max(score_f32, axis=1)                    # (HEADS_PER_BLOCK,)
        new_max   = tl.maximum(max_score, block_max)              # (HEADS_PER_BLOCK,)

        # rescale previous aggregates when max grows
        exp_factor = tl.exp(max_score - new_max)                  # (HEADS_PER_BLOCK,)
        sum_exp = sum_exp * exp_factor

        # exp_score = exp(score - new_max)
        exp_score = tl.exp(score_f32 - new_max[:, None])           # (HEADS_PER_BLOCK, BLOCK_K)
        sum_exp = sum_exp + tl.sum(exp_score, axis=1)             # (HEADS_PER_BLOCK,)

        # ----- accumulate Σ a·V (latent) ------------------------------------
        for start_d in tl.range(0, Dkv, BLOCK_DV):
            cur_d = start_d + tl.arange(0, BLOCK_DV, tl.int32)   # (BLOCK_DV,)
            d_mask = cur_d < Dkv

            # Load V_latent slice (BLOCK_K, BLOCK_DV)
            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(V_ptr + offs_v,
                              mask=k_mask[:, None] & d_mask[None, :],
                              other=0.0,
                              cache_modifier='CA')                # (BLOCK_K, BLOCK_DV) bf16
            v_fp32 = tl.cast(v_slice, tl.float32)                # (BLOCK_K, BLOCK_DV)

            # Σ a·V contribution for this latent block
            X = tl.dot(exp_score, v_fp32)                         # (HEADS_PER_BLOCK, BLOCK_DV)

            # Load the current Σ a·V from the global buffer, apply scaling, add X
            offs_sumV = (
                b * stride_sumV_batch
                + (head_start + head_range)[:, None] * stride_sumV_head
                + cur_d[None, :] * stride_sumV_dim
            )
            sumV_block = tl.load(sumV_ptr + offs_sumV,
                                 mask=head_valid[:, None] & d_mask[None, :],
                                 other=0.0)                         # (HEADS_PER_BLOCK, BLOCK_DV) fp32

            # scale previously accumulated Σ a·V (same factor as softmax denominator)
            sumV_block = sumV_block * exp_factor[:, None]

            # add the new contribution
            sumV_block = sumV_block + X

            # write back
            tl.store(sumV_ptr + offs_sumV,
                     sumV_block,
                     mask=head_valid[:, None] & d_mask[None, :])

        # update the running max for the next block
        max_score = new_max

    # ------------------------------------------------------------------
    # 4️⃣  Normalise Σ a·V and apply the per‑head output projection
    # ------------------------------------------------------------------
    for start_d in tl.range(0, Dkv, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV, tl.int32)   # (BLOCK_DV,)
        d_mask = cur_d < Dkv

        # Load Σ a·V slice
        offs_sumV = (
            b * stride_sumV_batch
            + (head_start + head_range)[:, None] * stride_sumV_head
            + cur_d[None, :] * stride_sumV_dim
        )
        sumV_block = tl.load(sumV_ptr + offs_sumV,
                             mask=head_valid[:, None] & d_mask[None, :],
                             other=0.0)                     # (HEADS_PER_BLOCK, BLOCK_DV) fp32

        # Normalise by the softmax denominator (per‑head)
        sumV_norm = sumV_block / sum_exp[:, None]            # (HEADS_PER_BLOCK, BLOCK_DV)

        # Load weight slice wV_T (BLOCK_DV, Dv)
        offs_wV = (
            (head_start + head_range)[:, None, None] * stride_wV_T_head
            + cur_d[None, :, None] * stride_wV_T_lat
            + tl.arange(0, Dv, tl.int32)[None, None, :] * stride_wV_T_out
        )
        wV_block = tl.load(wV_T_ptr + offs_wV,
                           mask=head_valid[:, None] & d_mask[None, :],
                           other=0.0)                     # (HEADS_PER_BLOCK, BLOCK_DV, Dv) bf16
        wV_block_f32 = tl.cast(wV_block, tl.float32)          # (HEADS_PER_BLOCK, BLOCK_DV, Dv)

        # contribution = Σ a·V_norm @ wV_T   → (HEADS_PER_BLOCK, Dv)
        contrib = tl.sum(sumV_norm[:, :, None] * wV_block_f32, axis=1)   # (HEADS_PER_BLOCK, Dv)

        # Accumulate
        v_head_acc = v_head_acc + contrib

    # ------------------------------------------------------------------
    # 5️⃣  Store final per‑head projected values (bf16) – overwrite Q buffer
    # ------------------------------------------------------------------
    offs_out = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dv)[None, :] * stride_q_dim
    )
    tl.store(Q_ptr + offs_out,
             tl.cast(v_head_acc, tl.bfloat16),
             mask=head_valid[:, None])

# ----------------------------------------------------------------------
# Fast‑path (qk_nope_head_dim == 0) – uses the fused kernel above
# ----------------------------------------------------------------------
def _fast_forward_multihead(
    config: Config,
    x: torch.Tensor,
    kv_cache: KVCache,
    wDQ: torch.Tensor,
    wDKV: torch.Tensor,
    wUQ: torch.Tensor,
    wUKV: torch.Tensor,
    wO: torch.Tensor,
    cos_tbl: torch.Tensor,
    sin_tbl: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward for the common case `qk_nope_head_dim == 0`.
    """
    bs, _, dim = x.shape
    nh   = config.n_heads
    drope = config.qk_rope_head_dim
    dkv  = config.kv_lora_rank
    dv   = config.v_head_dim

    # ------------------------------------------------------------------
    # 0️⃣  KV down‑projection + KV‑cache update (identical to reference)
    # ------------------------------------------------------------------
    x2 = x.squeeze(1)               # (B, Dim)
    kv_lora = F.linear(x2, wDKV)    # (B, dkv + drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora[:, :dkv]                 # (B, dkv)
    rope_raw_new  = kv_lora[:, dkv:]                 # (B, drope)

    # RoPE for the new key token
    cos_k = cos_tbl[cur_len]                         # (drope,)
    sin_k = sin_tbl[cur_len]                         # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write into cache (latent part + rope‑rotated part)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # ------------------------------------------------------------------
    # 1️⃣  Q down‑projection + Q up‑projection
    # ------------------------------------------------------------------
    q_lora = F.linear(x2, wDQ)                       # (B, q_lora_rank)
    q = F.linear(q_lora, wUQ)                        # (B, nh * drope)
    q = q.view(bs, nh, drope)                        # (B, nh, drope)

    # ------------------------------------------------------------------
    # 2️⃣  RoPE for queries (single token)
    # ------------------------------------------------------------------
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                           # (drope,)
    sin_q = sin_tbl[q_pos]                           # (drope,)
    q = q * cos_q + _rotate_half(q) * sin_q          # (B, nh, drope)

    # ------------------------------------------------------------------
    # 3️⃣  Prepare weight – cache transposed wV_T (dkv → dv) once
    # ------------------------------------------------------------------
    global _cached_wV_T
    if _cached_wV_T is None:
        # wUKV shape: ((d_nope+dv)*nh, dkv) → (nh, dv, dkv) after reshaping & transpose
        _cached_wV_T = wUKV.view(nh, dv, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # 4️⃣  Allocate (or reuse) Σ a·V buffer (fp32) – zero it in‑place
    # ------------------------------------------------------------------
    global _cached_sumV
    if _cached_sumV is None or _cached_sumV.shape != (bs, nh, dkv):
        _cached_sumV = torch.empty((bs, nh, dkv), dtype=torch.float32, device=x.device)
    sumV = _cached_sumV
    sumV.zero_()

    # ------------------------------------------------------------------
    # 5️⃣  Launch the fused Triton kernel
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(drope)   # Dq == drope

    # Triton grid: one program per (batch, head‑tile)
    grid = lambda meta: (bs * triton.cdiv(nh, meta["HEADS_PER_BLOCK"]),)

    _triton_attn_fused_kernel[grid](
        # pointers
        q,                     # Q (also output)
        kv_cache.data,         # K – already contains rope‑rotated keys
        kv_cache.data,         # V – latent part lives in the same cache tensor (first dkv columns)
        _cached_wV_T,
        sumV,
        # strides
        q.stride(0), q.stride(1), q.stride(2),                 # Q strides
        kv_cache.data.stride(0), kv_cache.data.stride(1), kv_cache.data.stride(2),  # K strides
        kv_cache.data.stride(0), kv_cache.data.stride(1), kv_cache.data.stride(2),  # V strides
        _cached_wV_T.stride(0), _cached_wV_T.stride(1), _cached_wV_T.stride(2),   # wV_T strides
        sumV.stride(0), sumV.stride(1), sumV.stride(2),       # sumV strides
        # compile‑time args
        bs, nh, drope, dkv, dv, scale,
        64,            # HEADS_PER_BLOCK  (kept at 64 – safe on registers)
        1024,          # BLOCK_K
        128,           # BLOCK_DV – larger block reduces inner loop count
        # runtime arg
        new_len
    )

    # ------------------------------------------------------------------
    # 6️⃣  Final linear projection back to model dimension
    # ------------------------------------------------------------------
    y_head = q.view(bs, nh * dv)                     # (B, nh*dv)
    out = F.linear(y_head, wO)                       # (B, dim)
    out = out.unsqueeze(1)                           # (B, 1, dim)

    return out, kv_cache.data

# ----------------------------------------------------------------------
# Compiled fallback (qk_nope_head_dim > 0) – unchanged reference path
# ----------------------------------------------------------------------
_compiled_forward = None
def _build_compiled_forward():
    """Compiled fallback used when `qk_nope_head_dim > 0`."""
    import torch.nn.functional as F
    def _inner(
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
    ):
        # reference implementation – unchanged
        q_lora = F.linear(x, wDQ)               # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)            # (bs, 1, dkv + d_rope)

        new_len = cur_len + kv_lora0.shape[1]
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]       # (bs, kv_len, dkv + d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        q_up = F.linear(q_lora.squeeze(1), wUQ)               # (bs, nh*d_nope+d_rope)
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)    # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)  # kv_nope unused
        kv_latent = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk', q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                         dtype=torch.bfloat16,
                                         device=x.device)

        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, nh, kv_len, d_rope)

        scores_rope = torch.matmul(q_rope_rot,
                                   k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

        bh = x.shape[0] * nh
        scores_flat = scores.reshape(bh, -1)
        attn = F.softmax(scores_flat, dim=-1).to(torch.bfloat16).view(x.shape[0], nh, -1)

        latent_agg = torch.matmul(attn, kv_latent)               # (bs, nh, dkv)

        y_head = torch.einsum('bhd, hdf -> bhf', latent_agg, wV_T)   # (bs, nh, dv)

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
# Main entry point (custom_kernel)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expected entry point for the benchmark harness.
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Extract scalar config values (plain python ints)
    # --------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dim = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # --------------------------------------------------------------
    # Weight tensors (already on the correct device & dtype)
    # --------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # Build / fetch RoPE tables (cached globally)
    # --------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # --------------------------------------------------------------
    # Fast‑path – d_nope == 0 (the most common configuration)
    # --------------------------------------------------------------
    if d_nope == 0:
        out, new_kv = _fast_forward_multihead(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin,
        )
        # kv_cache is already updated inside the fast‑path function
        return out, new_kv

    # --------------------------------------------------------------
    # General case – fallback to compiled reference implementation
    # --------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,                               # (bs, 1, dim)
        kv_cache.data,                   # (bs, max_seq_len, dkv+drope)
        kv_cache.seq_len,                # current cache length
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
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data