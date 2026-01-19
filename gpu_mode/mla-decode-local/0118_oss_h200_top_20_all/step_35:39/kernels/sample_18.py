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
# Helper utilities
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Exact half‑rotation used by RoPE (identical to reference)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ----------------------------------------------------------------------
# Global RoPE tables – lazily created on the first call
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Create (cos, sin) tables for rotary embeddings (bfloat16)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half,
                                      dtype=torch.float32,
                                      device=device) / half)).to(torch.bfloat16)  # (half,)
    pos = torch.arange(max_seq_len,
                       dtype=torch.int64,
                       device=device).unsqueeze_(1)          # (max_seq_len, 1)
    idx = pos * theta                                            # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                         # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel – fused stable‑softmax + per‑head value projection.
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_vhead_onepass(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bfloat16
    K_ptr,                # (B, L, Dq)                 bfloat16
    V_ptr,                # (B, L, Dkv)                bfloat16
    wV_T_ptr,             # (H, Dkv, Dv)               bfloat16
    Y_ptr,                # (B, H, Dv)                 bfloat16   (per‑head output)

    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,   # Q    (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,   # K    (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,   # V    (B, L, Dkv)

    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,  # wV_T (H, Dkv, Dv)

    stride_y_batch, stride_y_head, stride_y_dv,           # Y    (B, H, Dv)

    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    L: tl.constexpr,          # current KV length
    Dq: tl.constexpr,         # rope dim (e.g. 64)
    Dkv: tl.constexpr,        # kv‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value dim (e.g. 128)
    scale: tl.constexpr,      # 1/sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Single‑pass stable softmax with per‑head value projection.
    """
    pid = tl.program_id(0)

    # ---------------------------------------------------------------
    # 0️⃣ Identify batch and head‑tile this program works on
    # ---------------------------------------------------------------
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles
    tile = pid % num_head_tiles
    head_start = tile * HEADS_PER_BLOCK

    # ---------------------------------------------------------------
    # 1️⃣ Load Q‑vectors for the heads in this tile (once)
    # ---------------------------------------------------------------
    hs = tl.arange(0, HEADS_PER_BLOCK)                     # (HEADS_PER_BLOCK,)
    head_valid = head_start + hs < H                        # mask for the last tile

    offs_q = (
        b * stride_q_batch
        + (head_start + hs)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)               # (HEADS_PER_BLOCK, Dq)
    q = tl.cast(q, tl.float32)

    # ---------------------------------------------------------------
    # 2️⃣ Initialise running max, normaliser and latent accumulator
    # ---------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)

    # ---------------------------------------------------------------
    # 3️⃣ Scan over K / V blocks (single pass)
    # ---------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)          # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- load K block -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0)                 # (BLOCK_K, Dq)
        k_block = tl.cast(k_block, tl.float32)

        # ----- Q·K dot‑product (scaled) ------------------------------------
        # (HEADS_PER_BLOCK, Dq) @ (Dq, BLOCK_K)  -> (HEADS_PER_BLOCK, BLOCK_K)
        prod = tl.dot(q, tl.trans(k_block), out_dtype=tl.float32)
        prod = prod * scale

        # ----- Update running max (stable softmax) -------------------------
        block_max = tl.max(prod, axis=1)                # (HEADS_PER_BLOCK,)
        new_max   = tl.maximum(max_score, block_max)

        # factor to rescale the previous accumulators
        scale_prev = tl.exp(max_score - new_max)        # (HEADS_PER_BLOCK,)

        # ----- Rescale previous state ---------------------------------------
        sum_exp   = sum_exp * scale_prev
        latent_acc = latent_acc * tl.broadcast_to(scale_prev[:, None], [HEADS_PER_BLOCK, Dkv])

        # ----- exponentials of the current block -----------------------------
        exp_scores = tl.exp(prod - new_max[:, None])   # (HEADS_PER_BLOCK, BLOCK_K)

        # ----- update normaliser ---------------------------------------------
        sum_exp = sum_exp + tl.sum(exp_scores, axis=1)   # (HEADS_PER_BLOCK,)

        # ----- weighted accumulation of V (latent vectors) -------------------
        for start_d in range(0, Dkv, BLOCK_DV):
            cur_d = start_d + tl.arange(0, BLOCK_DV)
            d_mask = cur_d < Dkv

            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(V_ptr + offs_v,
                              mask=k_mask[:, None] & d_mask[None, :],
                              other=0.0)               # (BLOCK_K, BLOCK_DV)
            v_slice = tl.cast(v_slice, tl.float32)

            # exp_scores (HEADS, BLOCK_K) ×  v_slice (BLOCK_K, BLOCK_DV)
            # → (HEADS, BLOCK_DV)   – tensor‑core accelerated
            acc = tl.dot(exp_scores, v_slice, out_dtype=tl.float32)

            latent_acc[:, start_d:start_d + BLOCK_DV] = \
                latent_acc[:, start_d:start_d + BLOCK_DV] + acc

        # ----- store new max for the next iteration -------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # 4️⃣ Normalise the latent accumulator (still FP32)
    # ------------------------------------------------------------------
    latent = latent_acc / sum_exp[:, None]

    # ------------------------------------------------------------------
    # 5️⃣ Project latent → per‑head output (size Dv)
    # ------------------------------------------------------------------
    y_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)

    for start_d in range(0, Dkv, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV)
        d_mask = cur_d < Dkv

        # load wV_T slice: (HEADS, BLOCK_DV, Dv)
        offs_wV = (
            (head_start + hs)[:, None, None] * stride_wV_T_head
            + cur_d[None, :, None] * stride_wV_T_lat
            + tl.arange(0, Dv)[None, None, :] * stride_wV_T_out
        )
        wV_block = tl.load(wV_T_ptr + offs_wV,
                           mask=head_valid[:, None] & d_mask[None, :],
                           other=0.0)               # (HEADS, BLOCK_DV, Dv)
        wV_block = tl.cast(wV_block, tl.float32)

        lat_slice = latent[:, start_d:start_d + BLOCK_DV]    # (HEADS, BLOCK_DV)

        # (HEADS, BLOCK_DV) × (BLOCK_DV, Dv) → (HEADS, Dv)
        y_head = y_head + tl.dot(lat_slice, wV_block, out_dtype=tl.float32)

    # ------------------------------------------------------------------
    # 6️⃣ Store per‑head output (cast back to bfloat16)
    # ------------------------------------------------------------------
    offs_y = (
        b * stride_y_batch
        + (head_start + hs)[:, None] * stride_y_head
        + tl.arange(0, Dv)[None, :] * stride_y_dv
    )
    tl.store(Y_ptr + offs_y,
             tl.cast(y_head, tl.bfloat16),
             mask=head_valid[:, None])
    # ------------------------------------------------------------------


# ----------------------------------------------------------------------
# Fast‑path – d_nope == 0 (the common configuration)
# ----------------------------------------------------------------------
def _fast_forward_multihead_combined(
    config: Config,
    x: torch.Tensor,
    kv_cache: KVCache,
    wQ_combined: torch.Tensor,
    wDKV: torch.Tensor,
    wUKV: torch.Tensor,
    wO: torch.Tensor,
    cos_tbl: torch.Tensor,
    sin_tbl: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised fast‑path for the common case where qk_nope_head_dim == 0.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim

    # ------------------------------------------------------------------
    # 0️⃣ Input flattening
    # ------------------------------------------------------------------
    x2 = x.squeeze(1)                # (B, Dim)

    # ------------------------------------------------------------------
    # 1️⃣ KV‑cache update (store latent + rotated key)
    # ------------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_lora0 = F.linear(x2, wDKV)               # (B, dkv + drope)

    kv_latent_new = kv_lora0[:, :dkv]           # (B, dkv)
    rope_raw_new   = kv_lora0[:, dkv:]          # (B, drope)

    # RoPE for the new key (position = cur_len)
    cos_k = cos_tbl[cur_len]                    # (drope,)
    sin_k = sin_tbl[cur_len]                    # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len
    query_pos = new_len - 1

    # ------------------------------------------------------------------
    # 2️⃣ Query projection (combined down+up) + RoPE
    # ------------------------------------------------------------------
    q_all = F.linear(x2, wQ_combined)           # (B, nh * drope)
    q_all = q_all.view(bs, nh, drope)           # (B, nh, drope)

    cos_q = cos_tbl[query_pos]                  # (drope,)
    sin_q = sin_tbl[query_pos]                  # (drope,)
    q_rope = q_all * cos_q + _rotate_half(q_all) * sin_q   # (B, nh, drope)

    # ------------------------------------------------------------------
    # 3️⃣ Gather full KV from cache
    # ------------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]      # (B, L, dkv + drope)
    k_rope = kv_all[..., dkv:]                  # (B, L, drope)
    v_latent = kv_all[..., :dkv]                # (B, L, dkv)

    # ------------------------------------------------------------------
    # 4️⃣ Per‑head value‑projection matrix (transposed)
    # ------------------------------------------------------------------
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()  # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # 5️⃣ Allocate output buffer for per‑head values
    # ------------------------------------------------------------------
    y_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # ------------------------------------------------------------------
    # 6️⃣ Run the fused Triton kernel
    # ------------------------------------------------------------------
    # Strides
    stride_q_batch = q_rope.stride(0)
    stride_q_head  = q_rope.stride(1)
    stride_q_dim   = q_rope.stride(2)

    stride_k_batch = k_rope.stride(0)
    stride_k_len   = k_rope.stride(1)
    stride_k_dim   = k_rope.stride(2)

    stride_v_batch = v_latent.stride(0)
    stride_v_len   = v_latent.stride(1)
    stride_v_dim   = v_latent.stride(2)

    stride_wV_T_head = wV_T.stride(0)
    stride_wV_T_lat  = wV_T.stride(1)
    stride_wV_T_out  = wV_T.stride(2)

    stride_y_batch = y_head.stride(0)
    stride_y_head  = y_head.stride(1)
    stride_y_dv    = y_head.stride(2)

    # Tunable block sizes – chosen to improve occupancy & reduce loop count
    HEADS_PER_BLOCK = 64   # split heads into two tiles per batch (reduces register pressure)
    BLOCK_K = 256          # process 256 KV entries per iteration (fewer loops)
    BLOCK_DV = 128         # keep Dkv tiling as before

    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    scale = 1.0 / math.sqrt(drope)

    _triton_attn_vhead_onepass[grid](
        # pointers
        q_rope, k_rope, v_latent,
        wV_T, y_head,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len,   stride_k_dim,
        stride_v_batch, stride_v_len,   stride_v_dim,
        stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
        stride_y_batch, stride_y_head, stride_y_dv,
        # compile‑time constants
        bs, nh, new_len, drope, dkv, dv,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV,
        # launch config
        num_warps=8, num_stages=3,
    )

    # ------------------------------------------------------------------
    # 7️⃣ Final output projection (WO) – cuBLAS GEMM
    # ------------------------------------------------------------------
    y_head_flat = y_head.view(bs, nh * dv)          # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                 # (B, Dim)
    out = out.unsqueeze(1)                          # (B, 1, Dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Compiled fallback for the general case (d_nope > 0)
# ----------------------------------------------------------------------
_compiled_forward = None
def _build_compiled_forward():
    """Compiled fallback used when d_nope > 0."""
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
        # unchanged reference implementation (still torch‑compiled)
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
    Entry point expected by the benchmark harness.
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Extract scalar config values (plain python ints)
    # ------------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    d = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    # Build / fetch RoPE tables (cached globally)
    # ------------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # ------------------------------------------------------------------
    # Fast‑path – d_nope == 0 (the common configuration)
    # ------------------------------------------------------------------
    if d_nope == 0:
        # combine the two Q projections once (cached)
        if not hasattr(config, "combined_Q_proj_weight"):
            # compute in FP32 for better numerical stability, then cast back
            with torch.no_grad():
                config.combined_Q_proj_weight = torch.mm(
                    wUQ.float(),
                    wDQ.float()
                ).to(torch.bfloat16)
        wQ_combined = config.combined_Q_proj_weight

        out, kv_data = _fast_forward_multihead_combined(
            config, x, kv_cache,
            wQ_combined, wDKV, wUKV, wO,
            _cached_cos, _cached_sin,
        )
        # KVCache is mutated in‑place; keep reference up‑to‑date
        kv_cache.data = kv_data
        return out, kv_cache.data

    # ------------------------------------------------------------------
    # General case – fall back to compiled reference implementation
    # ------------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
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
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    # The compiled fallback already returns shape [B, 1, Dim].
    return out, kv_cache.data