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
# Helper: rotate‑half (identical to the Python implementation)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ----------------------------------------------------------------------
# Global RoPE lookup tables – lazily created on the first call
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
# Triton kernel – single‑pass stable‑softmax + per‑head value projection
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_vhead_onepass(
    # --------------------------------------------------------------
    # Pointers
    # --------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bfloat16
    K_ptr,                # (B, L, Dq)                 bfloat16
    V_ptr,                # (B, L, Dkv)                bfloat16
    wV_T_ptr,             # (H, Dkv, Dv)               bfloat16
    Y_ptr,                # (B, H, Dv)                 bfloat16   (per‑head output)

    # --------------------------------------------------------------
    # Strides
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,   # Q    (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,   # K    (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,   # V    (B, L, Dkv)

    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,  # wV_T (H, Dkv, Dv)

    stride_y_batch, stride_y_head, stride_y_dv,           # Y    (B, H, Dv)

    # --------------------------------------------------------------
    # Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    L: tl.constexpr,          # current KV length
    Dq: tl.constexpr,         # rope‑dim  (e.g. 64)
    Dkv: tl.constexpr,        # kv‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value dim (e.g. 128)
    scale: tl.constexpr,      # 1 / sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    One‑pass numerically‑stable softmax fused with the per‑head
    value‑projection (V → output).  The algorithm keeps a running
    max, normaliser and latent accumulator.
    """
    pid = tl.program_id(0)

    # --------------------------------------------------------------
    # 0️⃣ Identify batch and head‑tile this program works on
    # --------------------------------------------------------------
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                         # batch index
    tile = pid % num_head_tiles                       # which head‑tile
    head_start = tile * HEADS_PER_BLOCK               # first head in this tile

    # --------------------------------------------------------------
    # 1️⃣ Load Q‑vectors for the heads in this tile (once)
    # --------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_valid = head_start + head_range < H

    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)               # (HEADS_PER_BLOCK, Dq)

    # --------------------------------------------------------------
    # 2️⃣ Initialise running max, normaliser and latent accumulator
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)

    # --------------------------------------------------------------
    # 3️⃣ Scan over K / V blocks (single pass)
    # --------------------------------------------------------------
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

        # ----- Q·K dot‑product (scaled) ------------------------------------
        prod = tl.sum(q[:, None, :] * k_block[None, :, :], axis=2)   # (HEADS, BLOCK_K)
        prod = tl.cast(prod, tl.float32) * scale                    # apply 1/√Dq

        # ----- Update running max (stable softmax) -------------------------
        block_max = tl.max(prod, axis=1)                # (HEADS,)
        new_max   = tl.maximum(max_score, block_max)

        # factor to rescale the previous accumulators
        scale_prev = tl.exp(max_score - new_max)        # (HEADS,)

        # ----- Rescale previous state ---------------------------------------
        sum_exp   = sum_exp * scale_prev
        latent_acc = latent_acc * tl.broadcast_to(scale_prev[:, None],
                                                  [HEADS_PER_BLOCK, Dkv])

        # ----- exponentials of the current block -----------------------------
        exp_scores = tl.exp(prod - new_max[:, None])   # (HEADS, BLOCK_K)

        # ----- update normaliser ---------------------------------------------
        sum_exp = sum_exp + tl.sum(exp_scores, axis=1)   # (HEADS,)

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

            weighted = v_slice[None, :, :] * exp_scores[:, :, None]   # (HEADS, BLOCK_K, BLOCK_DV)
            latent_acc[:, start_d:start_d + BLOCK_DV] = \
                latent_acc[:, start_d:start_d + BLOCK_DV] + tl.sum(weighted, axis=1)

        # ----- store new max for the next iteration -------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # 4️⃣ Normalise the latent accumulator (still FP32)
    # ------------------------------------------------------------------
    latent = latent_acc / tl.broadcast_to(sum_exp[:, None], [HEADS_PER_BLOCK, Dkv])

    # ------------------------------------------------------------------
    # 5️⃣ Project latent → per‑head output (size Dv)
    # ------------------------------------------------------------------
    y_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)

    for start_d in range(0, Dkv, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV)
        d_mask = cur_d < Dkv

        # load wV_T slice: (HEADS, BLOCK_DV, Dv)
        offs_wV = (
            (head_start + head_range)[:, None, None] * stride_wV_T_head
            + cur_d[None, :, None] * stride_wV_T_lat
            + tl.arange(0, Dv)[None, None, :] * stride_wV_T_out
        )
        wV_block = tl.load(wV_T_ptr + offs_wV,
                           mask=head_valid[:, None] & d_mask[None, :],
                           other=0.0)               # (HEADS, BLOCK_DV, Dv)

        lat_slice = latent[:, start_d:start_d + BLOCK_DV]    # (HEADS, BLOCK_DV)

        y_head = y_head + tl.sum(wV_block * lat_slice[:, :, None], axis=1)

    # ------------------------------------------------------------------
    # 6️⃣ Store per‑head output (cast back to bfloat16)
    # ------------------------------------------------------------------
    offs_y = (
        b * stride_y_batch
        + (head_start + head_range)[:, None] * stride_y_head
        + tl.arange(0, Dv)[None, :] * stride_y_dv
    )
    tl.store(Y_ptr + offs_y,
             tl.cast(y_head, tl.bfloat16),
             mask=head_valid[:, None])
    # ------------------------------------------------------------------


# ----------------------------------------------------------------------
# Fast‑path – d_nope == 0 (the common configuration)
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
    Optimised path for the usual case where qk_nope_head_dim == 0.
    The heavy attention + per‑head projection work is done by a single
    Triton kernel that performs a numerically‑stable soft‑max in one
    pass.  Kernel launch parameters have been tuned for the H200:
        HEADS_PER_BLOCK = 16
        BLOCK_K         = 512 when seq_len > 4096 else 256
        BLOCK_DV        = 128
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv   = config.kv_lora_rank
    dv    = config.v_head_dim

    # --------------------------------------------------------------
    # 1️⃣ Down‑projection (two matmuls)
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                               # (B, dim)
    q_lora = F.linear(x2, wDQ)                      # (B, dq)
    kv_lora0 = F.linear(x2, wDKV)                   # (B, dkv + drope)

    # --------------------------------------------------------------
    # 2️⃣ KV‑cache update (store rotated key)
    # --------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora0[:, :dkv]                # (B, dkv)
    rope_raw_new   = kv_lora0[:, dkv:]               # (B, drope)

    # RoPE for the new key (position = cur_len)
    cos_k = cos_tbl[cur_len]                         # (drope,)
    sin_k = sin_tbl[cur_len]                         # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write latent + rotated key into the cache (contiguous layout)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 3️⃣ Up‑project Q and apply RoPE (single token)
    # --------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                    # (B, nh * drope)
    q_up = q_up.view(bs, nh, drope)                # (B, nh, drope)

    # RoPE for query (position = new_len‑1)
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                           # (drope,)
    sin_q = sin_tbl[q_pos]                           # (drope,)
    q_rope = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # --------------------------------------------------------------
    # 4️⃣ Gather full KV from cache
    # --------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]           # (B, L, dkv + drope)
    k_rope = kv_all[..., dkv:]                      # (B, L, drope)
    v_latent = kv_all[..., :dkv]                    # (B, L, dkv)

    # --------------------------------------------------------------
    # 5️⃣ Prepare per‑head value‑projection matrix (transpose)
    #    Shape: (nh, dkv, dv)
    # --------------------------------------------------------------
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()

    # --------------------------------------------------------------
    # 6️⃣ Triton kernel – fused attention + per‑head projection
    # --------------------------------------------------------------
    y_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # Strides (all contiguous)
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

    # ------------------------------------------------------------------
    # Choose launch‑time tiling parameters
    # ------------------------------------------------------------------
    # For very long sequences a larger BLOCK_K reduces the number of
    # kernel loop iterations.  512 works well when L >= 4096.
    BLOCK_K = 512 if new_len >= 4096 else 256
    HEADS_PER_BLOCK = 16               # 16*64 = 1024 threads < 2048 limit
    BLOCK_DV = 128                     # 512 / 128 = 4 loops over the latent dim

    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    scale = 1.0 / math.sqrt(drope)   # sqrt(Dq)

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
        # Triton launch config – more warps (1024 threads ⇒ 32 warps)
        num_warps=32,
        num_stages=4,
    )

    # ------------------------------------------------------------------
    # 7️⃣ Final output projection (WO) – fast GEMM from torch
    # ------------------------------------------------------------------
    y_head_flat = y_head.view(bs, nh * dv)          # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                 # (B, dim)
    out = out.unsqueeze(1)                          # (B, 1, dim)

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
        # reference implementation – unchanged (see description above)
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
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
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
    # Fast‑path – common case where d_nope == 0
    # --------------------------------------------------------------
    if d_nope == 0:
        out, kv_data = _fast_forward_multihead(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin,
        )
        # KVCache is mutated in‑place; keep the reference up‑to‑date
        kv_cache.data = kv_data
        return out, kv_cache.data

    # --------------------------------------------------------------
    # General case – fall back to compiled reference implementation
    # --------------------------------------------------------------
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