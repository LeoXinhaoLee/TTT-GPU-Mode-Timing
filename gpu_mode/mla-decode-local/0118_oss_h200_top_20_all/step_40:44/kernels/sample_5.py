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
# Helper utilities (rotate‑half and cached RoPE tables)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Exactly the same as the reference implementation."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Build (cos, sin) tables for rotary embeddings (bfloat16).
    The tables are small (max_seq_len × dim) and are reused across calls.
    """
    half = dim // 2
    # theta = 10000 ** (-i / half)   –‑> float32 → bfloat16
    theta = (10000.0 ** (-torch.arange(half,
                                        dtype=torch.float32,
                                        device=device) / half)).to(torch.bfloat16)   # (half,)
    pos = torch.arange(max_seq_len,
                       dtype=torch.int64,
                       device=device).unsqueeze_(1)          # (max_seq_len, 1)
    idx = pos * theta                                            # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                         # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel – efficient single‑pass stable‑softmax attention
# ----------------------------------------------------------------------
@triton.jit
def _triton_latent_onepass(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bfloat16
    K_ptr,                # (B, L, Dq)                 bfloat16
    V_ptr,                # (B, L, Dkv)                bfloat16
    LAT_ptr,              # (B, H, Dkv)                bfloat16   (latent output)

    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,   # Q    (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,   # K    (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,   # V    (B, L, Dkv)

    stride_lat_batch, stride_lat_head, stride_lat_dim,  # LAT (B, H, Dkv)

    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    L: tl.constexpr,          # current KV length
    Dq: tl.constexpr,         # rope head dimension (e.g. 64)
    Dkv: tl.constexpr,        # KV‑lora rank (e.g. 512)
    scale: tl.constexpr,      # 1/sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Stable‑softmax attention (latent = softmax(Q·Kᵀ) @ V)
    Vectorised over ``HEADS_PER_BLOCK`` heads per program.
    """
    pid = tl.program_id(0)

    # ---------------------------------------------------------------
    # 0️⃣  Identify batch and heads tile served by this program
    # ---------------------------------------------------------------
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles
    tile = pid % num_head_tiles
    head_start = tile * HEADS_PER_BLOCK

    # ---------------------------------------------------------------
    # 1️⃣ Load Q‑vectors for the heads in this tile (once)
    # ---------------------------------------------------------------
    hs = tl.arange(0, HEADS_PER_BLOCK)                     # (HEADS_PER_BLOCK,)
    head_valid = head_start + hs < H                        # mask for trailing heads

    offs_q = (
        b * stride_q_batch
        + (head_start + hs)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)               # (HEADS_PER_BLOCK, Dq)   – bfloat16
    # No cast – tl.dot will up‑cast to float32 automatically
    # ---------------------------------------------------------------
    # 2️⃣ Initialise stable‑softmax state
    # ---------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)

    # ---------------------------------------------------------------
    # 3️⃣ Scan over K/V blocks (single pass)
    # ---------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)          # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- Load K block -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0)                 # (BLOCK_K, Dq)   – bfloat16

        # ----- Q·K dot‑product (scaled) ------------------------------------
        prod = tl.dot(q, tl.trans(k_block), out_dtype=tl.float32)   # (HEADS_PER_BLOCK, BLOCK_K)
        prod = prod * scale

        # ----- Stable softmax update ---------------------------------------
        block_max = tl.max(prod, axis=1)                # (HEADS_PER_BLOCK,)
        new_max   = tl.maximum(max_score, block_max)

        # factor to rescale the previous accumulators
        scale_prev = tl.exp(max_score - new_max)        # (HEADS_PER_BLOCK,)

        # ----- Rescale previous state ---------------------------------------
        sum_exp   = sum_exp * scale_prev
        latent_acc = latent_acc * tl.broadcast_to(scale_prev[:, None],
                                                  [HEADS_PER_BLOCK, Dkv])

        # ----- Exponentials of the current block -----------------------------
        exp_scores = tl.exp(prod - new_max[:, None])   # (HEADS_PER_BLOCK, BLOCK_K)

        # ----- Update normaliser ---------------------------------------------
        sum_exp = sum_exp + tl.sum(exp_scores, axis=1)   # (HEADS_PER_BLOCK,)

        # ----- Weighted accumulation of V (latent vectors) -------------------
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
                              other=0.0)               # (BLOCK_K, BLOCK_DV) – bfloat16
            # Accumulate: (HEADS, BLOCK_K) × (BLOCK_K, BLOCK_DV) → (HEADS, BLOCK_DV)
            acc = tl.dot(exp_scores, v_slice, out_dtype=tl.float32)
            latent_acc[:, start_d:start_d + BLOCK_DV] = \
                latent_acc[:, start_d:start_d + BLOCK_DV] + acc

        # ----- Prepare for next iteration ------------------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # 4️⃣ Normalise the latent accumulator (still FP32)
    # ------------------------------------------------------------------
    latent = latent_acc / sum_exp[:, None]

    # ------------------------------------------------------------------
    # 5️⃣ Write latent (cast back to bfloat16)
    # ------------------------------------------------------------------
    offs_lat = (
        b * stride_lat_batch
        + (head_start + hs)[:, None] * stride_lat_head
        + tl.arange(0, Dkv)[None, :] * stride_lat_dim
    )
    tl.store(LAT_ptr + offs_lat,
             tl.cast(latent, tl.bfloat16),
             mask=head_valid[:, None])


# ----------------------------------------------------------------------
# Fast‑path (d_nope == 0) – fully fused inference
# ----------------------------------------------------------------------
def _fast_forward_multihead_fused(
    config: Config,
    x: torch.Tensor,
    kv_cache: KVCache,
    wDQ: torch.Tensor,            # (dq, dim)
    wUQ: torch.Tensor,            # (nh*rope_dim, dq)
    wDKV: torch.Tensor,           # (dkv+rope_dim, dim)
    wUKV: torch.Tensor,           # ((dv)*nh, dkv)  –‑> (nh, dv, dkv) after view
    wO: torch.Tensor,             # (dim, nh*dv)
    cos_tbl: torch.Tensor,
    sin_tbl: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The common scenario with **no‑pe** (qk_nope_head_dim == 0).

    1️⃣  Low‑rank “LoRA”‑style projections are performed in two
        GEMMs (down‑projection + up‑projection) – this halves the weight‑
        traffic compared with the monolithic GEMM used in the reference.
    2️⃣  RoPE is applied only to the *new* key (cached) and to the *new*
        query.
    3️⃣  Scaled‑dot‑product attention is executed by a custom Triton kernel
        that implements a *stable* soft‑max in a single pass.
    4️⃣  The value‑projection + output‑projection are carried out with
        two highly‑optimised linear layers (torch.nn.functional.linear)
        instead of a three‑operand ``einsum``.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim

    # ------------------------------------------------------------------
    # 0️⃣  Down‑projection + up‑projection for Q (two GEMMs)
    # ------------------------------------------------------------------
    x2 = x.squeeze(1)                                      # (bs, dim)
    q_lora = F.linear(x2, wDQ)                              # (bs, dq)
    q_all  = F.linear(q_lora, wUQ)                          # (bs, nh * drope)
    q_all  = q_all.view(bs, nh, drope)                     # (bs, nh, drope)

    # ------------------------------------------------------------------
    # 1️⃣  KV down‑projection (single GEMM)
    # ------------------------------------------------------------------
    kv_proj = F.linear(x2, wDKV)                            # (bs, dkv + drope)
    kv_latent_new = kv_proj[:, :dkv]                        # (bs, dkv)
    rope_raw_new  = kv_proj[:, dkv:]                        # (bs, drope)

    # ------------------------------------------------------------------
    # 2️⃣  Write new KV entry into the cache (latent part + rotated key)
    # ------------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    # Rope rotation for the *new* key (position = cur_len)
    cos_k = cos_tbl[cur_len]               # (drope,)
    sin_k = sin_tbl[cur_len]               # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k   # (bs, drope)

    # In‑place update of the cache
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len
    query_pos = new_len - 1

    # ------------------------------------------------------------------
    # 3️⃣  RoPE for the *new* query (position = query_pos)
    # ------------------------------------------------------------------
    cos_q = cos_tbl[query_pos]            # (drope,)
    sin_q = sin_tbl[query_pos]            # (drope,)
    q = q_all * cos_q + _rotate_half(q_all) * sin_q   # (bs, nh, drope)

    # ------------------------------------------------------------------
    # 4️⃣  Gather full KV cache (already rotated keys)
    # ------------------------------------------------------------------
    kv_cache_all = kv_cache.data[:, :new_len, :]            # (bs, L, dkv + drope)
    k_rope   = kv_cache_all[..., dkv:]                     # (bs, L, drope) – already rotated
    v_latent = kv_cache_all[..., :dkv]                     # (bs, L, dkv)

    # ------------------------------------------------------------------
    # 5️⃣  Prepare tensors for the Triton kernel
    # ------------------------------------------------------------------
    # Q needs a length‑1 dimension for the kernel: (bs, nh, 1, drope)
    q_kernel = q.unsqueeze(2)                              # (bs, nh, 1, drope)

    # K must be broadcast to every head: (bs, nh, L, drope)
    k_kernel = k_rope[:, None, :, :].expand(-1, nh, -1, -1)   # (bs, nh, L, drope)

    # Allocate latent buffer (FP16) – will be filled by the Triton kernel
    latent = torch.empty((bs, nh, dkv), dtype=torch.bfloat16, device=x.device)

    # Strides for the kernel
    stride_q_batch = q_kernel.stride(0)
    stride_q_head  = q_kernel.stride(1)
    stride_q_dim   = q_kernel.stride(3)      # dim is last after the length‑1 axis

    stride_k_batch = k_kernel.stride(0)
    stride_k_len   = k_kernel.stride(2)
    stride_k_dim   = k_kernel.stride(3)

    stride_v_batch = v_latent.stride(0)
    stride_v_len   = v_latent.stride(1)
    stride_v_dim   = v_latent.stride(2)

    stride_lat_batch = latent.stride(0)
    stride_lat_head  = latent.stride(1)
    stride_lat_dim   = latent.stride(2)

    # Tuning parameters – chosen to keep register pressure low while
    # still delivering high occupancy on H200.
    HEADS_PER_BLOCK = 16               # 16 heads per program → good occupancy
    BLOCK_K         = 256              # KV‑tile (covers many tokens per iteration)
    BLOCK_DV        = 128              # DKV‑tile (512 → 4 iterations)

    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    scale = 1.0 / math.sqrt(drope)   #  = 1/√(drope)  because d_nope == 0

    _triton_latent_onepass[grid](
        # pointers
        q_kernel, k_kernel, v_latent, latent,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len,   stride_k_dim,
        stride_v_batch, stride_v_len,   stride_v_dim,
        stride_lat_batch, stride_lat_head, stride_lat_dim,
        # compile‑time constants
        bs, nh, new_len, drope, dkv, scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV,
        # launch config
        num_warps=8, num_stages=3,
    )

    # ------------------------------------------------------------------
    # 6️⃣  Value‑projection (per‑head) + final linear projection
    # ------------------------------------------------------------------
    # wV_T : (nh, dkv, dv)
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)

    # Multiply latent (bs, nh, dkv) × wV_T (nh, dkv, dv) → (bs, nh, dv)
    y_head = torch.einsum('bhd,hdv->bhv', latent, wV_T)          # (bs, nh, dv)

    # Collapse heads and project to model dimension
    y_head_flat = y_head.reshape(bs, nh * dv)                     # (bs, nh*dv)
    out = F.linear(y_head_flat, wO.t())                           # (bs, dim)

    out = out.unsqueeze(1)   # (bs, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Fallback (general case – d_nope > 0)
# ----------------------------------------------------------------------
_compiled_forward = None
def _build_compiled_forward():
    """Compiled fallback used when d_nope > 0 (unchanged reference logic)."""
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
        # reference implementation (unchanged)
        q_lora = F.linear(x, wDQ)                                   # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)                                # (bs, 1, dkv + d_rope)

        new_len = cur_len + kv_lora0.shape[1]
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]                           # (bs, kv_len, dkv + d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        q_up = F.linear(q_lora.squeeze(1), wUQ)                     # (bs, nh*d_nope+d_rope)
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)          # (bs, nh, d_nope+d_rope)
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
    Expected signature for the benchmark harness.
    Returns:
        - output   : Tensor of shape [batch, seq_len, dim] (seq_len == 1)
        - kv_cache : The updated KV cache data tensor
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Extract scalar config values (plain python ints)
    # --------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    d_nope = config.qk_nope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim

    # --------------------------------------------------------------
    # Extract weight tensors (all still in BF16)
    # --------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wUQ  = config.Q_proj_up_weight            # (nh*drope, dq)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUKV = config.KV_proj_up_weight           # ((dv)*nh, dkv)   –‑> view later
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
    # Fast‑path – d_nope == 0 (the common configuration)
    # --------------------------------------------------------------
    if d_nope == 0:
        out, kv_data = _fast_forward_multihead_fused(
            config, x, kv_cache,
            wDQ, wUQ, wDKV, wUKV, wO,
            _cached_cos, _cached_sin,
        )
        # KVCache is mutated in‑place; keep reference up‑to‑date
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