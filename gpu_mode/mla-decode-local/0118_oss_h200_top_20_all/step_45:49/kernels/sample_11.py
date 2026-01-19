### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config   # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###

# -------------------------------------------------------------------------
#  Helper – RoPE (identical to the reference implementation)
# -------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the last dimension by half (identical to reference)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# -------------------------------------------------------------------------
#  Global cached sinusoid tables (created lazily)
# -------------------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Create (cos, sin) tables for Rotary embeddings (bfloat16)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half,
                                       dtype=torch.float32,
                                       device=device) / half)).to(torch.bfloat16)   # (half,)
    pos = torch.arange(max_seq_len,
                       dtype=torch.int64,
                       device=device).unsqueeze_(1)                        # (max_seq_len, 1)
    idx = pos * theta                                                         # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                                      # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# -------------------------------------------------------------------------
#  Triton‑fused attention + per‑head value projection + final WO projection
# -------------------------------------------------------------------------
@triton.jit
def _triton_latent_fused(
    # ---------------------------------------------------------------
    #  Pointers
    # ---------------------------------------------------------------
    Q_ptr,          # (B, H, Dq)               bfloat16
    K_ptr,          # (B, L, Dq)               bfloat16
    V_ptr,          # (B, L, Dkv)              bfloat16
    WV_ptr,         # (H, Dkv, Dv)             bfloat16
    WO_ptr,         # (dim, H*Dv)              bfloat16
    OUT_ptr,        # (B, dim)                 bfloat16

    # ---------------------------------------------------------------
    #  Strides
    # ---------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,        # Q    (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,        # K    (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,        # V    (B, L, Dkv)

    stride_wv_head, stride_wv_in, stride_wv_out,        # WV   (H, Dkv, Dv)
    stride_wo_batch, stride_wo_head, stride_wo_dim,    # WO   (dim, H*Dv)

    stride_out_batch, stride_out_dim,                   # OUT  (B, dim)

    # ---------------------------------------------------------------
    #  Compile‑time constants
    # ---------------------------------------------------------------
    B:           tl.constexpr,   # batch size
    H:           tl.constexpr,   # total heads
    L:           tl.constexpr,   # current KV length
    Dq:          tl.constexpr,   # rope head dim
    Dkv:         tl.constexpr,   # latent dim (kv‑lora rank)
    Dv:          tl.constexpr,   # per‑head value dim
    dim:         tl.constexpr,   # model dimension (output dim)
    scale:       tl.constexpr,   # 1 / sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K:         tl.constexpr,
    BLOCK_DV:        tl.constexpr,
):
    """
    * One program processes one (batch, head‑tile) pair.
    * Performs a stable‑softmax scan over the KV cache, accumulates the latent
      representation Σ attn·V, directly projects it to per‑head values (WV) and
      finally folds the per‑head output into the final model output via WO.
    * All work is done in FP32 (except the final store) – this triggers Tensor
      Cores on H200 and gives the best throughput for bf16 inputs.
    """
    pid = tl.program_id(0)

    # ---------------------------------------------------------------
    #  Decode program‑id → batch index + head‑tile start
    # ---------------------------------------------------------------
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK      # tiles per batch
    b   = pid // num_head_tiles                                        # batch index
    tile = pid % num_head_tiles                                         # tile index inside batch
    head_start = tile * HEADS_PER_BLOCK                                 # first head handled by this program

    # ---------------------------------------------------------------
    #  Masks for the (possibly) last partially‑filled head tile
    # ---------------------------------------------------------------
    hs = tl.arange(0, HEADS_PER_BLOCK)                # (HEADS_PER_BLOCK,)
    head_valid = (head_start + hs) < H                 # bool[HEADS_PER_BLOCK]

    # ---------------------------------------------------------------
    #  Load Q for the whole tile – shape (HEADS_PER_BLOCK, Dq)
    # ---------------------------------------------------------------
    offs_q = (
        b * stride_q_batch
        + (head_start + hs)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q, mask=head_valid[:, None], other=0.0)
    q = tl.cast(q, tl.float32)       # FP32 for arithmetic

    # ---------------------------------------------------------------
    #  Stable‑softmax state (max & sum of exp)
    # ---------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)

    # ---------------------------------------------------------------
    #  Accumulator for the *latent* vector (bf16 to keep register pressure low)
    # ---------------------------------------------------------------
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.bfloat16)

    # ---------------------------------------------------------------
    #  Scan the KV cache block‑wise
    # ---------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)                # (BLOCK_K,)
        k_mask = cur_k < L

        # -----  K block -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0)
        k_block = tl.cast(k_block, tl.float32)

        # -----  Q·Kᵀ  (scaled) -----------------------------------------
        prod = tl.dot(q, tl.trans(k_block), out_dtype=tl.float32)   # (HEADS_PER_BLOCK, BLOCK_K)
        prod = prod * scale

        # -----  Stable‑softmax update ------------------------------------
        block_max = tl.max(prod, axis=1)                # (HEADS_PER_BLOCK,)
        new_max = tl.maximum(max_score, block_max)      # (HEADS_PER_BLOCK,)

        # rescaling factor for previously accumulated state
        scale_prev = tl.exp(max_score - new_max)        # (HEADS_PER_BLOCK,)

        # rescale the running soft‑max state
        sum_exp = sum_exp * scale_prev
        latent_fp32 = tl.cast(latent_acc, tl.float32) * scale_prev[:, None]
        latent_acc = tl.cast(latent_fp32, tl.bfloat16)

        # -----  exponentials of the current block -----------------------
        exp_scores = tl.exp(prod - new_max[:, None])   # (HEADS_PER_BLOCK, BLOCK_K)

        # -----  update normaliser ---------------------------------------
        sum_exp = sum_exp + tl.sum(exp_scores, axis=1)   # (HEADS_PER_BLOCK,)

        # -----  V block ------------------------------------------------
        offs_v = (
            b * stride_v_batch
            + cur_k[:, None] * stride_v_len
            + tl.arange(0, Dkv)[None, :] * stride_v_dim
        )
        v_block = tl.load(V_ptr + offs_v,
                          mask=k_mask[:, None],
                          other=0.0)                 # (BLOCK_K, Dkv)
        v_block_f32 = tl.cast(v_block, tl.float32)

        # -----  latent ← latent + exp_scores · V ------------------------
        # (HEADS_PER_BLOCK, BLOCK_K) @ (BLOCK_K, Dkv) → (HEADS_PER_BLOCK, Dkv)
        accum_f32 = tl.dot(exp_scores, v_block_f32)          # (HEADS_PER_BLOCK, Dkv)
        latent_f32 = tl.cast(latent_acc, tl.float32)
        latent_f32 = latent_f32 + accum_f32
        latent_acc = tl.cast(latent_f32, tl.bfloat16)

        # ---------------------------------------------------------------
        #  Update running max for the next iteration
        # ---------------------------------------------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    #  Normalise latent accumulator (still FP32)
    # ------------------------------------------------------------------
    latent_norm = tl.cast(latent_acc, tl.float32) / sum_exp[:, None]   # (HEADS_PER_BLOCK, Dkv)

    # ------------------------------------------------------------------
    #  Per‑head value projection : y = latent_norm @ WV_h   (Dv)
    # ------------------------------------------------------------------
    y_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)

    for start_dv in range(0, Dv, BLOCK_DV):
        cur_dv = start_dv + tl.arange(0, BLOCK_DV)
        dv_mask = cur_dv < Dv

        # WV layout : (H, Dkv, Dv)
        offs_wv = (
            (head_start + hs)[:, None, None] * stride_wv_head
            + tl.arange(0, Dkv)[None, :, None] * stride_wv_in
            + cur_dv[None, None, :] * stride_wv_out
        )
        wv_slice = tl.load(WV_ptr + offs_wv,
                            mask=head_valid[:, None, None] & dv_mask[None, None, :],
                            other=0.0)                     # (HEADS_PER_BLOCK, Dkv, BLOCK_DV)
        # y_head_tile = latent_norm @ wv_slice
        y_head_tile = tl.dot(latent_norm, wv_slice)               # (HEADS_PER_BLOCK, BLOCK_DV)
        y_head = y_head + y_head_tile

    # ------------------------------------------------------------------
    #  Fold the per‑head output into the final model output.
    #  We compute   out += WO_tileᵀ @ y_head_tile   (dim‑wise)
    # ------------------------------------------------------------------
    # flatten the head‑tile so we can treat it as a single vector of length
    # (HEADS_PER_BLOCK * Dv).  The corresponding slice of WO has exactly that
    # shape (dim, HEADS_PER_BLOCK*Dv).
    y_flat = tl.reshape(y_head, (HEADS_PER_BLOCK * Dv,))

    # Offsets for the appropriate slice of WO.
    # WO layout : (dim, H*Dv) → stride_wo_batch = dim * stride_wo_head (i.e. dim*Dv*H)
    # Slice start = head_start * Dv
    wo_start = head_start * Dv
    offs_wo = (
        tl.arange(0, dim)[:, None] * stride_wo_batch
        + (wo_start + tl.arange(0, HEADS_PER_BLOCK * Dv))[None, :] * stride_wo_head
    )
    w_wo_slice = tl.load(WO_ptr + offs_wo,
                         mask=tl.full((dim, HEADS_PER_BLOCK * Dv), True, tl.bool),
                         other=0.0)                               # (dim, HEADS_PER_BLOCK*Dv)

    # out_tile = w_wo_slice @ y_flat   → (dim,)
    out_tile = tl.dot(w_wo_slice, y_flat)                     # (dim,)

    # ------------------------------------------------------------------
    #  Atomic add the contribution of this tile to the final output buffer.
    # ------------------------------------------------------------------
    offs_out = b * stride_out_batch + tl.arange(0, dim) * stride_out_dim
    tl.atomic_add(OUT_ptr + offs_out, out_tile, mask=tl.full((dim,), True, tl.bool))

# -------------------------------------------------------------------------
#  Fast‑path for the common configuration (d_nope == 0)
# -------------------------------------------------------------------------
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
    Optimised fast‑path for the case d_nope == 0.
    All heavy work (attention + per‑head value projection + final projection)
    is performed by a single Triton kernel.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim
    dim = config.dim
    msl = config.max_seq_len

    # ------------------------------------------------------------------
    # 1️⃣  Merge Q‑down/up weights once per config
    # ------------------------------------------------------------------
    if not hasattr(config, "combined_Q_proj_weight"):
        with torch.no_grad():
            config.combined_Q_proj_weight = torch.mm(
                config.Q_proj_up_weight.float(),
                config.Q_proj_down_weight.float(),
            ).to(torch.bfloat16)               # (nh * drope, dim)

    wQ_combined = config.combined_Q_proj_weight      # (nh*drope, dim)

    # ------------------------------------------------------------------
    # 2️⃣ Linear projection for Q (fused) and KV (down‑projection)
    # ------------------------------------------------------------------
    x2 = x.squeeze(1)                                 # (B, dim)
    q_all = F.linear(x2, wQ_combined)                 # (B, nh*drope)
    q_all = q_all.view(bs, nh, drope)                 # (B, nh, drope)

    kv_all = F.linear(x2, wDKV)                       # (B, dkv + drope)

    # ------------------------------------------------------------------
    # 3️⃣ KV‑cache update (write latent + rotated key)
    # ------------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_all[:, :dkv]                   # (B, dkv)
    rope_raw_new   = kv_all[:, dkv:]                  # (B, drope)

    # ----- RoPE for the new key (position = cur_len) -----
    cos_k = cos_tbl[cur_len]                          # (drope,)
    sin_k = sin_tbl[cur_len]                          # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write directly into cache (no extra copy)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len
    query_pos = new_len - 1

    # ------------------------------------------------------------------
    # 4️⃣ RoPE for queries (they have no “no‑pe” part)
    # ------------------------------------------------------------------
    cos_q = cos_tbl[query_pos]                        # (drope,)
    sin_q = sin_tbl[query_pos]                        # (drope,)
    q = q_all * cos_q + _rotate_half(q_all) * sin_q   # (B, nh, drope)

    # ------------------------------------------------------------------
    # 5️⃣ Gather full KV from cache (keys already rotated)
    # ------------------------------------------------------------------
    kv_cache_all = kv_cache.data[:, :new_len, :]      # (B, L, dkv+drope)
    k_rope   = kv_cache_all[..., dkv:]                # (B, L, drope) – already rotated
    v_latent = kv_cache_all[..., :dkv]                # (B, L, dkv)

    # ------------------------------------------------------------------
    # 6️⃣ Layout transformations
    # ------------------------------------------------------------------
    # Q must be (B, H, Dq) – already the case
    # K must be (B, L, Dq) – already the case
    # V must be (B, L, Dkv) – already the case
    # WV layout: (H, Dkv, Dv)
    WV = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (H, Dkv, Dv)

    # WO layout: (dim, H*Dv) – already the case
    # Output buffer
    out = torch.empty((bs, dim), dtype=torch.bfloat16, device=x.device)

    # --------------------------------------------------------------
    # 7️⃣  Strides for the kernel
    # --------------------------------------------------------------
    # Q
    stride_q_batch = q.stride(0)
    stride_q_head  = q.stride(1)
    stride_q_dim   = q.stride(2)

    # K
    stride_k_batch = k_rope.stride(0)
    stride_k_len   = k_rope.stride(1)
    stride_k_dim   = k_rope.stride(2)

    # V
    stride_v_batch = v_latent.stride(0)
    stride_v_len   = v_latent.stride(1)
    stride_v_dim   = v_latent.stride(2)

    # WV
    stride_wv_head = WV.stride(0)
    stride_wv_in   = WV.stride(1)
    stride_wv_out  = WV.stride(2)

    # WO (note: layout is (dim, H*Dv))
    stride_wo_batch = wO.stride(0)      # dim stride (i.e. dim stride for rows)
    stride_wo_head  = wO.stride(1)      # stride between consecutive heads*Dv columns
    stride_wo_dim   = wO.stride(0)      # re‑use dim stride for clarity
    # The above is a bit of a hack – we only need stride between rows (dim) and
    # between columns (head*Dv).  In the kernel we will address the matrix as
    # (dim, H*Dv) → row‑major with stride_wo_batch = dim‑wise and stride_wo_head = 1.

    # OUT
    stride_out_batch = out.stride(0)
    stride_out_dim   = out.stride(1)

    # --------------------------------------------------------------
    # 8️⃣  Launch configuration
    # --------------------------------------------------------------
    HEADS_PER_BLOCK = 32          # keep the register pressure low
    BLOCK_K         = 512
    BLOCK_DV        = 128

    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    scale = 1.0 / math.sqrt(drope)   # drope == total QK dim when d_nope == 0

    _triton_latent_fused[grid](
        # pointers
        q, k_rope, v_latent, WV, wO, out,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len,  stride_k_dim,
        stride_v_batch, stride_v_len,  stride_v_dim,
        stride_wv_head, stride_wv_in, stride_wv_out,
        stride_wo_batch, stride_wo_head, stride_wo_dim,
        stride_out_batch, stride_out_dim,
        # compile‑time constants
        bs, nh, new_len, drope, dkv, dv, dim,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV,
        # launch config – 1 warp per block, 3 stages (same as reference)
        num_warps=1, num_stages=3,
    )

    # ------------------------------------------------------------------
    # 9️⃣  Reshape to the expected [B, 1, Dim] shape
    # ------------------------------------------------------------------
    out = out.unsqueeze(1)   # (B, 1, Dim)

    return out, kv_cache.data


# -------------------------------------------------------------------------
#  Compiled fallback (used when d_nope > 0)
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
#  Main entry point (custom_kernel)
# -------------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Entry point expected by the benchmark harness.
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Extract scalar config values (plain python ints)
    # ------------------------------------------------------------------
    bs   = config.batch_size
    nh   = config.n_heads
    drope = config.qk_rope_head_dim
    d_nope = config.qk_nope_head_dim
    dkv  = config.kv_lora_rank
    dv   = config.v_head_dim
    dim  = config.dim

    # ------------------------------------------------------------------
    # Extract weight tensors (already on device, bfloat16)
    # ------------------------------------------------------------------
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
    # Fast‑path – the common configuration (d_nope == 0)
    # ------------------------------------------------------------------
    if d_nope == 0:
        # we cache the fused Q‑up weight inside the fast‑path function
        out, kv_data = _fast_forward_multihead_combined(
            config,
            x,
            kv_cache,
            config.combined_Q_proj_weight,
            wDKV,
            wUKV,
            wO,
            _cached_cos,
            _cached_sin,
        )
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