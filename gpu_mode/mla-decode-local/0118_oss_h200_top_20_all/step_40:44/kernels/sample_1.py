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

# ------------------------------------------------------------------
# Utility – RoPE rotate half (identical to reference)
# ------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ------------------------------------------------------------------
# Global RoPE tables – lazily created on first call
# ------------------------------------------------------------------
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
                       device=device).unsqueeze_(1)                     # (max_seq_len, 1)
    idx = pos * theta                                                    # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                                 # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ------------------------------------------------------------------
# Triton kernel – fused latent attention (bf16‑dot for Q·K, mixed‑precision for V)
# ------------------------------------------------------------------
@triton.jit
def _triton_latent_onepass_fused(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bfloat16
    K_ptr,                # (B, L, Dq)                 bfloat16
    V_ptr,                # (B, L, Dkv)                bfloat16
    WV_ptr,               # (H, Dkv, Dv)               bfloat16
    Y_ptr,                # (B, H, Dv)                 bfloat16

    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,   # Q    (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,   # K    (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,   # V    (B, L, Dkv)

    stride_wv_head, stride_wv_in, stride_wv_out,   # WV   (H, Dkv, Dv)

    stride_y_batch, stride_y_head, stride_y_dim,   # Y    (B, H, Dv)

    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    L: tl.constexpr,          # current KV length
    Dq: tl.constexpr,         # rope head dim (e.g. 64)
    Dkv: tl.constexpr,        # latent dimension (kv‑lora rank, e.g. 512)
    Dv: tl.constexpr,         # per‑head value dim (e.g. 128)
    scale: tl.constexpr,      # 1 / sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,   # tile size for value‑projection (<= Dv)
):
    """
    Fused attention for the fast‑path (d_nope == 0).
    - Q·Kᵀ computed in bfloat16 (Tensor‑Core) with fp32 accumulation.
    - exp‑scores are fp32, but V accumulation uses mixed‑precision:
      exp‑scores are cast to bfloat16 to enable a bf16∙bf16 → fp32 GEMM.
    - A single program processes HEADS_PER_BLOCK heads of one batch element.
    """
    pid = tl.program_id(0)

    # ---------------------------------------------------------------
    # Identify batch index and head tile handled by this program
    # ---------------------------------------------------------------
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles
    tile = pid % num_head_tiles
    head_start = tile * HEADS_PER_BLOCK

    # ---------------------------------------------------------------
    # Mask for possibly partially‑filled head tile
    # ---------------------------------------------------------------
    hs = tl.arange(0, HEADS_PER_BLOCK)                     # (HEADS_PER_BLOCK,)
    head_valid = head_start + hs < H                        # bool mask

    # ---------------------------------------------------------------
    # Load Q for all heads in the tile (vector‑load per thread)
    # ---------------------------------------------------------------
    offs_q = (
        b * stride_q_batch
        + (head_start + hs)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)               # (HEADS_PER_BLOCK, Dq), bfloat16
    # Keep bf16 – Tensor‑Core will be used. Accumulate in fp32 later.
    # ---------------------------------------------------------------
    # Stable‑softmax state
    # ---------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)

    # ---------------------------------------------------------------
    # Scan KV cache block‑wise (single pass)
    # ---------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)                  # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- Load K block -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0)                 # (BLOCK_K, Dq), bfloat16

        # ----- Q·K dot‑product (scaled) ------------------------------------
        prod = tl.dot(q, tl.trans(k_block), out_dtype=tl.float32)   # (HEADS_PER_BLOCK, BLOCK_K)
        prod = prod * scale

        # ----- Update running max (stable softmax) -------------------------
        block_max = tl.max(prod, axis=1)                # (HEADS_PER_BLOCK,)
        new_max   = tl.maximum(max_score, block_max)

        # ----- Rescaling factor for previous accumulators -------------------
        scale_prev = tl.exp(max_score - new_max)        # (HEADS_PER_BLOCK,)

        # ----- Rescale previous state ---------------------------------------
        sum_exp   = sum_exp * scale_prev
        latent_acc = latent_acc * tl.broadcast_to(scale_prev[:, None], [HEADS_PER_BLOCK, Dkv])

        # ----- Exponentials of the current block -----------------------------
        exp_scores = tl.exp(prod - new_max[:, None])   # (HEADS_PER_BLOCK, BLOCK_K)

        # ----- Normaliser update ---------------------------------------------
        sum_exp = sum_exp + tl.sum(exp_scores, axis=1)   # (HEADS_PER_BLOCK,)

        # ----- Weighted accumulation of V (latent) ---------------------------
        # V block : (BLOCK_K, Dkv)
        offs_v = (
            b * stride_v_batch
            + cur_k[:, None] * stride_v_len
            + tl.arange(0, Dkv)[None, :] * stride_v_dim
        )
        v_block = tl.load(V_ptr + offs_v,
                          mask=k_mask[:, None],
                          other=0.0)                 # (BLOCK_K, Dkv), bfloat16

        # Mixed‑precision accumulation:
        #   exp_scores (fp32) -> bf16 for GEMM, accumulate in fp32.
        exp_bf16 = tl.cast(exp_scores, tl.bfloat16)
        latent_acc = latent_acc + tl.dot(exp_bf16, v_block, out_dtype=tl.float32)

        # ----- Prepare for next iteration ------------------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # Normalise the latent accumulator
    # ------------------------------------------------------------------
    latent = latent_acc / sum_exp[:, None]   # (HEADS_PER_BLOCK, Dkv), fp32

    # ------------------------------------------------------------------
    # Per‑head value projection : y = latent @ WV_h   (DW = Dv)
    # ------------------------------------------------------------------
    y_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)

    for start_dv in range(0, Dv, BLOCK_DV):
        cur_dv = start_dv + tl.arange(0, BLOCK_DV)
        dv_mask = cur_dv < Dv

        # Offsets to WV (layout: (H, Dkv, Dv))
        offs_wv = (
            (head_start + hs)[:, None, None] * stride_wv_head
            + tl.arange(0, Dkv)[None, :, None] * stride_wv_in
            + cur_dv[None, None, :] * stride_wv_out
        )
        wv_slice = tl.load(WV_ptr + offs_wv,
                           mask=head_valid[:, None, None] & dv_mask[None, None, :],
                           other=0.0)                     # (HEADS_PER_BLOCK, Dkv, BLOCK_DV), bf16
        # dot: (HEADS_PER_BLOCK, Dkv) x (HEADS_PER_BLOCK, Dkv, BLOCK_DV) -> (HEADS_PER_BLOCK, BLOCK_DV)
        y_head_tile = tl.dot(latent, wv_slice)               # (HEADS_PER_BLOCK, BLOCK_DV), fp32
        y_head = y_head + y_head_tile

    # ------------------------------------------------------------------
    # Store per‑head output (cast back to bfloat16)
    # ------------------------------------------------------------------
    offs_y = (
        b * stride_y_batch
        + (head_start + hs)[:, None] * stride_y_head
        + tl.arange(0, Dv)[None, :] * stride_y_dim
    )
    tl.store(Y_ptr + offs_y,
             tl.cast(y_head, tl.bfloat16),
             mask=head_valid[:, None])


# ------------------------------------------------------------------
# Fast‑path – d_nope == 0 (the common configuration)
# ------------------------------------------------------------------
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
    Heavy attention (latent + per‑head value projection) is done by the
    fused Triton kernel above. The final output projection (WO) remains a cuBLAS GEMM.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim

    # --------------------------------------------------
    # 0️⃣ Merge the Q‑down/up projections once (cached)
    # --------------------------------------------------
    x2 = x.squeeze(1)                                     # (B, dim)
    all_proj = F.linear(x2, wQ_combined)                  # (B, nh*drope + dkv + drope)

    total_q_dim = nh * drope
    q_all   = all_proj[:, :total_q_dim].view(bs, nh, drope)   # (B, nh, drope)
    kv_lora = all_proj[:, total_q_dim:]                       # (B, dkv + drope)

    # --------------------------------------------------
    # 3️⃣ KV cache update (store latent + rotated key)
    # --------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora[:, :dkv]           # (B, dkv)
    rope_raw_new   = kv_lora[:, dkv:]          # (B, drope)

    # ----- RoPE for the new key (position = cur_len) -----
    cos_k = cos_tbl[cur_len]                   # (drope,)
    sin_k = sin_tbl[cur_len]                   # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write to cache (no extra copy – cache already bfloat16)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len
    query_pos = new_len - 1

    # --------------------------------------------------
    # 4️⃣ RoPE for queries (they have no “no‑pe” part)
    # --------------------------------------------------
    cos_q = cos_tbl[query_pos]                  # (drope,)
    sin_q = sin_tbl[query_pos]                  # (drope,)
    q = q_all * cos_q + _rotate_half(q_all) * sin_q   # (B, nh, drope)

    # --------------------------------------------------
    # 5️⃣ Gather full KV from cache (rotated keys already stored)
    # --------------------------------------------------
    kv_all   = kv_cache.data[:, :new_len, :]      # (B, L, dkv + drope)
    k_rope   = kv_all[..., dkv:]                  # (B, L, drope) – already ROTATED
    v_latent = kv_all[..., :dkv]                  # (B, L, dkv)

    # --------------------------------------------------
    # 6️⃣ Allocate per‑head output buffer (y_head)
    # --------------------------------------------------
    y_head = torch.empty((bs, nh, dv),
                         dtype=torch.bfloat16,
                         device=x.device)

    # --------------------------------------------------
    # 7️⃣ Run the fused Triton kernel (latent → head)
    # --------------------------------------------------
    # Strides for Q, K, V
    stride_q_batch = q.stride(0)
    stride_q_head  = q.stride(1)
    stride_q_dim   = q.stride(2)

    stride_k_batch = k_rope.stride(0)
    stride_k_len   = k_rope.stride(1)
    stride_k_dim   = k_rope.stride(2)

    stride_v_batch = v_latent.stride(0)
    stride_v_len   = v_latent.stride(1)
    stride_v_dim   = v_latent.stride(2)

    # Strides for WV (value‑projection weights)
    # WV layout is (nh, dkv, dv)
    wV = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)
    stride_wv_head = wV.stride(0)
    stride_wv_in   = wV.stride(1)
    stride_wv_out  = wV.stride(2)

    # Strides for Y (output per‑head)
    stride_y_batch = y_head.stride(0)
    stride_y_head  = y_head.stride(1)
    stride_y_dim   = y_head.stride(2)

    # Tuned block sizes – empirically good for H200
    HEADS_PER_BLOCK = 128          # all heads in one program
    BLOCK_K = 256                  # KV‑length tile size
    BLOCK_DV = 128                 # value‑projection tile (dv == 128)

    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    scale = 1.0 / math.sqrt(drope)   # rope dim == total QK dim when d_nope == 0

    _triton_latent_onepass_fused[grid](
        # pointers
        q, k_rope, v_latent, wV, y_head,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len,   stride_k_dim,
        stride_v_batch, stride_v_len,   stride_v_dim,
        stride_wv_head, stride_wv_in, stride_wv_out,
        stride_y_batch, stride_y_head, stride_y_dim,
        # compile‑time constants
        bs, nh, new_len, drope, dkv, dv,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV,
        # launch config – 4 warps = 128 threads (exactly HEADS_PER_BLOCK)
        num_warps=4, num_stages=3,
    )

    # --------------------------------------------------
    # 8️⃣ Final output projection (WO) – cuBLAS GEMM
    # --------------------------------------------------
    y_head_flat = y_head.view(bs, nh * dv)                # (B, H*Dv)
    out = F.linear(y_head_flat, wO)                       # (B, Dim)
    out = out.unsqueeze(1)                                # (B, 1, Dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Compiled fallback – unchanged (used when d_nope > 0)
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
    drope = config.qk_rope_head_dim
    d_nope = config.qk_nope_head_dim
    dkv = config.kv_lora_rank
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
        # Merge the Q‑down/up projections once (cached)
        if not hasattr(config, "combined_Q_proj_weight"):
            # Compute in FP32 for better numerical stability, then cast back
            with torch.no_grad():
                config.combined_Q_proj_weight = torch.mm(
                    wUQ.float(),
                    wDQ.float()
                ).to(torch.bfloat16)   # (nh * drope, dim)
        wQ_combined = config.combined_Q_proj_weight

        out, kv_data = _fast_forward_multihead_combined(
            config, x, kv_cache,
            wQ_combined, wDKV, wUKV, wO,
            _cached_cos, _cached_sin,
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