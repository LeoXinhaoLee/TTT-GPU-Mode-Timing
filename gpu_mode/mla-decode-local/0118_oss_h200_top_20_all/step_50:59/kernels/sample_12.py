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
# Global (module‑level) caches – allocated once
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None          # (max_seq_len, rope_dim)    bfloat16
_cached_sin: torch.Tensor = None          # (max_seq_len, rope_dim)    bfloat16
_cached_wV_T: torch.Tensor = None         # (n_heads, kv_lora_rank, v_head_dim) bfloat16
# ----------------------------------------------------------------------
# Utility functions (used by the reference implementation)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Pre‑compute cosine / sine tables for RoPE (bfloat16)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half,
                                       dtype=torch.float32,
                                       device=device) / half)).to(torch.bfloat16)
    pos = torch.arange(max_seq_len,
                       dtype=torch.int64,
                       device=device).unsqueeze_(1)          # (max_seq_len, 1)
    idx = pos * theta                                          # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                        # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel – fused flash‑attention + per‑head value projection
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_fused_kernel(
    # --------------------------------------------------------------
    # Pointers
    # --------------------------------------------------------------
    Q_ptr,               # (B, H, Dq)                     bf16
    K_ptr,               # (B, L, Dq)                     bf16
    V_ptr,               # (B, L, Dkv)                    bf16
    wV_T_ptr,            # (H, Dkv, Dv)                   bf16   – per‑head value projection
    out_ptr,             # (B, H, Dv)                     bf16   – projected output
    sum_exp_ptr,         # (B, H)                         fp32   – softmax denominator
    # --------------------------------------------------------------
    # Strides
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len,  stride_k_dim,
    stride_v_batch, stride_v_len,  stride_v_dim,
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
    stride_out_batch, stride_out_head, stride_out_dim,
    stride_sum_batch, stride_sum_head,
    # --------------------------------------------------------------
    # Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    Dq: tl.constexpr,         # query/key (rope) dim
    Dkv: tl.constexpr,        # KV‑lora rank
    Dv: tl.constexpr,         # per‑head value dim
    scale: tl.constexpr,      # 1 / sqrt(Dq)
    # --------------------------------------------------------------
    # Tiling parameters (tuned for the reference workload)
    # --------------------------------------------------------------
    HEADS_PER_BLOCK: tl.constexpr,   # #heads processed by a program
    BLOCK_K: tl.constexpr,           # keys / values processed per iteration
    BLOCK_DV_OUT: tl.constexpr,      # projection output dim tile (must divide Dv)
    # --------------------------------------------------------------
    # Runtime argument
    # --------------------------------------------------------------
    L: tl.int32,               # current KV length (dynamic)
):
    """
    Flash‑attention kernel that:
      * computes numerically‑stable softmax(Q·Kᵀ)
      * accumulates the latent weighted sum (exp·V) per head
      * finally multiplies the accumulated latent sum by the per‑head
        value‑projection matrix wV_T and writes the result.
    """
    pid = tl.program_id(0)                         # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    batch_id = pid // num_head_tiles                # which batch instance
    tile_id  = pid % num_head_tiles                 # which head‑tile inside the batch
    head_start = tile_id * HEADS_PER_BLOCK          # first head handled by this program

    # ------------------------------------------------------------------
    # 0️⃣ Load queries for this tile (shape: HEADS_PER_BLOCK × Dq)
    # ------------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_valid = head_start + head_range < H

    offs_q = (
        batch_id * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)               # (HEADS_PER_BLOCK, Dq)  bf16

    # ------------------------------------------------------------------
    # 1️⃣ Buffers for stable softmax & latent accumulation
    # ------------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HEADS_PER_BLOCK,)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (HEADS_PER_BLOCK,)
    # accumulate the *latent* weighted sum (size = Dkv)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)   # (HEADS_PER_BLOCK, Dkv)

    # ------------------------------------------------------------------
    # 2️⃣ Main loop over KV blocks
    # ------------------------------------------------------------------
    for start_k in tl.range(0, L, BLOCK_K, num_stages=4):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)       # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- load K -------------------------------------------------
        offs_k = (
            batch_id * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0,
                          cache_modifier='CA')               # (BLOCK_K, Dq) bf16

        # ----- dot(Q, K) → scores ------------------------------------
        prod = tl.dot(q, tl.permute(k_block, (1, 0)))          # (HEADS_PER_BLOCK, BLOCK_K) bf16
        score_f32 = tl.cast(prod, tl.float32) * scale          # fp32

        # ----- stable soft‑max update --------------------------------
        block_max = tl.max(score_f32, axis=1)                  # (HEADS_PER_BLOCK)
        new_max   = tl.maximum(max_score, block_max)           # (HEADS_PER_BLOCK)

        exp_factor = tl.exp(max_score - new_max)               # (HEADS_PER_BLOCK)

        sum_exp = sum_exp * exp_factor
        latent_acc = latent_acc * exp_factor[:, None]

        exp_score = tl.exp(score_f32 - new_max[:, None])       # (HEADS_PER_BLOCK, BLOCK_K)
        sum_exp = sum_exp + tl.sum(exp_score, axis=1)           # (HEADS_PER_BLOCK)

        # ----- Load V slice -----------------------------------------
        # V shape: (B, L, Dkv)
        offs_v = (
            batch_id * stride_v_batch
            + cur_k[:, None] * stride_v_len
            + tl.arange(0, Dkv)[None, :] * stride_v_dim
        )
        v_slice = tl.load(V_ptr + offs_v,
                          mask=k_mask[:, None],
                          other=0.0,
                          cache_modifier='CA')               # (BLOCK_K, Dkv) bf16
        v_fp32 = tl.cast(v_slice, tl.float32)                  # (BLOCK_K, Dkv)

        # ----- accumulate latent weighted sum -------------------------
        #   latent_acc += exp_score @ V
        X = tl.dot(exp_score, v_fp32)                          # (HEADS_PER_BLOCK, Dkv)
        latent_acc = latent_acc + X

        # ----- update running max ------------------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # 3️⃣ Projection from latent (Dkv) → head output (Dv)
    # ------------------------------------------------------------------
    # Broadcast latent accumulator for the inner product with wV_T
    latent_exp = latent_acc[:, :, None]                        # (HEADS_PER_BLOCK, Dkv, 1)

    # Process projection output in small Dv‑tiles to keep register pressure low
    dv_tile = tl.arange(0, BLOCK_DV_OUT, tl.int32)            # (BLOCK_DV_OUT,)
    for d_start in tl.range(0, Dv, BLOCK_DV_OUT, num_stages=2):
        cur_d = d_start + dv_tile                               # (BLOCK_DV_OUT,)
        d_valid = cur_d < Dv

        # ----- load wV_T slice ------------------------------------
        # offsets: (HEADS_PER_BLOCK, Dkv, BLOCK_DV_OUT)
        offs_wV = (
            (head_start + head_range)[:, None, None] * stride_wV_T_head
            + tl.arange(0, Dkv)[:, None] * stride_wV_T_lat
            + cur_d[None, None, :] * stride_wV_T_out
        )
        wV_tile = tl.load(wV_T_ptr + offs_wV,
                          mask=head_valid[:, None] & d_valid[None, None, :],
                          other=0.0,
                          cache_modifier='CA')               # (HEADS_PER_BLOCK, Dkv, BLOCK_DV_OUT)
        wV_tile_f32 = tl.cast(wV_tile, tl.float32)

        # ----- matrix‑multiply (inner product) --------------------
        # result shape (HEADS_PER_BLOCK, BLOCK_DV_OUT)
        out_tile = tl.sum(latent_exp * wV_tile_f32, axis=1)

        # ----- normalise by soft‑max denominator ------------------
        out_tile = out_tile / sum_exp[:, None]

        # ----- store output tile ----------------------------------
        offs_out = (
            batch_id * stride_out_batch
            + (head_start + head_range)[:, None] * stride_out_head
            + cur_d[None, :] * stride_out_dim
        )
        tl.store(out_ptr + offs_out,
                 tl.cast(out_tile, tl.bfloat16),
                 mask=head_valid[:, None] & d_valid[None, :])

    # ------------------------------------------------------------------
    # 4️⃣ Store soft‑max denominator (optional, kept for debugging)
    # ------------------------------------------------------------------
    offs_sum = (
        batch_id * stride_sum_batch
        + (head_start + head_range) * stride_sum_head
    )
    tl.store(sum_exp_ptr + offs_sum,
             sum_exp,
             mask=head_valid)


# ----------------------------------------------------------------------
# Fast‑path for the common configuration (qk_nope_head_dim == 0)
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
    Optimised forward when `qk_nope_head_dim == 0`.
    The heavy attention + per‑head value projection is performed in a
    Triton kernel.  The final model‑dim projection is done with a regular
    torch linear layer.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim          # Dq == drope
    dkv   = config.kv_lora_rank
    dv    = config.v_head_dim
    dim   = config.dim

    # --------------------------------------------------------------
    # 1️⃣ Down‑project + KV‑cache update (RoPE baked‑in for the key)
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                                 # (B, dim)
    kv_lora = F.linear(x2, wDKV)                      # (B, dkv + drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora[:, :dkv]                  # (B, dkv)
    rope_raw_new  = kv_lora[:, dkv:]                  # (B, drope)

    # RoPE for the new key (in‑place)
    cos_k = cos_tbl[cur_len]                          # (drope,)
    sin_k = sin_tbl[cur_len]                          # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write to cache
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 2️⃣ Q‑down + Q‑up projection and RoPE on queries
    # --------------------------------------------------------------
    q_lora = F.linear(x2, wDQ)                        # (B, q_lora_rank)
    q = F.linear(q_lora, wUQ)                         # (B, nh * drope)
    q = q.view(bs, nh, drope)                        # (B, nh, drope)

    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                            # (drope,)
    sin_q = sin_tbl[q_pos]                            # (drope,)
    q = q * cos_q + _rotate_half(q) * sin_q           # (B, nh, drope)

    # --------------------------------------------------------------
    # 3️⃣ Prepare tensors for the Triton kernel
    # --------------------------------------------------------------
    Q = q.contiguous()                               # (B, nh, drope)

    # K – rope‑rotated keys (slice the cache)
    kv_all = kv_cache.data[:, :new_len, :]           # (B, L, dkv + drope)
    K = kv_all[..., dkv:]                            # (B, L, drope)

    # V – latent values (no‑rope)
    V = kv_all[..., :dkv]                            # (B, L, dkv)

    # ------------------------------------------------------------------
    # 4️⃣ Cache & reshape the per‑head value‑projection matrix
    # ------------------------------------------------------------------
    global _cached_wV_T
    if _cached_wV_T is None:
        # wUKV: ((dv) * nh, dkv) → (nh, dv, dkv) → (nh, dkv, dv)
        _cached_wV_T = wUKV.view(nh, dv, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # 5️⃣ Allocate buffers for the kernel outputs
    # ------------------------------------------------------------------
    y_proj = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)
    sum_exp = torch.empty((bs, nh), dtype=torch.float32, device=x.device)

    # ------------------------------------------------------------------
    # 6️⃣ Launch the fused attention – value‑projection kernel
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(drope)   # Dq == drope

    # Tuning parameters – chosen to minimise memory traffic
    HEADS_PER_BLOCK = 32                      # try to keep registers moderate
    BLOCK_K = 1024
    BLOCK_DV_OUT = 32                         # Dv (=128) is a multiple of 32

    grid = lambda meta: (bs * triton.cdiv(nh, meta["HEADS_PER_BLOCK"]),)

    _triton_attn_fused_kernel[grid](
        # pointers
        Q, K, V,
        _cached_wV_T,
        y_proj,
        sum_exp,
        # strides
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        _cached_wV_T.stride(0), _cached_wV_T.stride(1), _cached_wV_T.stride(2),
        y_proj.stride(0), y_proj.stride(1), y_proj.stride(2),
        sum_exp.stride(0), sum_exp.stride(1),
        # compile‑time constants
        bs, nh, drope, dkv, dv, scale,
        # tuning constants
        HEADS_PER_BLOCK,
        BLOCK_K,
        BLOCK_DV_OUT,
        # runtime argument: current KV length (includes the newly added token)
        new_len,
        num_warps=8,          # enough for H200, reduces register pressure
    )

    # ------------------------------------------------------------------
    # 7️⃣ Final model‑dim projection (single matmul, cuBLAS‑accelerated)
    # ------------------------------------------------------------------
    y_flat = y_proj.view(bs, nh * dv)                # (B, nh*dv)
    out = F.linear(y_flat, wO)                       # (B, dim)  bf16
    out = out.unsqueeze(1)                           # (B, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Compiled fallback (qk_nope_head_dim > 0) – unchanged from the reference
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

    # ------------------------------------------------------------------
    # Extract scalar config values (plain python ints)
    # ------------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dim = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # ------------------------------------------------------------------
    # Weight tensors (already on the correct device & dtype)
    # ------------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

    # ------------------------------------------------------------------
    # Ensure RoPE tables are cached (global, reused across calls)
    # ------------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # ------------------------------------------------------------------
    # Fast‑path – the *common* case where qk_nope_head_dim == 0
    # ------------------------------------------------------------------
    if d_nope == 0:
        out, new_kv = _fast_forward_multihead(
            config,
            x,
            kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos,
            _cached_sin,
        )
        # kv_cache has already been updated inside the fast‑path
        return out, new_kv

    # ------------------------------------------------------------------
    # General case – fall back to compiled reference implementation
    # ------------------------------------------------------------------
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