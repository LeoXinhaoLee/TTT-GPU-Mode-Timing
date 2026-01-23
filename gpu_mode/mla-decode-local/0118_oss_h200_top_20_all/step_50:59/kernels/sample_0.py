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
# ----------------------------------------------------------------------
# Utility functions (used by the reference implementation)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half (used by RoPE)."""
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
# Triton kernel – fused attention that only returns the (soft‑max‑weighted)
# latent values (no projection).  Projection is applied afterwards
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_no_proj_kernel(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,               # (B, H, Dq)                 bf16   – input queries
    K_ptr,               # (B, L, Dq)                 bf16   – keys (rope‑rotated)
    V_ptr,               # (B, L, Dkv)                bf16   – latent values (no‑rope)
    out_X_ptr,           # (B, H, Dkv)                fp32   – weighted sum of V (normalised)
    sum_exp_ptr,         # (B, H)                     fp32   – softmax denominator (debug / optional)
    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len,  stride_k_dim,
    stride_v_batch, stride_v_len,  stride_v_dim,
    stride_outX_batch, stride_outX_head, stride_outX_dim,
    stride_sum_batch, stride_sum_head,
    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    Dq: tl.constexpr,         # rope‑dim (e.g. 64)
    Dkv: tl.constexpr,        # kv‑lora rank (e.g. 512)
    scale: tl.constexpr,      # 1 / sqrt(Dq)
    # ------------------------------------------------------------------
    # Tiling parameters (tuned for the reference workload)
    # ------------------------------------------------------------------
    HEADS_PER_BLOCK: tl.constexpr,   # #heads processed by a program
    BLOCK_K: tl.constexpr,           # keys / values processed per iteration
    # ------------------------------------------------------------------
    # Runtime argument
    # ------------------------------------------------------------------
    L: tl.int32,               # current KV length (dynamic)
):
    """
    Attention kernel that:
      * computes softmax‑scaled dot‑product Q·Kᵀ,
      * accumulates the weighted latent values V (normalised),
      * returns X = (∑ exp·V) / (∑ exp) for each head.
    The per‑head value‑projection (V × wVᵀ) is performed **outside** this kernel.
    """
    pid = tl.program_id(0)                         # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    batch_id = pid // num_head_tiles                # which batch instance
    tile_id  = pid % num_head_tiles                 # which head‑tile inside the batch
    head_start = tile_id * HEADS_PER_BLOCK          # first head handled by this program

    # ------------------------------------------------------------------
    # 1️⃣ Load queries for this tile (shape: HEADS_PER_BLOCK × Dq)
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
    # 2️⃣ Buffers for stable softmax & accumulation of V
    # ------------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (Hpb,)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (Hpb,)
    X_acc     = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)    # (Hpb, Dkv)

    # ------------------------------------------------------------------
    # 3️⃣ Main loop over KV blocks
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
                          cache_modifier='CA')               # (BLOCK_K, Dq)  bf16

        # ----- dot(Q, K) –‑> scores ------------------------------------
        prod = tl.dot(q, tl.permute(k_block, (1, 0)))            # (HEADS_PER_BLOCK, BLOCK_K) bf16
        score_f32 = tl.cast(prod, tl.float32) * scale           # fp32

        # ----- stable soft‑max update ---------------------------------
        block_max = tl.max(score_f32, axis=1)                   # (HEADS_PER_BLOCK)
        new_max   = tl.maximum(max_score, block_max)            # (HEADS_PER_BLOCK)

        exp_factor = tl.exp(max_score - new_max)                # (HEADS_PER_BLOCK)

        sum_exp = sum_exp * exp_factor
        X_acc   = X_acc * exp_factor[:, None]

        exp_score = tl.exp(score_f32 - new_max[:, None])        # (HEADS_PER_BLOCK, BLOCK_K)
        sum_exp = sum_exp + tl.sum(exp_score, axis=1)           # (HEADS_PER_BLOCK)

        # ----- Load V slice & accumulate -------------------------------
        offs_v = (
            batch_id * stride_v_batch
            + cur_k[:, None] * stride_v_len
            + tl.arange(0, Dkv)[None, :] * stride_v_dim
        )
        v_slice = tl.load(V_ptr + offs_v,
                          mask=k_mask[:, None],
                          other=0.0,
                          cache_modifier='CA')                # (BLOCK_K, Dkv) bf16
        v_fp32 = tl.cast(v_slice, tl.float32)                # (BLOCK_K, Dkv)

        # weighted sum of V for this block
        X_block = tl.dot(exp_score, v_fp32)                    # (HEADS_PER_BLOCK, Dkv)

        # accumulate
        X_acc = X_acc + X_block

        # update running max for the next block
        max_score = new_max

    # ------------------------------------------------------------------
    # 4️⃣ Normalise the accumulated latent values (divide by softmax denominator)
    # ------------------------------------------------------------------
    norm = 1.0 / sum_exp[:, None]                # (HEADS_PER_BLOCK, 1)
    X_norm = X_acc * norm                        # (HEADS_PER_BLOCK, Dkv)

    # ------------------------------------------------------------------
    # 5️⃣ Store the normalised latent values (will be projected later)
    # ------------------------------------------------------------------
    offs_outX = (
        batch_id * stride_outX_batch
        + (head_start + head_range)[:, None] * stride_outX_head
        + tl.arange(0, Dkv)[None, :] * stride_outX_dim
    )
    tl.store(out_X_ptr + offs_outX,
             X_norm,
             mask=head_valid[:, None])

    # ------------------------------------------------------------------
    # 6️⃣ (Optional) store the denominator – callers may ignore it
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
    The heavy attention is performed by a Triton kernel that returns the
    soft‑max‑normalised latent values.  The per‑head value projection
    is then applied with a tiny torch einsum, dramatically reducing the
    amount of data read from the GPU.
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
    # 3️⃣ Gather K (rope‑rotated) and V (latent) from cache
    # --------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]           # (B, L, dkv + drope)
    K = kv_all[..., dkv:]                            # (B, L, drope) – rope‑rotated keys
    V = kv_all[..., :dkv]                            # (B, L, dkv)  – latent values

    # --------------------------------------------------------------
    # 4️⃣ Triton kernel – attention, returns normalised latent sum X
    # --------------------------------------------------------------
    # Buffer for the (normalised) latent values X = (∑exp·V)/∑exp
    X_norm = torch.empty((bs, nh, dkv), dtype=torch.float32, device=x.device)
    # (optional) denominator – not used later, but allocated for kernel API
    sum_exp = torch.empty((bs, nh), dtype=torch.float32, device=x.device)

    scale = 1.0 / math.sqrt(drope)   # Dq == drope

    grid = lambda meta: (bs * triton.cdiv(nh, meta["HEADS_PER_BLOCK"]),)

    # Tuned launch configuration – one tile processes all heads
    _triton_attn_no_proj_kernel[grid](
        # pointers
        Q, K, V,
        X_norm,
        sum_exp,
        # strides
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        X_norm.stride(0), X_norm.stride(1), X_norm.stride(2),
        sum_exp.stride(0), sum_exp.stride(1),
        # compile‑time constants
        bs, nh, drope, dkv, scale,
        # tuning constants
        128,           # HEADS_PER_BLOCK – process all heads in one tile
        1024,          # BLOCK_K – good L2 reuse for our prefills
        # runtime argument
        new_len,
        num_warps=16   # more warps help on Hopper (H200)
    )

    # --------------------------------------------------------------
    # 5️⃣ Per‑head value projection (X_norm @ wV_T)
    # --------------------------------------------------------------
    # wUKV has shape ((dv) * nh, dkv) → (nh, dv, dkv) → (nh, dkv, dv)
    wV_T = wUKV.view(nh, dv, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    # einsum:  (B, H, Dkv)  ×  (H, Dkv, Dv)  →  (B, H, Dv)
    # Cast wV_T to fp32 for the matmul (torch will promote automatically)
    y_head = torch.einsum('bhd,hdv->bhv', X_norm, wV_T.to(torch.float32))
    y_head = y_head.to(torch.bfloat16)          # back to model dtype

    # --------------------------------------------------------------
    # 6️⃣ Final model‑dim projection (single matmul, cuBLAS‑accelerated)
    # --------------------------------------------------------------
    y_flat = y_head.view(bs, nh * dv)                # (B, nh*dv)
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