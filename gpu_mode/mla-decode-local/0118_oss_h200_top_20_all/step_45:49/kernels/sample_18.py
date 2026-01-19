### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATMENTS BLOCK ###

# ----------------------------------------------------------------------
# Global caches (shared across kernel calls) – never re‑allocated
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None          # (max_seq_len, rope_dim)    bfloat16
_cached_sin: torch.Tensor = None          # (max_seq_len, rope_dim)    bfloat16
_cached_wV_T: torch.Tensor = None         # (n_heads, kv_lora_rank, v_head_dim) bfloat16

# ----------------------------------------------------------------------
# Utility helpers
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
                       device=device).unsqueeze_(1)           # (max_seq_len, 1)
    idx = pos * theta                                          # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                        # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)

# ----------------------------------------------------------------------
# Triton kernel – fused attention (Q·Kᵀ + softmax) + weighted sum of V‑latent
# ----------------------------------------------------------------------
@triton.jit
def _attn_latent_kernel(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,               # (B, H, Dq)                     bf16
    K_ptr,               # (B, L, Dq)                     bf16
    V_ptr,               # (B, L, Dv)                     bf16
    Out_ptr,             # (B, H, Dv)                     bf16
    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len,  stride_k_dim,
    stride_v_batch, stride_v_len,  stride_v_dim,
    stride_out_batch, stride_out_head, stride_out_dim,
    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    Dq: tl.constexpr,         # rope dim (e.g. 64)
    Dv: tl.constexpr,         # latent value dim (e.g. 512)
    scale: tl.constexpr,      # 1 / sqrt(Dq)
    # ------------------------------------------------------------------
    # Tiling parameters (tuned for the reference workload)
    # ------------------------------------------------------------------
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # ------------------------------------------------------------------
    # Runtime argument
    # ------------------------------------------------------------------
    L: tl.int32,               # current KV length (dynamic)
):
    """
    Computes a single‑token attention where:
        * Q : (B, H, Dq)  – already RoPE‑applied
        * K : (B, L, Dq)  – already RoPE‑applied (stored in the KV cache)
        * V : (B, L, Dv)  – latent values (no RoPE)
      Returns:
        * Out : (B, H, Dv) = Σₖ softmax(Q·Kᵀ)·V   (bfloat16)
    The algorithm uses the classic “online” softmax to avoid a
    separate max‑pass, which is crucial for long KV sequences.
    """
    pid = tl.program_id(0)                         # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # head‑tile inside batch
    head_start = tile * HEADS_PER_BLOCK              # first head processed by this program

    # --------------------------------------------------------------
    # 1️⃣ Load the queries for this tile (shape: HEADS_PER_BLOCK × Dq)
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
                 other=0.0)                      # (HEADS_PER_BLOCK, Dq)  bf16

    # --------------------------------------------------------------
    # 2️⃣ Allocate buffers for the stable‑softmax and result accumulation
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)
    out_acc   = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)

    # --------------------------------------------------------------
    # 3️⃣ Main loop over key/value blocks
    # --------------------------------------------------------------
    for start_k in tl.range(0, L, BLOCK_K, num_stages=4):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)          # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- Load keys -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0,
                          cache_modifier='CA')               # (BLOCK_K, Dq)  bf16

        # ----- Dot‑product Q·Kᵀ -----------------------------------------
        prod = tl.dot(q, tl.permute(k_block, (1, 0)))            # (HEADS_PER_BLOCK, BLOCK_K)  bf16
        score_f32 = tl.cast(prod, tl.float32) * scale

        # ----- Stable soft‑max update ------------------------------------
        block_max = tl.max(score_f32, axis=1)                    # (HEADS_PER_BLOCK)
        new_max   = tl.maximum(max_score, block_max)

        # rescale previous sums (see FlashAttention paper)
        exp_factor = tl.exp(max_score - new_max)                 # (HEADS_PER_BLOCK)
        sum_exp = sum_exp * exp_factor
        out_acc = out_acc * exp_factor[:, None]

        exp_score = tl.exp(score_f32 - new_max[:, None])         # (HEADS_PER_BLOCK, BLOCK_K)
        sum_exp = sum_exp + tl.sum(exp_score, axis=1)            # (HEADS_PER_BLOCK)

        # ----- Load values (latent) --------------------------------------
        offs_v = (
            b * stride_v_batch
            + cur_k[:, None] * stride_v_len
            + tl.arange(0, Dv)[None, :] * stride_v_dim
        )
        v_block = tl.load(V_ptr + offs_v,
                          mask=k_mask[:, None],
                          other=0.0,
                          cache_modifier='CA')               # (BLOCK_K, Dv)  bf16
        v_fp32 = tl.cast(v_block, tl.float32)                   # (BLOCK_K, Dv)

        # ----- Weighted sum of values ------------------------------------
        #   exp_score (HEADS_PER_BLOCK, BLOCK_K)  ×  v_fp32 (BLOCK_K, Dv)
        #   → (HEADS_PER_BLOCK, Dv)
        weighted = tl.dot(exp_score, v_fp32)                     # fp32
        out_acc = out_acc + weighted

        # ----- Update running max for next block -------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # 4️⃣ Normalise (divide by softmax denominator)
    # ------------------------------------------------------------------
    out = out_acc / sum_exp[:, None]                           # (HEADS_PER_BLOCK, Dv)

    # ------------------------------------------------------------------
    # 5️⃣ Store the result
    # ------------------------------------------------------------------
    offs_out = (
        b * stride_out_batch
        + (head_start + head_range)[:, None] * stride_out_head
        + tl.arange(0, Dv)[None, :] * stride_out_dim
    )
    tl.store(Out_ptr + offs_out,
             tl.cast(out, tl.bfloat16),
             mask=head_valid[:, None])

# ----------------------------------------------------------------------
# Optimised fast‑path for the common configuration (no “no‑PE” head dim)
# ----------------------------------------------------------------------
def _fast_forward_multihead_optimized(
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
):
    """
    Fast‑path when qk_nope_head_dim == 0.
    Steps:
      1. Down‑project KV & insert into cache (rope baked‑in).
      2. Down‑project Q, up‑project, apply RoPE.
      3. Triton kernel → weighted sum of latent V.
      4. Per‑head V‑projection (wV_T).
      5. Final linear projection (wO).
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim          # = Dq
    dkv   = config.kv_lora_rank              # = Dv (latent dim)
    dv    = config.v_head_dim                # head‑dim after projection
    dim   = config.dim

    # --------------------------------------------------------------
    # 1️⃣ KV down‑projection + cache update (embedded RoPE)
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                 # (B, dim)
    kv_lora = F.linear(x2, wDKV)      # (B, dkv + drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora[:, :dkv]               # (B, dkv)
    rope_raw_new   = kv_lora[:, dkv:]               # (B, drope)

    # rotate‑half + apply sin/cos for the *key* at position cur_len
    cos_k = cos_tbl[cur_len]                       # (drope,)
    sin_k = sin_tbl[cur_len]                       # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write into cache (latent first, then rope‑rotated part)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 2️⃣ Q down‑projection → up‑projection → RoPE
    # --------------------------------------------------------------
    q_lora = F.linear(x2, wDQ)                     # (B, q_lora_rank)
    q_up   = F.linear(q_lora, wUQ)                 # (B, nh * drope)
    q_up   = q_up.view(bs, nh, drope)              # (B, nh, drope)

    # apply RoPE to queries at position new_len‑1
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                         # (drope,)
    sin_q = sin_tbl[q_pos]                         # (drope,)
    q = q_up * cos_q + _rotate_half(q_up) * sin_q  # (B, nh, drope)

    # --------------------------------------------------------------
    # 3️⃣ Gather K (rope part) & V (latent part) from the cache
    # --------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]          # (B, L, dkv + drope)
    k = kv_all[..., dkv:]                           # (B, L, drope)  – keys (already RoPE‑ed)
    v = kv_all[..., :dkv]                           # (B, L, dkv)    – latent values

    # --------------------------------------------------------------
    # 4️⃣ Triton kernel → weighted sum of V (shape B×nh×dkv)
    # --------------------------------------------------------------
    out_latent = torch.empty((bs, nh, dkv), dtype=torch.bfloat16, device=x.device)

    # grid: one program per (batch, head‑tile)
    grid = lambda meta: (bs * triton.cdiv(nh, meta["HEADS_PER_BLOCK"]),)

    scale = 1.0 / math.sqrt(drope)

    _attn_latent_kernel[grid](
        # pointers
        q, k, v, out_latent,
        # strides
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out_latent.stride(0), out_latent.stride(1), out_latent.stride(2),
        # compile‑time constants
        bs, nh, drope, dkv, scale,
        # tuning constants
        32,            # HEADS_PER_BLOCK – kept moderate to stay within registers
        1024,          # BLOCK_K (keys/values processed per iteration)
        # runtime argument: current KV length
        new_len
    )

    # --------------------------------------------------------------
    # 5️⃣ Per‑head V‑projection (latent → head‑dim)
    # --------------------------------------------------------------
    global _cached_wV_T
    if _cached_wV_T is None:
        # wUKV: ((nope+dv)*nh, dkv)  →  (nh, dkv, dv) after transpose
        _cached_wV_T = wUKV.view(nh, dv, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    # einsum does a batched matmul for each head:
    #   out_latent  : (B, nh, dkv)
    #   _cached_wV_T: (nh, dkv, dv)
    # → y_head : (B, nh, dv)
    y_head = torch.einsum('bhd, hdk -> bhk', out_latent, _cached_wV_T)

    # --------------------------------------------------------------
    # 6️⃣ Final output projection (all heads mixed together)
    # --------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)      # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                # (B, dim)
    out = out.unsqueeze(1)                         # (B, 1, dim)

    return out, kv_cache.data

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
    bs   = config.batch_size
    nh   = config.n_heads
    dim  = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope  = config.qk_rope_head_dim
    dv    = config.v_head_dim

    # --------------------------------------------------------------
    # Weight tensors (already on the correct device & dtype)
    # --------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
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
        out, new_kv = _fast_forward_multihead_optimized(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin,
        )
        # kv_cache has been updated in‑place inside the fast‑path
        return out, new_kv

    # --------------------------------------------------------------
    # General case – fallback to compiled reference implementation
    # --------------------------------------------------------------
    # (unchanged from the original reference – retained for correctness)
    global _compiled_forward
    if '_compiled_forward' not in globals() or _compiled_forward is None:
        # Build compiled fallback (same as the reference implementation)
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

        _compiled_forward = torch.compile(
            _inner,
            backend="inductor",
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
    # --------------------------------------------------------------
    # Call compiled fallback
    # --------------------------------------------------------------
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

    return out, kv_cache.data