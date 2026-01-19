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
_cached_wO_T: torch.Tensor = None         # (n_heads, v_head_dim, dim)            bfloat16

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
# Triton kernel – fused attention (stable soft‑max) + per‑head V‑proj +
# final output‑proj (wO).  Importantly, the heavy wO matrix is loaded **once**
# after the full attention reduction, cutting the memory traffic by a factor
# equal to the number of KV‑blocks.
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_fused_kernel_opt(
    # --------------------------------------------------------------
    # Pointer arguments
    # --------------------------------------------------------------
    Q_ptr,               # (B, H, Dim)  – also used as output buffer  bf16
    K_ptr,               # (B, L, Dq)                                 bf16
    V_ptr,               # (B, L, Dkv)                               bf16
    wV_T_ptr,            # (H, Dkv, Dv)                              bf16
    wO_T_ptr,            # (H, Dv, Dim)                              bf16
    # --------------------------------------------------------------
    # Strides
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len,  stride_k_dim,
    stride_v_batch, stride_v_len,  stride_v_dim,
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
    stride_wO_T_head, stride_wO_T_mid, stride_wO_T_out,
    # --------------------------------------------------------------
    # Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    Dq: tl.constexpr,         # rope dim (e.g. 64)
    Dkv: tl.constexpr,        # kv‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value dim (e.g. 128)
    Dim: tl.constexpr,        # model dim (e.g. 7168)
    scale: tl.constexpr,      # 1/sqrt(Dq)

    # --------------------------------------------------------------
    # Tiling parameters (tuned for the reference workload)
    # --------------------------------------------------------------
    HEADS_PER_BLOCK: tl.constexpr,   # heads processed together
    BLOCK_K: tl.constexpr,           # KV‑length chunk
    BLOCK_DV: tl.constexpr,          # latent‑dim chunk

    # --------------------------------------------------------------
    # Runtime argument
    # --------------------------------------------------------------
    L: tl.int32,               # current KV length (dynamic)
):
    """
    Fully fused flash‑attention kernel.

    * Computes a numerically‑stable softmax over Q·Kᵀ.
    * Accumulates the weighted latent values (V) **once** per KV‑block,
      projects them (V → dv) with wV_T and stores the intermediate result.
    * After the full sequence is processed a single load of wO_T produces the
      final model‑dim output, which is written back into Q_ptr.
    """
    pid = tl.program_id(0)                         # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # head‑tile inside batch
    head_start = tile * HEADS_PER_BLOCK              # first head of this program

    # --------------------------------------------------------------
    # 1️⃣ Load queries for this tile (shape: HEADS_PER_BLOCK × Dq)
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
                 other=0.0)               # (HEADS_PER_BLOCK, Dq)  bf16

    # --------------------------------------------------------------
    # 2️⃣ Buffers for stable softmax & result accumulation
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HEADS_PER_BLOCK)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)           # (HEADS_PER_BLOCK)

    # accumulator for the **projected** latent values (HEADS_PER_BLOCK × Dv)
    y_acc      = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)

    # ------------------------------------------------------------------
    # 3️⃣ Main loop over KV blocks (stable soft‑max)
    # ------------------------------------------------------------------
    for start_k in tl.range(0, L, BLOCK_K, num_stages=4):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)          # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- load K -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0,
                          cache_modifier='CA')               # (BLOCK_K, Dq)  bf16

        # ----- Q·Kᵀ -------------------------------------------------
        # (HEADS_PER_BLOCK, Dq) × (Dq, BLOCK_K) → (HEADS_PER_BLOCK, BLOCK_K)
        prod = tl.dot(q, tl.permute(k_block, (1, 0)))            # bf16
        score_f32 = tl.cast(prod, tl.float32) * scale           # fp32

        # ----- stable‑softmax bookkeeping ----------------------------
        block_max = tl.max(score_f32, axis=1)                   # (HEADS_PER_BLOCK)
        new_max   = tl.maximum(max_score, block_max)            # (HEADS_PER_BLOCK)

        # rescale previous sums & projected values
        exp_factor = tl.exp(max_score - new_max)                # (HEADS_PER_BLOCK)
        sum_exp    = sum_exp * exp_factor
        y_acc      = y_acc * exp_factor[:, None]

        # now the new contributions
        exp_score = tl.exp(score_f32 - new_max[:, None])        # (HEADS_PER_BLOCK, BLOCK_K)
        sum_exp   = sum_exp + tl.sum(exp_score, axis=1)         # (HEADS_PER_BLOCK)

        # ------------------------------------------------------------------
        # 4️⃣ Accumulate V‑latent → (HEADS × Dv)  (still in FP32)
        # ------------------------------------------------------------------
        for start_d in tl.range(0, Dkv, BLOCK_DV, num_stages=2):
            cur_d = start_d + tl.arange(0, BLOCK_DV, tl.int32)   # (BLOCK_DV,)
            d_mask = cur_d < Dkv

            # ----- load V block ---------------------------------------
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

            # ----- weighted sum over keys (exp_scoreᵀ·V) -------------
            # X = exp_score @ v_fp32  → (HEADS_PER_BLOCK, BLOCK_DV)
            X = tl.dot(exp_score, v_fp32)                        # fp32

            # ----- load wV_T slice ------------------------------------
            offs_wV = (
                (head_start + head_range)[:, None, None] * stride_wV_T_head
                + cur_d[None, :, None] * stride_wV_T_lat
                + tl.arange(0, Dv, tl.int32)[None, None, :] * stride_wV_T_out
            )
            wV_block = tl.load(wV_T_ptr + offs_wV,
                               mask=head_valid[:, None] & d_mask[None, :],
                               other=0.0)                               # (HEADS_PER_BLOCK, BLOCK_DV, Dv) bf16
            wV_block_f32 = tl.cast(wV_block, tl.float32)          # fp32

            # ----- project to per‑head dv (X @ wV_T) -----------------
            #   X : (HEADS_PER_BLOCK, BLOCK_DV)
            #   wV_block_f32 : (HEADS_PER_BLOCK, BLOCK_DV, Dv)
            #   ⇒ Y_slice : (HEADS_PER_BLOCK, Dv)
            X_exp = X[:, :, None]                                 # (HEADS_PER_BLOCK, BLOCK_DV, 1)
            Y_slice = tl.sum(X_exp * wV_block_f32, axis=1)        # (HEADS_PER_BLOCK, Dv)

            # ----- accumulate into the per‑head dv buffer -----------
            y_acc = y_acc + Y_slice

        # --------------------------------------------------------------
        # End of KV‑block
        # --------------------------------------------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # 5️⃣ Final output projection (single load of wO_T)
    # ------------------------------------------------------------------
    offs_wO = (
        (head_start + head_range)[:, None, None] * stride_wO_T_head
        + tl.arange(0, Dv, tl.int32)[None, :, None] * stride_wO_T_mid
        + tl.arange(0, Dim, tl.int32)[None, None, :] * stride_wO_T_out
    )
    wO_block = tl.load(wO_T_ptr + offs_wO,
                       mask=head_valid[:, None],
                       other=0.0)                               # (HEADS_PER_BLOCK, Dv, Dim) bf16
    wO_block_f32 = tl.cast(wO_block, tl.float32)                # fp32

    # Z = y_acc @ wO_T  → (HEADS_PER_BLOCK, Dim)
    y_exp = y_acc[:, :, None]                                   # (HEADS_PER_BLOCK, Dv, 1)
    Z = tl.sum(y_exp * wO_block_f32, axis=1)                    # (HEADS_PER_BLOCK, Dim)

    # ------------------------------------------------------------------
    # 6️⃣ Normalisation (divide by the softmax denominator)
    # ------------------------------------------------------------------
    out = Z / sum_exp[:, None]                                   # (HEADS_PER_BLOCK, Dim)

    # ------------------------------------------------------------------
    # 7️⃣ Store final model‑dim output (re‑using the Q buffer)
    # ------------------------------------------------------------------
    offs_out = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dim)[None, :] * stride_q_dim
    )
    tl.store(Q_ptr + offs_out,
             tl.cast(out, tl.bfloat16),
             mask=head_valid[:, None])

# ----------------------------------------------------------------------
# Fast‑path (qk_nope_head_dim == 0) – uses the new fused kernel
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
    The heavy wO matrix is now loaded only once per forward.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim
    dim = config.dim

    # --------------------------------------------------------------
    # 1️⃣ Down‑projection + KV‑cache update (with RoPE baked‑in)
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                     # (B, Dim)
    kv_lora = F.linear(x2, wDKV)          # (B, dkv + drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora[:, :dkv]               # (B, dkv)
    rope_raw_new  = kv_lora[:, dkv:]               # (B, drope)

    # RoPE for the key at position `cur_len` (in‑place, no extra tensor)
    cos_k = cos_tbl[cur_len]                       # (drope,)
    sin_k = sin_tbl[cur_len]                       # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write into cache (latent first, then rope‑rotated part)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 2️⃣ Q‑down + Q‑up projection (low‑rank)  → (B, nh, drope)
    # --------------------------------------------------------------
    q_lora = F.linear(x2, wDQ)                     # (B, q_lora_rank)
    q = F.linear(q_lora, wUQ)                      # (B, nh * drope)
    q = q.view(bs, nh, drope)                     # (B, nh, drope)

    # --------------------------------------------------------------
    # 3️⃣ Apply RoPE to queries (in‑place)
    # --------------------------------------------------------------
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                         # (drope,)
    sin_q = sin_tbl[q_pos]                         # (drope,)
    q = q * cos_q + _rotate_half(q) * sin_q        # (B, nh, drope)

    # --------------------------------------------------------------
    # 4️⃣ Prepare pointers / strides for Triton kernel
    # --------------------------------------------------------------
    # The output buffer also serves as the Q‑pointer expected by the kernel
    q_out = torch.empty((bs, nh, dim), dtype=torch.bfloat16, device=x.device)
    q_out.copy_(q)   # copy the (already RoPE‑ed) queries into the buffer

    # K pointer – the rope‑rotated keys stored in the cache
    kv_all = kv_cache.data[:, :new_len, :]               # (B, L, dkv + drope)
    k = kv_all[..., dkv:]                               # (B, L, drope)

    # V‑latent pointer (the part that does NOT receive RoPE)
    v_latent = kv_all[..., :dkv]                        # (B, L, dkv)

    # --------------------------------------------------------------
    # 5️⃣ Cache & reshape the per‑head projection weights (once)
    # --------------------------------------------------------------
    global _cached_wV_T, _cached_wO_T
    if _cached_wV_T is None:
        # wUKV: ((dv) * nh, dkv) → reshape → (nh, dv, dkv) → transpose → (nh, dkv, dv)
        _cached_wV_T = wUKV.view(nh, dv, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    if _cached_wO_T is None:
        # wO: (dim, nh*dv) → reshape → (nh, dv, dim)
        _cached_wO_T = wO.view(dim, nh, dv).transpose(1, 0).contiguous()   # (nh, dv, dim)

    # --------------------------------------------------------------
    # 6️⃣ Launch the fused attention+output‑projection kernel
    # --------------------------------------------------------------
    scale = 1.0 / math.sqrt(drope)          # Dq == drope

    # Grid: one program per (batch, head‑tile)
    grid = lambda meta: (bs * triton.cdiv(nh, meta["HEADS_PER_BLOCK"]),)

    _triton_attn_fused_kernel_opt[grid](
        # pointers
        q_out, k, v_latent,
        _cached_wV_T,
        _cached_wO_T,
        # strides
        q_out.stride(0), q_out.stride(1), q_out.stride(2),   # Q‑/output strides
        k.stride(0), k.stride(1), k.stride(2),
        v_latent.stride(0), v_latent.stride(1), v_latent.stride(2),
        _cached_wV_T.stride(0), _cached_wV_T.stride(1), _cached_wV_T.stride(2),
        _cached_wO_T.stride(0), _cached_wO_T.stride(1), _cached_wO_T.stride(2),
        # compile‑time constants (B, H, Dq, Dkv, Dv, Dim, scale)
        bs, nh, drope, dkv, dv, dim, scale,
        # kernel‑tuning constants (empirically good for the reference shape)
        32,            # HEADS_PER_BLOCK
        1024,          # BLOCK_K
        128,           # BLOCK_DV
        # runtime argument: current KV length (includes the newly added token)
        new_len
    )

    # --------------------------------------------------------------
    # 7️⃣ Reshape fused output back to (B, 1, Dim) and return.
    # --------------------------------------------------------------
    out = q_out.view(bs, 1, dim)                     # (B, 1, Dim)
    return out, kv_cache.data


# ----------------------------------------------------------------------
# Compiled fallback (qk_nope_head_dim > 0) – unchanged from reference
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
    # Fast‑path – d_nope == 0 (the common configuration)
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