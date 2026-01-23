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
# Global caches – allocated only once (shared among all calls)
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None      # (max_seq_len, rope_dim)   bfloat16
_cached_sin: torch.Tensor = None      # (max_seq_len, rope_dim)   bfloat16
# ----------------------------------------------------------------------
#  Utility helpers
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half‑rotate used by RoPE (identical to the reference implementation)."""
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
    idx = pos * theta                                        # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                      # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)

# ----------------------------------------------------------------------
# Triton kernel – fused attention + final 𝑤ₒ projection
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_fused_opt(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bf16
    K_ptr,                # (B, L, Dq)                 bf16   (already RoPE‑rotated)
    V_ptr,                # (B, L, Dkv)                bf16
    wV_T_ptr,             # (H, Dkv, Dv)               bf16
    wO_ptr,               # (Dim, H*Dv)                bf16
    out_ptr,              # (B, Dim)                   bf16
    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len,  stride_k_dim,
    stride_v_batch, stride_v_len,  stride_v_dim,
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
    stride_wO_out, stride_wO_in,               # wO: (Dim, H*Dv)
    stride_out_batch, stride_out_dim,
    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,        # batch size
    H: tl.constexpr,        # total heads
    Dq: tl.constexpr,       # rope dimension (e.g. 64)
    Dkv: tl.constexpr,      # KV‑lora rank (e.g. 512)
    Dv: tl.constexpr,       # per‑head value dimension (e.g. 128)
    Dim: tl.constexpr,      # model dimension (e.g. 7168)
    scale: tl.constexpr,    # 1 / sqrt(Dq)  (fp32)
    # ------------------------------------------------------------------
    # Tiling parameters
    # ------------------------------------------------------------------
    HEADS_PER_BLOCK: tl.constexpr,   # #heads processed by one program
    BLOCK_K: tl.constexpr,           # #keys processed per iteration
    BLOCK_DKV: tl.constexpr,         # #latent (dkv) processed per inner‑loop
    BLOCK_DV: tl.constexpr,          # tile size for the V‑projection dimension
    BLOCK_OUT: tl.constexpr,         # tile size for the final 𝑤ₒ‑projection
    # ------------------------------------------------------------------
    # Runtime arguments
    # ------------------------------------------------------------------
    L: tl.int32,               # current KV length (dynamic)
):
    """
    Fused attention + final projection.
    1️⃣  Compute numerically‑stable softmax (Q·Kᵀ) over the KV sequence.
    2️⃣  Accumulate the latent value aggregation Σ exp·V (still in fp32).
    3️⃣  Normalise the latent tensor.
    4️⃣  Apply the per‑head value‑projection (latent·wVᵀ).
    5️⃣  Fuse the final model‑dim projection (out += (latent·wVᵀ)·wₒ) – no
        intermediate write‑back of the head‑wise y tensor.
    """
    pid = tl.program_id(0)                     # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    batch_id = pid // num_head_tiles
    tile_id  = pid % num_head_tiles
    head_start = tile_id * HEADS_PER_BLOCK

    # ------------------------------------------------------------------
    # 1️⃣ Load Q tile (HEADS_PER_BLOCK × Dq) – already RoPE‑rotated
    # ------------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_valid = head_start + head_range < H

    offs_q = (
        batch_id * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    Q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)
    Q = tl.cast(Q, tl.float32)                # (HEADS_PER_BLOCK, Dq)

    # ------------------------------------------------------------------
    # 2️⃣ Allocate numerically‑stable softmax state & latent accumulator
    # ------------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HEADS_PER_BLOCK,)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (HEADS_PER_BLOCK,)
    latent    = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)    # (HEADS_PER_BLOCK, Dkv)

    # ------------------------------------------------------------------
    # 3️⃣ Main loop over KV blocks
    # ------------------------------------------------------------------
    for start_k in tl.range(0, L, BLOCK_K, num_stages=4):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)           # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- Load K (BLOCK_K × Dq) – already RoPE‑rotated ---------------
        offs_k = (
            batch_id * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        K = tl.load(K_ptr + offs_k,
                     mask=k_mask[:, None],
                     other=0.0,
                     cache_modifier='CA')
        K = tl.cast(K, tl.float32)                 # (BLOCK_K, Dq)

        # ----- Compute raw scores Q·Kᵀ ------------------------------------
        scores = tl.dot(Q, tl.permute(K, (1, 0))) * scale   # (HEADS_PER_BLOCK, BLOCK_K)

        # ----- Stable soft‑max update ------------------------------------
        block_max = tl.max(scores, axis=1)                       # (HEADS_PER_BLOCK,)
        new_max   = tl.maximum(max_score, block_max)              # (HEADS_PER_BLOCK,)

        # Rescale previous contributions
        exp_factor = tl.exp(max_score - new_max)                  # (HEADS_PER_BLOCK,)
        sum_exp   = sum_exp * exp_factor
        latent    = latent * exp_factor[:, None]

        # New block contribution
        exp_score = tl.exp(scores - new_max[:, None])             # (HEADS_PER_BLOCK, BLOCK_K)
        sum_exp   = sum_exp + tl.sum(exp_score, axis=1)           # (HEADS_PER_BLOCK,)

        # ----- Accumulate latent values  (exp_scoreᵀ @ V_block) ---------
        for d_start in tl.range(0, Dkv, BLOCK_DKV):
            cur_d = d_start + tl.arange(0, BLOCK_DKV, tl.int32)   # (BLOCK_DKV,)
            d_mask = cur_d < Dkv

            offs_v = (
                batch_id * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            V = tl.load(V_ptr + offs_v,
                        mask=k_mask[:, None] & d_mask[None, :],
                        other=0.0,
                        cache_modifier='CA')
            V = tl.cast(V, tl.float32)

            # latent contribution for this DKV‑tile
            # (HEADS_PER_BLOCK, BLOCK_K) @ (BLOCK_K, BLOCK_DKV) -> (HEADS_PER_BLOCK, BLOCK_DKV)
            X = tl.dot(exp_score, V)

            # accumulate
            mask_tile = head_valid[:, None] & d_mask[None, :]
            latent = tl.where(mask_tile,
                              latent + X,
                              latent)

    # ------------------------------------------------------------------
    # 4️⃣ Normalise the latent aggregation
    # ------------------------------------------------------------------
    latent = latent / sum_exp[:, None]                # (HEADS_PER_BLOCK, Dkv)

    # ------------------------------------------------------------------
    # 5️⃣ Fuse per‑head value‑projection and final 𝑤ₒ projection
    # ------------------------------------------------------------------
    # out_acc accumulates partial results for (batch, Dim)
    out_acc = tl.zeros([HEADS_PER_BLOCK, BLOCK_OUT], dtype=tl.float32)

    # Loop over output tiles
    for d_out in tl.range(0, Dim, BLOCK_OUT):
        cur_out = d_out + tl.arange(0, BLOCK_OUT, tl.int32)   # (BLOCK_OUT,)
        out_mask = cur_out < Dim

        # ------------------------------------------------------------------
        # Load wO slice for the current output tile.
        # wO layout: (Dim, H*Dv)  where inner dim = head*Dv + dv_idx
        # For a given head h we need the Dv columns that belong to that head.
        # ------------------------------------------------------------------
        # column offset for each head:
        #   col_base[h] = (head_start + h) * Dv
        col_base = (head_start + head_range) * Dv                 # (HEADS_PER_BLOCK,)

        # Build a 2‑D offset: rows = cur_out, cols = col_base + dv_idx
        dv_idx = tl.arange(0, Dv, tl.int32)                       # (Dv,)
        col_offset = col_base[:, None] + dv_idx[None, :]          # (HEADS_PER_BLOCK, Dv)

        # Broadcast row dimension:
        row_offset = d_out + tl.arange(0, BLOCK_OUT, tl.int32)    # (BLOCK_OUT,)
        offs_wO = row_offset[None, :] * stride_wO_out + col_offset[:, :, None] * stride_wO_in
        #   shape -> (HEADS_PER_BLOCK, Dv, BLOCK_OUT)

        wO_tile = tl.load(wO_ptr + offs_wO,
                          mask=head_valid[:, None, None] & out_mask[None, None, :],
                          other=0.0,
                          cache_modifier='CA')
        wO_tile = tl.cast(wO_tile, tl.float32)                     # (HEADS_PER_BLOCK, Dv, BLOCK_OUT)

        # ------------------------------------------------------------------
        # Loop over Dv tiles – fuse (latent·wVᵀ)·wO in one go.
        # ------------------------------------------------------------------
        for d_v_start in tl.range(0, Dv, BLOCK_DV):
            cur_dv = d_v_start + tl.arange(0, BLOCK_DV, tl.int32)   # (BLOCK_DV,)
            dv_mask = cur_dv < Dv

            # ----- Load wVᵀ slice (HEADS_PER_BLOCK, Dkv, BLOCK_DV) -----
            offs_wV = (
                (head_start + head_range)[:, None, None] * stride_wV_T_head
                + tl.arange(0, Dkv)[:, None] * stride_wV_T_lat
                + cur_dv[None, None, :] * stride_wV_T_out
            )
            wV = tl.load(wV_T_ptr + offs_wV,
                         mask=head_valid[:, None, None] & dv_mask[None, None, :],
                         other=0.0,
                         cache_modifier='CA')
            wV = tl.cast(wV, tl.float32)                         # (HEADS_PER_BLOCK, Dkv, BLOCK_DV)

            # ----- Fuse the two matmuls:
            #   (latent (Bk, Dkv) @ wV (Bk, Dkv, Bd)) -> (Bk, Bd)
            #   then (· @ wO (Bk, Bd, Bout)) -> (Bk, Bout)
            # ------------------------------------------------------------
            # Part 1: compute tmp = latent @ wV   (Bk, Bd)
            tmp = tl.dot(latent, wV)                           # (HEADS_PER_BLOCK, BLOCK_DV)

            # Part 2: compute out contribution = tmp @ wO_slice[:, :, :]
            #   wO_slice has shape (Bk, Dv, BLOCK_OUT); we need the slice
            #   that corresponds to the same Dv‑indices we just loaded.
            #   Extract the relevant sub‑matrix:
            wO_sub = wO_tile[:, cur_dv, :]                     # (HEADS_PER_BLOCK, BLOCK_DV, BLOCK_OUT)

            #   Compute dot product per head
            out_tile = tl.dot(tmp, wO_sub)                     # (HEADS_PER_BLOCK, BLOCK_OUT)

            # Accumulate into the output tile
            out_acc = tl.where(out_mask[None, :],
                               out_acc + out_tile,
                               out_acc)

    # ------------------------------------------------------------------
    # 6️⃣ Store the final output (B, Dim)
    # ------------------------------------------------------------------
    offs_out = (
        batch_id * stride_out_batch
        + tl.arange(0, BLOCK_OUT)[None, :] * stride_out_dim
    )
    tl.store(out_ptr + offs_out,
             tl.cast(out_acc, tl.bfloat16),
             mask=out_mask[None, :])


# ----------------------------------------------------------------------
# Fast‑path – common case where the “no‑rope” part is disabled (qk_nope_head_dim == 0)
# ----------------------------------------------------------------------
def _fast_forward_opt(
    cfg: Config,
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
    Fully‑fused forward for the fast‑path (no‑rope part disabled).
    The heavy lifting (attention + final projection) lives in a single Triton
    kernel.
    """
    bs = cfg.batch_size
    nh = cfg.n_heads
    drope = cfg.qk_rope_head_dim      # Dq == drope (no‑rope part is zero)
    dkv   = cfg.kv_lora_rank
    dim   = cfg.dim

    # ------------------------------------------------------------------
    # 1️⃣ Down‑projection + KV‑cache update (still done in PyTorch)
    # ------------------------------------------------------------------
    x_flat = x.squeeze(1)                        # (B, dim)

    kv_lora = F.linear(x_flat, wDKV)             # (B, dkv + drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    # split and write to cache
    kv_latent_new = kv_lora[:, :dkv]              # (B, dkv)
    rope_raw_new   = kv_lora[:, dkv:]             # (B, drope)

    # RoPE for the new key (in‑place)
    cos_k = cos_tbl[cur_len]                      # (drope,)
    sin_k = sin_tbl[cur_len]                      # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # ------------------------------------------------------------------
    # 2️⃣ Query side – down‑/up‑projection + RoPE (still done in PyTorch)
    # ------------------------------------------------------------------
    q_lora = F.linear(x_flat, wDQ)                # (B, dq)
    q_up   = F.linear(q_lora, wUQ)                # (B, nh*drope)
    q_up   = q_up.view(bs, nh, drope)             # (B, nh, drope)

    # RoPE for query (single token, position = new_len‑1)
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                        # (drope,)
    sin_q = sin_tbl[q_pos]                        # (drope,)
    q_rope = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # ------------------------------------------------------------------
    # 3️⃣ Prepare tensors for the Triton kernel
    # ------------------------------------------------------------------
    Q = q_rope.contiguous()                       # (B, nh, drope) – already RoPE‑rotated
    K = kv_cache.data[:, :new_len, dkv:].contiguous()   # (B, L, drope)
    V = kv_cache.data[:, :new_len, :dkv].contiguous()   # (B, L, dkv)

    # ------------------------------------------------------------------
    # 4️⃣ Cache per‑head value‑projection matrix (dkv → dv)
    # ------------------------------------------------------------------
    global _cached_wV_T
    if _cached_wV_T is None:
        # wUKV shape: ((dv) * nh, dkv)
        # reshape → (nh, dv, dkv) → transpose → (nh, dkv, dv)
        _cached_wV_T = wUKV.view(nh, dv, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # 5️⃣ Allocate output buffer (final model output)
    # ------------------------------------------------------------------
    out = torch.empty((bs, dim), dtype=torch.bfloat16, device=x.device)

    # ------------------------------------------------------------------
    # 6️⃣ Launch the fused kernel
    # ------------------------------------------------------------------
    HEADS_PER_BLOCK = 32      # 128 heads → 4 tiles per batch
    BLOCK_K          = 256
    BLOCK_DKV        = 64
    BLOCK_DV         = 64
    BLOCK_OUT        = 32      # tile size for final projection

    scale = 1.0 / math.sqrt(drope)           # Dq == drope

    grid = lambda meta: (bs * triton.cdiv(nh, meta["HEADS_PER_BLOCK"]),)

    _triton_attn_fused_opt[grid](
        # pointers
        Q, K, V,
        _cached_wV_T,
        wO,
        out,
        # strides
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        _cached_wV_T.stride(0), _cached_wV_T.stride(1), _cached_wV_T.stride(2),
        wO.stride(0), wO.stride(1),
        out.stride(0), out.stride(1),
        # compile‑time constants
        bs, nh, drope, dkv, dv, dim, scale,
        # tiling parameters (compile‑time)
        HEADS_PER_BLOCK,
        BLOCK_K,
        BLOCK_DKV,
        BLOCK_DV,
        BLOCK_OUT,
        # runtime argument
        new_len,
        num_warps=4,                        # 4 warps → 128 threads per block
    )

    # ------------------------------------------------------------------
    # 7️⃣ Reshape to the expected (B, 1, Dim) output format
    # ------------------------------------------------------------------
    out = out.unsqueeze(1)                         # (B, 1, Dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Fallback – compile the reference implementation for the general case
# ----------------------------------------------------------------------
_compiled_fwd = None
def _build_fallback():
    """Compile the reference implementation (used when qk_nope_head_dim > 0)."""
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
        # reference implementation – unchanged (exact copy from the prompt)
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
    Entry point used by the benchmark harness.
    """
    global _cached_cos, _cached_sin, _cached_wV_T, _compiled_fwd

    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Extract scalar configuration values (plain Python ints)
    # ------------------------------------------------------------------
    bs   = config.batch_size
    nh   = config.n_heads
    dim  = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope  = config.qk_rope_head_dim
    dv   = config.v_head_dim

    # ------------------------------------------------------------------
    # Weight tensors (already on the correct device & dtype)
    # ------------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    # Ensure RoPE cosine / sine tables are cached (global, reused)
    # ------------------------------------------------------------------
    if _cached_cos is None or _cached_cos.size(0) < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # ------------------------------------------------------------------
    # Fast‑path – common case where the “no‑rope” part is disabled
    # ------------------------------------------------------------------
    if d_nope == 0:
        out, new_kv = _fast_forward_opt(
            config,
            x,
            kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos,
            _cached_sin,
        )
        return out, new_kv

    # ------------------------------------------------------------------
    # General case – fall back to the compiled reference implementation
    # ------------------------------------------------------------------
    if _compiled_fwd is None:
        _compiled_fwd = _build_fallback()

    out, new_kv_data, new_len = _compiled_fwd(
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
    # update KV cache state
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data