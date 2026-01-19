### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###


# --------------------------------------------------------------
# Helper utilities (identical to the reference)
# --------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half (used by RoPE)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# --------------------------------------------------------------
# Global cache for rotary tables (cos / sin)
# --------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)   bfloat16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)   bfloat16


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
# Triton kernel – attention + per‑head value projection (fast path)
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_vhead_kernel_opt(
    # --------------------------------------------------------------
    # Pointers
    # --------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bfloat16 – already rotated
    K_ptr,                # (B, L, Dq)                 bfloat16 – already rotated
    V_ptr,                # (B, L, Dv_lat)             bfloat16
    wV_T_ptr,             # (H, Dv_lat, Dv)            bfloat16
    Y_ptr,                # (B, H, Dv)                 bfloat16

    # --------------------------------------------------------------
    # Strides
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,   # Q    (B, H, Dq)
    stride_k_batch, stride_k_len , stride_k_dim,   # K    (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,   # V    (B, L, Dv_lat)

    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,  # wV_T (H, Dv_lat, Dv)

    stride_y_batch, stride_y_head, stride_y_dv,           # Y    (B, H, Dv)

    # --------------------------------------------------------------
    # Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    L: tl.constexpr,          # current KV length (after insert)
    Dq: tl.constexpr,         # rope dimension (e.g. 64)
    Dv_lat: tl.constexpr,     # kv‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value‑dim (e.g. 128)
    scale: tl.constexpr,      # 1/sqrt(Dq)  (qk_nope_head_dim==0)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_DV_LAT: tl.constexpr,
):
    """
    Optimised attention kernel for the case d_nope == 0.
    * Keys (K) and Values (V) are already RoPE‑rotated.
    * The kernel shares K and V loads across the heads in a block using shared memory.
      This removes the O(heads) duplication present in the naïve implementation.
    * After the soft‑max accumulation we project the latent vector to per‑head
      output space (size Dv) inside the kernel (still cheap compared to the
      attention work).
    """
    pid = tl.program_id(0)

    # --------------------------------------------------------------
    # Tile indexing (batch + head‑tile)
    # --------------------------------------------------------------
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # head‑tile inside the batch
    head_start = tile * HEADS_PER_BLOCK             # first head this program handles

    # --------------------------------------------------------------
    # Helpers – thread identifiers
    # --------------------------------------------------------------
    head_idx = tl.arange(0, HEADS_PER_BLOCK)                     # (HPB,)
    head_mask = (head_start + head_idx) < H

    # --------------------------------------------------------------
    # Load Q for the heads handled by this program (already rotated)
    # --------------------------------------------------------------
    offs_q = (
        b * stride_q_batch
        + (head_start + head_idx)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )                                                            # (HPB, Dq)
    q = tl.load(Q_ptr + offs_q,
                 mask=head_mask[:, None],
                 other=0.0)                                   # (HPB, Dq)

    # --------------------------------------------------------------
    # Allocate per‑head accumulators (for stable soft‑max)
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HPB,)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (HPB,)
    # latent accumulator (partial sums of V weighted by exp‑score)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dv_lat], dtype=tl.bfloat16)

    # --------------------------------------------------------------
    # Shared memory tiles
    # --------------------------------------------------------------
    #   K tile : (BLOCK_K, Dq)  –  fits in shared memory (≤ 64 KB)
    #   V tile : (BLOCK_K, TILE_DV_LAT) – we stream V in chunks along the
    #                                    latent dimension.
    sK = tl.shared([BLOCK_K, Dq], dtype=tl.bfloat16)
    sV = tl.shared([BLOCK_K, TILE_DV_LAT], dtype=tl.bfloat16)

    # --------------------------------------------------------------
    # Main loop over key‑value blocks
    # --------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)          # (BLOCK_K,)
        k_mask = cur_k < L

        # ----------------------------------------------------------
        # Load K block (once, shared across heads)
        # ----------------------------------------------------------
        # Each thread loads a *strip* of the K‑tile.
        #   rows_per_thread = ceil(BLOCK_K / HEADS_PER_BLOCK)
        rows_per_thread = (BLOCK_K + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
        row_start = tl.program_id(0) * rows_per_thread   # this is *b* × num_head_tiles + tile,
                                                         # but we only need a unique id per thread:
                                                         # head_idx already provides 0‑…‑HEADS_PER_BLOCK‑1.
        row = row_start + head_idx                      # (HPB,)

        # Guard against out‑of‑range rows
        k_row_valid = row < BLOCK_K
        k_col = tl.arange(0, Dq)                        # (Dq,)

        # Compute global offsets for the loaded K elements
        offs_k = (
            b * stride_k_batch
            + (cur_k[:, None] + row[:, None])[k_row_valid, :] * stride_k_len
            + k_col[None, :] * stride_k_dim
        )
        # Load the valid rows; out‑of‑range rows are set to 0.0.
        k_tile = tl.load(K_ptr + offs_k,
                         mask=k_row_valid[:, None] & k_mask[:, None],
                         other=0.0)                     # (HPB, Dq)  – per‑thread row
        # Store into shared memory (broadcast across heads)
        tl.store(sK + (row[:, None] * Dq + k_col[None, :]),
                 k_tile,
                 mask=k_row_valid[:, None])

        tl.broadcast(sK, BLOCK_K * Dq)   # make sure the entire tile is visible

        # ----------------------------------------------------------
        # Load V block (once, shared across heads) – streamed in tile‑wise
        # ----------------------------------------------------------
        # We tile over the latent dimension (Dv_lat) to keep shared mem < 64 KB.
        for dv_start in range(0, Dv_lat, TILE_DV_LAT):
            cur_dv = dv_start + tl.arange(0, TILE_DV_LAT)          # (TILE_DV_LAT,)
            dv_mask = cur_dv < Dv_lat

            # Load V slice for the current dv‑tile.
            # Re‑use the same `row` logic as for K.
            offs_v = (
                b * stride_v_batch
                + (cur_k[:, None] + row[:, None])[k_row_valid, :] * stride_v_len
                + cur_dv[None, :] * stride_v_dim
            )
            v_tile = tl.load(V_ptr + offs_v,
                             mask=k_row_valid[:, None] & k_mask[:, None] & dv_mask[None, :],
                             other=0.0)                     # (HPB, TILE_DV_LAT)

            tl.store(sV + (row[:, None] * TILE_DV_LAT + cur_dv[None, :]),
                     v_tile,
                     mask=k_row_valid[:, None] & dv_mask[None, :])

            tl.broadcast(sV, BLOCK_K * TILE_DV_LAT)   # make the whole tile visible

            # ------------------------------------------------------
            # Compute raw scores = q ⋅ kᵀ      (HPB, BLOCK_K)
            # ------------------------------------------------------
            # q : (HPB, Dq)
            # kᵀ: (Dq, BLOCK_K)  – we read from shared memory
            k_shared = tl.load(sK + tl.arange(0, BLOCK_K)[:, None] * Dq + tl.arange(0, Dq)[None, :])
            # (BLOCK_K, Dq)
            score = tl.dot(q, tl.trans(k_shared))               # (HPB, BLOCK_K) bf16
            score_f32 = tl.cast(score, tl.float32) * scale

            # ------------------------------------------------------
            # Stable‑softmax bookkeeping
            # ------------------------------------------------------
            block_max = tl.max(score_f32, axis=1)               # (HPB,)
            new_max = tl.maximum(max_score, block_max)          # (HPB,)

            scale_factor = tl.exp(max_score - new_max)          # (HPB,)
            sum_exp = sum_exp * scale_factor
            latent_acc = latent_acc * tl.cast(scale_factor, tl.bfloat16)[:, None]

            exp_score_f32 = tl.exp(score_f32 - new_max[:, None])    # (HPB, BLOCK_K)
            exp_score = tl.cast(exp_score_f32, tl.bfloat16)         # bf16

            sum_exp = sum_exp + tl.sum(exp_score, axis=1)           # (HPB,)

            # ------------------------------------------------------
            # Load the V tile (already in shared memory) and accumulate
            # ------------------------------------------------------
            v_shared = tl.load(
                sV + tl.arange(0, BLOCK_K)[:, None] * TILE_DV_LAT
                + cur_dv[None, :],
                mask=k_mask[:, None] & dv_mask[None, :],
                other=0.0)                                 # (BLOCK_K, TILE_DV_LAT)

            # latent_acc[:, dv_start:dv_start+TILE_DV_LAT] += exp_scoreᵀ ⋅ v_tile
            # (HPB, BLOCK_K)ᵀ @ (BLOCK_K, TILE_DV_LAT) -> (HPB, TILE_DV_LAT)
            latent_acc = tl.where(
                dv_mask[None, :],
                latent_acc + tl.dot(exp_score, v_shared),
                latent_acc
            )

            max_score = new_max

    # --------------------------------------------------------------
    # Normalise the latent accumulator (divide by Σexp)
    # --------------------------------------------------------------
    latent = latent_acc / tl.cast(sum_exp[:, None], tl.bfloat16)   # (HPB, Dv_lat)

    # --------------------------------------------------------------
    # Project latent → per‑head output (size Dv) – use the weight matrix
    # --------------------------------------------------------------
    # Load the wV_T sub‑matrix for the heads handled by this program.
    # Shape: (HEADS_PER_BLOCK, Dv_lat, Dv)
    offs_wV_T = (
        (head_start + head_idx)[:, None, None] * stride_wV_T_head
        + tl.arange(0, Dv_lat)[None, :, None] * stride_wV_T_lat
        + tl.arange(0, Dv)[None, None, :] * stride_wV_T_out
    )
    wV_block = tl.load(wV_T_ptr + offs_wV_T,
                       mask=head_mask[:, None, None],
                       other=0.0)                     # (HPB, Dv_lat, Dv)

    # y_head = latent @ wV_T   (batched matmul)
    # latent : (HPB, Dv_lat)   → (HPB, 1, Dv_lat)
    # wV_T   : (HPB, Dv_lat, Dv)
    y_head = tl.sum(latent[:, :, None] * wV_block, axis=1)      # (HPB, Dv)

    # --------------------------------------------------------------
    # Store per‑head outputs
    # --------------------------------------------------------------
    offs_y = (
        b * stride_y_batch
        + (head_start + head_idx)[:, None] * stride_y_head
        + tl.arange(0, Dv)[None, :] * stride_y_dv
    )
    tl.store(Y_ptr + offs_y,
             y_head,
             mask=head_mask[:, None])


# ----------------------------------------------------------------------
# Fast‑path implementation (d_nope == 0) – now uses the optimiser kernel
# ----------------------------------------------------------------------
def _fast_forward_multihead_opt(
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
    Fast‑path where qk_nope_head_dim == 0.
    All heavy work (attention + per‑head value projection) is performed by a
    highly‑optimised Triton kernel that shares K/V loads across heads.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim

    # --------------------------------------------------------------
    # 1️⃣ Down‑projection (Q) and KV down‑projection (raw KV)
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                     # (B, Dim)
    q_lora  = F.linear(x2, wDQ)           # (B, dq)
    kv_lora = F.linear(x2, wDKV)          # (B, dkv + drope)

    # --------------------------------------------------------------
    # 2️⃣ KV‑cache update: store latent V + *rotated* rope key
    # --------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora[:, :dkv]      # (B, dkv)
    rope_raw_new   = kv_lora[:, dkv:]     # (B, drope)

    # rotate the new key according to its absolute position (cur_len)
    cos_k = cos_tbl[cur_len]                     # (drope,)
    sin_k = sin_tbl[cur_len]                     # (drope,)
    rope_key_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k   # (B, drope)

    # write into cache (store rotated key)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_key_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 3️⃣ Up‑project Q and apply RoPE (single token)
    # --------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)               # (B, nh*drope)
    q_up = q_up.view(bs, nh, drope)           # (B, nh, drope)

    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                     # (drope,)
    sin_q = sin_tbl[q_pos]                     # (drope,)
    q_rope = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # --------------------------------------------------------------
    # 4️⃣ Gather KV from cache (latent + already‑rotated rope part)
    # --------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]          # (B, L, dkv+drope)
    k_rope = kv_all[..., dkv:]                      # (B, L, drope) – already rotated
    v_latent = kv_all[..., :dkv]                    # (B, L, dkv)

    # --------------------------------------------------------------
    # 5️⃣ Value‑projection matrix (transposed)
    # --------------------------------------------------------------
    # wUKV shape: ((d_nope + dv) * nh , dkv)  ; d_nope==0
    # → (nh, dv, dkv) → permute → (nh, dkv, dv)
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 6️⃣ Allocate output buffer for each head
    # --------------------------------------------------------------
    y_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # --------------------------------------------------------------
    # 7️⃣ Compute strides (contiguous tensors)
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim = q_rope.stride()
    stride_k_batch, stride_k_len , stride_k_dim  = k_rope.stride()
    stride_v_batch, stride_v_len, stride_v_dim   = v_latent.stride()
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out = wV_T.stride()
    stride_y_batch, stride_y_head, stride_y_dv = y_head.stride()

    # --------------------------------------------------------------
    # 8️⃣ Triton launch configuration
    # --------------------------------------------------------------
    # tuned for the 128‑head / 128‑batch configuration
    HEADS_PER_BLOCK = 32
    BLOCK_K = 512                # production‑ready block size (covers most pre‑fills)
    TILE_DV_LAT = 64             # V‑latent tiling – 64 bfloat16 → 128 bytes per row

    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    scale = 1.0 / math.sqrt(drope)    # d_nope == 0 → scale = 1/√(drope)

    _triton_attn_vhead_kernel_opt[grid](
        # pointers
        q_rope,
        k_rope,
        v_latent,
        wV_T,
        y_head,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len, stride_k_dim,
        stride_v_batch, stride_v_len, stride_v_dim,
        stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
        stride_y_batch, stride_y_head, stride_y_dv,
        # compile‑time constants
        bs, nh, new_len, drope, dkv, dv,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, TILE_DV_LAT,
        # launch config
        num_warps=8, num_stages=5,
    )

    # --------------------------------------------------------------
    # 9️⃣ Final linear projection (per‑head outputs → model dimension)
    # --------------------------------------------------------------
    y_head_flat = y_head.view(bs, nh * dv)          # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                 # (B, dim)
    out = out.unsqueeze(1)                          # (B, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Compiled fallback for the general case (d_nope > 0)
# ----------------------------------------------------------------------
_compiled_forward: torch._dynamo.eval_frame.OptimizedModule = None


def _build_compiled_forward():
    """Compiled reference implementation (handles d_nope > 0)."""
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
        # --------------------------------------------------------------
        # Reference implementation (identical to the original module)
        # --------------------------------------------------------------
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
        kv_latent   = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

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
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)

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
    # Extract scalar config values
    # --------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim

    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight        # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh , dq)
    wUKV = config.KV_proj_up_weight          # ((d_nope + dv) * nh , dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

    # --------------------------------------------------------------
    # Build / fetch RoPE tables (cached globally)
    # --------------------------------------------------------------
    global _cached_cos, _cached_sin
    if (_cached_cos is None) or (_cached_cos.shape[0] < config.max_seq_len):
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # --------------------------------------------------------------
    # Fast‑path – d_nope == 0  (the very common configuration)
    # --------------------------------------------------------------
    if d_nope == 0:
        out, new_kv = _fast_forward_multihead_opt(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin,
        )
        # update KVCache state (kv_cache.data already contains the new cached data)
        kv_cache.data = new_kv
        return out, kv_cache.data

    # --------------------------------------------------------------
    # General case – use compiled reference implementation
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
    # update KVCache state with new length
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    # output already has shape [B, 1, Dim]
    return out, kv_cache.data