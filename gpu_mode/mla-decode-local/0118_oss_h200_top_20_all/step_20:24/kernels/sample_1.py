### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way.
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# Helper: rotate‑half (same as the reference)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ----------------------------------------------------------------------
# Global caches for RoPE tables
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Create (cos, sin) tables for rotary embeddings (bfloat16)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32,
                                        device=device) / half)).to(torch.bfloat16)  # (half,)
    pos = torch.arange(max_seq_len, dtype=torch.int64,
                       device=device).unsqueeze_(1)                       # (max_seq_len, 1)
    idx = pos * theta                                                   # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                                 # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel – multi‑head (tile of heads) fused attention + output projection
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_out_kernel_multihead(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bfloat16
    K_ptr,                # (B, L, Dq)                 bfloat16
    V_ptr,                # (B, L, Dv_lat)             bfloat16   (Dv_lat = kv_lora_rank)
    wV_T_ptr,             # (H, Dv_lat, Dv)            bfloat16
    wO_ptr,               # (Dim, H*Dv)                bfloat16
    Out_ptr,              # (B, Dim)                   bfloat16

    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,   # Q    (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,   # K    (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,   # V    (B, L, Dv_lat)

    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,  # wV_T (H, Dv_lat, Dv)
    stride_wO_out, stride_wO_in,                 # wO   (Dim, H*Dv)

    stride_out_batch, stride_out_dim,             # Out  (B, Dim)

    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    L: tl.constexpr,          # current KV length
    Dq: tl.constexpr,         # rope dimension (e.g. 64)
    Dv_lat: tl.constexpr,     # kv‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value dim (e.g. 128)
    Dim: tl.constexpr,        # model dimension (e.g. 7168)
    scale: tl.constexpr,      # 1/sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,    # how many heads a program processes
    BLOCK_K: tl.constexpr,    # positions processed per iteration (≈64)
    BLOCK_DV: tl.constexpr,   # block size for the value‑latent accumulation (≈32)
):
    """
    Each program processes a tile of `HEADS_PER_BLOCK` heads for a single batch entry.
    The algorithm is the classic stable‑softmax attention:
        1. first pass → per‑head max over the key dimension
        2. second pass → per‑head exp‑sum and weighted latent sum
        3. normalise, project with wV_T and finally accumulate with wO.
    All intermediate memory traffic stays inside registers / shared memory.
    """
    # ------------------------------------------------------------------
    # 0️⃣  Identify batch entry and head‑tile
    # ------------------------------------------------------------------
    pid = tl.program_id(0)                         # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # tile index inside batch
    head_start = tile * HEADS_PER_BLOCK              # first head handled by this program

    # ------------------------------------------------------------------
    # 1️⃣  Load Q‑vectors for the heads in this tile
    # ------------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_valid = head_start + head_range < H          # mask for the last (partial) tile
    # offsets: (HEADS_PER_BLOCK, Dq)
    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)                     # (HEADS_PER_BLOCK, Dq)

    # ------------------------------------------------------------------
    # 2️⃣  First pass – per‑head max over K
    # ------------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)

    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)                # (BLOCK_K,)
        k_mask = cur_k < L

        # K block: (BLOCK_K, Dq)
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0)                     # (BLOCK_K, Dq)

        # prod: (HEADS_PER_BLOCK, BLOCK_K)
        #  q (Hpb, Dq) broadcast against k_block (BK, Dq)
        prod = tl.sum(q[:, None, :] * k_block[None, :, :], axis=2)

        # update max per head
        max_score = tl.maximum(max_score, tl.cast(prod, tl.float32))

    # ------------------------------------------------------------------
    # 3️⃣  Second pass – exp‑sum and weighted latent sum
    # ------------------------------------------------------------------
    sum_exp = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)
    # accumulator for the latent vector (processed block‑wise)
    acc = tl.zeros([HEADS_PER_BLOCK, BLOCK_DV], dtype=tl.bfloat16)

    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)                # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- load K block (same as above) -----
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0)                     # (BLOCK_K, Dq)

        # ----- dot(q, k)  → (HEADS_PER_BLOCK, BLOCK_K) -----
        prod = tl.sum(q[:, None, :] * k_block[None, :, :], axis=2)

        # ----- soft‑max numerator (exp) ------------------------------
        score_f32 = tl.cast(prod, tl.float32) * scale
        exp_score = tl.exp(score_f32 - max_score[:, None])      # (HEADS_PER_BLOCK, BLOCK_K)

        # ----- update normalisation denominator -----------------------
        sum_exp += tl.sum(exp_score, axis=1)                    # (HEADS_PER_BLOCK,)

        # ----- load V block (latent values) --------------------------
        # V shape: (B, L, Dv_lat)
        # We will accumulate the weighted sum in chunks of BLOCK_DV.
        for start_d in range(0, Dv_lat, BLOCK_DV):
            cur_d = start_d + tl.arange(0, BLOCK_DV)            # (BLOCK_DV,)
            d_mask = cur_d < Dv_lat

            # slice of V we need: (BLOCK_K, BLOCK_DV)
            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(V_ptr + offs_v,
                              mask=k_mask[:, None] & d_mask[None, :],
                              other=0.0)                         # (BLOCK_K, BLOCK_DV)

            # weight with the scalar exp_score per head
            # exp_score : (HEADS_PER_BLOCK, BLOCK_K) → broadcast over BLOCK_DV
            weighted = v_slice[None, :, :] * exp_score[:, :, None]   # (HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV)

            # reduction over the K‑dimension → (HEADS_PER_BLOCK, BLOCK_DV)
            acc += tl.sum(weighted, axis=1)

    # ------------------------------------------------------------------
    # 4️⃣  Normalise the latent accumulator  (latent = acc / sum_exp)
    # ------------------------------------------------------------------
    # sum_exp is float32, acc is bfloat16 → cast
    latent = acc / tl.cast(sum_exp[:, None], tl.bfloat16)        # (HEADS_PER_BLOCK, BLOCK_DV)

    # ------------------------------------------------------------------
    # 5️⃣  Multiply with per‑head value‑projection (wV_T) → get head output
    # ------------------------------------------------------------------
    # wV_T layout: (H, Dv_lat, Dv)  – we will also process it in BLOCK_DV chunks
    v_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.bfloat16)  # (HEADS_PER_BLOCK, Dv)

    for start_d in range(0, Dv_lat, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV)                 # (BLOCK_DV,)
        d_mask = cur_d < Dv_lat

        # load a (HEADS_PER_BLOCK, BLOCK_DV, Dv) sub‑matrix of wV_T
        # wV_T stride: (head, latent, out)
        offs_wV = (
            (head_start + head_range)[:, None, None] * stride_wV_T_head
            + cur_d[None, :, None] * stride_wV_T_lat
            + tl.arange(0, Dv)[None, None, :] * stride_wV_T_out
        )
        wV_block = tl.load(wV_T_ptr + offs_wV,
                           mask=head_valid[:, None] & d_mask[None, :],
                           other=0.0)                          # (HEADS_PER_BLOCK, BLOCK_DV, Dv)

        # current slice of latent (already normalised)
        lat_slice = latent[:, start_d:start_d + BLOCK_DV]          # (HEADS_PER_BLOCK, BLOCK_DV)

        # multiply‑accumulate:  (HEADS_PER_BLOCK, Dv)
        #  lat_slice[..., None] * wV_block → (HEADS_PER_BLOCK, BLOCK_DV, Dv)
        v_head += tl.sum(wV_block * lat_slice[:, :, None], axis=1)

    # ------------------------------------------------------------------
    # 6️⃣  Accumulate head contributions into the final output vector
    # ------------------------------------------------------------------
    # wO layout: (Dim, H*Dv).  Slice belonging to this tile:
    #   columns [head_start*Dv , (head_start+HEADS_PER_BLOCK)*Dv)
    out_base = b * stride_out_batch
    for dim_start in range(0, Dim, 32):            # 32‑wide block for wO
        cur_dim = dim_start + tl.arange(0, 32)
        dim_mask = cur_dim < Dim

        # wO slice → (32, HEADS_PER_BLOCK*Dv)
        # We load the whole tile of heads at once and then do a small matmul.
        # Offsets for wO:
        #   row offset = cur_dim * stride_wO_out
        #   column offset = head_start*Dv * stride_wO_in
        offs_wO = (
            cur_dim[:, None] * stride_wO_out
            + (head_start * Dv + tl.arange(0, HEADS_PER_BLOCK * Dv)[None, :]) * stride_wO_in
        )
        # mask for the tail of the wO column‑wise slice
        col_mask = tl.arange(0, HEADS_PER_BLOCK * Dv) < (H * Dv)
        wO_block = tl.load(wO_ptr + offs_wO,
                           mask=dim_mask[:, None] & col_mask[None, :],
                           other=0.0)                     # (32, HEADS_PER_BLOCK*Dv)

        # reshaping the block so we can treat it as (32, HEADS_PER_BLOCK, Dv)
        wO_block = wO_block.view(32, HEADS_PER_BLOCK, Dv)          # (32, Hpb, Dv)

        # head output v_head : (Hpb, Dv) → broadcast over dim
        # contribution = Σ_{head} wO[:,head,:] · v_head[head,:]
        #  → (32,) for each head, then summed across heads
        out_update = tl.sum(wO_block * v_head[None, :, :], axis=1)   # (32,)

        # store (accumulate) into the final output
        out_ptr = Out_ptr + out_base + cur_dim * stride_out_dim
        prev = tl.load(out_ptr, mask=dim_mask, other=0.0)
        tl.store(out_ptr, prev + out_update, mask=dim_mask)


# ----------------------------------------------------------------------
# Fast‑path (d_nope == 0) – now using the multi‑head Triton kernel above
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
    Same logical flow as the reference‑implementation but the heavy
    attention + output‑projection part is executed by the Triton kernel
    `_triton_attn_out_kernel_multihead`.  This kernel processes several heads
    per program, drastically reducing the number of redundant loads of K/V.
    """
    # --------------------------------------------------------------
    # 0️⃣  Shape / config shortcuts
    # --------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim
    dim = config.dim

    # --------------------------------------------------------------
    # 1️⃣  Down‑projection
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                               # (B, Dim)
    q_lora = F.linear(x2, wDQ)                      # (B, dq)
    kv_lora0 = F.linear(x2, wDKV)                   # (B, dkv + drope)

    # --------------------------------------------------------------
    # 2️⃣  Write new token into KV‑cache (including RoPE for the key)
    # --------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora0[:, :dkv]               # (B, dkv)
    rope_raw_new   = kv_lora0[:, dkv:]              # (B, drope)

    # RoPE for key (position = cur_len)
    cos_k = cos_tbl[cur_len]                         # (drope,)
    sin_k = sin_tbl[cur_len]                         # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write latent + rotated key into the cache (contiguous layout)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 3️⃣  Up‑project Q and apply RoPE to the query
    # --------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                    # (B, nh*drope)
    q_up = q_up.view(bs, nh, drope)                # (B, nh, drope)

    # RoPE for query (position = new_len‑1)
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                           # (drope,)
    sin_q = sin_tbl[q_pos]                           # (drope,)
    q_rot = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # --------------------------------------------------------------
    # 4️⃣  Gather full KV from the cache
    # --------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]           # (B, L, dkv+drope)
    k_rope = kv_all[..., dkv:]                      # (B, L, drope)
    v_latent = kv_all[..., :dkv]                    # (B, L, dkv)

    # --------------------------------------------------------------
    # 5️⃣  Prepare weight layouts
    # --------------------------------------------------------------
    # wV_T : (nh, dkv, dv)
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()

    # --------------------------------------------------------------
    # 6️⃣  Launch the fused Triton kernel
    # --------------------------------------------------------------
    # Strides for the tensors
    stride_q_batch = q_rot.stride(0)
    stride_q_head  = q_rot.stride(1)
    stride_q_dim   = q_rot.stride(2)

    stride_k_batch = k_rope.stride(0)
    stride_k_len   = k_rope.stride(1)
    stride_k_dim   = k_rope.stride(2)

    stride_v_batch = v_latent.stride(0)
    stride_v_len   = v_latent.stride(1)
    stride_v_dim   = v_latent.stride(2)

    stride_wV_T_head = wV_T.stride(0)
    stride_wV_T_lat  = wV_T.stride(1)
    stride_wV_T_out  = wV_T.stride(2)

    stride_wO_out = wO.stride(0)            # (Dim,)
    stride_wO_in  = wO.stride(1)            # (nh*dv,)

    out = torch.empty((bs, dim), dtype=torch.bfloat16, device=x.device)

    # Launch configuration
    HEADS_PER_BLOCK = 8                     # experimentally good trade‑off on H200
    BLOCK_K = 64
    BLOCK_DV = 32

    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    scale = 1.0 / math.sqrt(drope)

    _triton_attn_out_kernel_multihead[grid](
        # pointers
        q_rot, k_rope, v_latent,
        wV_T, wO, out,

        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len, stride_k_dim,
        stride_v_batch, stride_v_len, stride_v_dim,
        stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
        stride_wO_out, stride_wO_in,
        stride_out_batch, out.stride(1),

        # runtime constants
        bs, nh, new_len, drope, dkv, dv, dim,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV,

        # Triton launch‑config
        num_warps=8, num_stages=4,
    )

    # --------------------------------------------------------------
    # 7️⃣  Final reshape to (B, 1, Dim) as required by the original API
    # --------------------------------------------------------------
    out = out.unsqueeze(1)          # (B, 1, Dim)

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
        # exact reference implementation – unchanged
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
    wDKV = config.KV_proj_down_weight         # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # Build / fetch RoPE tables
    # --------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope, config.max_seq_len,
                                                    x.device)

    # --------------------------------------------------------------
    # Fast‑path – the common configuration has d_nope == 0
    # --------------------------------------------------------------
    if d_nope == 0:
        return _fast_forward_multihead(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin
        )

    # --------------------------------------------------------------
    # General case – fall back to the compiled reference implementation
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