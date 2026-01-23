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
# Helper: rotate‑half (identical to the reference implementation)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half‑rotate for RoPE – identical to the reference implementation."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# Global RoPE tables (cos / sin) – cached on‑the‑fly
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None        # (max_seq_len, rope_dim)   bfloat16
_cached_sin: torch.Tensor = None        # (max_seq_len, rope_dim)   bfloat16


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Build the (cos, sin) tables for rotary embeddings.
    The tables are (max_seq_len, dim) in bfloat16.
    """
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half,
                                        dtype=torch.float32,
                                        device=device) / half)).to(torch.bfloat16)  # (half,)
    pos = torch.arange(max_seq_len,
                       dtype=torch.int64,
                       device=device).unsqueeze_(1)                 # (max_seq_len, 1)
    idx = pos * theta                                            # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                         # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel – attention for the “no‑NoPE” case (d_nope == 0)
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_nope_opt(
    # --------------------------------------------------------------
    # Pointers
    # --------------------------------------------------------------
    Q_ptr,                # (B, H, D)               bfloat16 – already rotated
    K_ptr,                # (B, L, D)               bfloat16 – already rotated
    V_ptr,                # (B, L, Dv_lat)          bfloat16
    wV_T_ptr,             # (H, Dv_lat, Dv)         bfloat16
    Y_ptr,                # (B, H, Dv)              bfloat16   (intermediate Y)
    # --------------------------------------------------------------
    # Strides
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,   # Q    (B, H, D)
    stride_k_batch, stride_k_len , stride_k_dim,   # K    (B, L, D)
    stride_v_batch, stride_v_len , stride_v_dim,   # V    (B, L, Dv_lat)
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,  # wV_T (H, Dv_lat, Dv)
    stride_y_batch, stride_y_head, stride_y_dv,           # Y    (B, H, Dv)
    # --------------------------------------------------------------
    # Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    L: tl.constexpr,          # KV length after insertion
    D: tl.constexpr,          # rope dimension (e.g. 64)
    Dv_lat: tl.constexpr,     # kv‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value‑dim (e.g. 128)
    scale: tl.constexpr,      # 1/sqrt(D)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton implementation for the no‑NoPE MLA case.
    Performs a numerically‑stable softmax, weighted‑sum of the latent
    values, and per‑head value projection.
    """
    pid = tl.program_id(0)

    # --------------------------------------------------------------
    # 1️⃣  Tile indexing (batch * head‑tile)
    # --------------------------------------------------------------
    heads_per_grid = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // heads_per_grid                     # batch index
    tile = pid % heads_per_grid                   # which head‑tile inside the batch
    head_start = tile * HEADS_PER_BLOCK           # first head handled by this program

    # --------------------------------------------------------------
    # 2️⃣  Load Q for the heads handled by this program.
    # --------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_mask = (head_start + head_range) < H

    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, D)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_mask[:, None],
                 other=0.0)                         # (HPB, D)

    # --------------------------------------------------------------
    # 3️⃣  Numerically‑stable softmax bookkeeping
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HPB,)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (HPB,)
    # latent accumulator: Σ  (attn • V)   –  shape (HPB, Dv_lat)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dv_lat], dtype=tl.bfloat16)

    # --------------------------------------------------------------
    # 4️⃣  Main loop over K‑blocks
    # --------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)           # (BLOCK_K,)
        k_mask = cur_k < L

        # ----------------------------------------------------------
        # Load K‑block (already rotated)
        # ----------------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, D)[None, :] * stride_k_dim
        )
        k = tl.load(K_ptr + offs_k,
                     mask=k_mask[:, None],
                     other=0.0)                     # (BLOCK_K, D)

        # ----------------------------------------------------------
        # Compute raw scores = q ⋅ kᵀ
        # ----------------------------------------------------------
        # (HPB, D) @ (D, BLOCK_K) -> (HPB, BLOCK_K)
        score = tl.dot(q, tl.transpose(k))               # (HPB, BLOCK_K)
        score_f32 = tl.cast(score, tl.float32) * scale   # (HPB, BLOCK_K)

        # ----------------------------------------------------------
        # Stable‑softmax bookkeeping
        # ----------------------------------------------------------
        block_max = tl.max(score_f32, axis=1)            # (HPB,)
        new_max = tl.maximum(max_score, block_max)       # (HPB,)

        # scale previous sum/latent by exp(old_max‑new_max)
        scale_factor = tl.exp(max_score - new_max)        # (HPB,)
        sum_exp = sum_exp * scale_factor
        latent_acc = latent_acc * tl.cast(scale_factor, tl.bfloat16)[:, None]

        # compute exp of the new (shifted) scores
        exp_score_f32 = tl.exp(score_f32 - new_max[:, None])   # (HPB, BLOCK_K)
        exp_score = tl.cast(exp_score_f32, tl.bfloat16)         # (HPB, BLOCK_K)

        sum_exp = sum_exp + tl.sum(exp_score, axis=1)            # (HPB,)

        # ----------------------------------------------------------
        # 5️⃣  Accumulate the weighted value vectors
        # ----------------------------------------------------------
        offs_v = (
            b * stride_v_batch
            + cur_k[:, None] * stride_v_len
            + tl.arange(0, Dv_lat)[None, :] * stride_v_dim
        )
        v_slice = tl.load(V_ptr + offs_v,
                          mask=k_mask[:, None],
                          other=0.0)                # (BLOCK_K, Dv_lat)

        # latent_acc  (HPB, Dv_lat)  +=  exp_score (HPB, BLOCK_K)  @  v_slice (BLOCK_K, Dv_lat)
        latent_acc += tl.dot(exp_score, v_slice)

        max_score = new_max

    # ------------------------------------------------------------------
    # 6️⃣  Normalise the latent accumulator (divide by Σexp)
    # ------------------------------------------------------------------
    latent = latent_acc / tl.cast(sum_exp[:, None], tl.bfloat16)   # (HPB, Dv_lat)

    # ------------------------------------------------------------------
    # 7️⃣  Project latent → per‑head output (size Dv)
    # ------------------------------------------------------------------
    offs_wV = (
        (head_start + head_range)[:, None, None] * stride_wV_T_head
        + tl.arange(0, Dv_lat)[None, :, None] * stride_wV_T_lat
        + tl.arange(0, Dv)[None, None, :] * stride_wV_T_out
    )
    wV_block = tl.load(wV_T_ptr + offs_wV,
                       mask=head_mask[:, None, None],
                       other=0.0)                     # (HPB, Dv_lat, Dv)

    # y_head = latent @ wV_T   (batched matmul)
    y_head = tl.sum(latent[:, :, None] * wV_block, axis=1)      # (HPB, Dv)

    # ------------------------------------------------------------------
    # 8️⃣  Store per‑head outputs (still **before** final linear)
    # ------------------------------------------------------------------
    offs_y = (
        b * stride_y_batch
        + (head_start + head_range)[:, None] * stride_y_head
        + tl.arange(0, Dv)[None, :] * stride_y_dv
    )
    tl.store(Y_ptr + offs_y,
             y_head,
             mask=head_mask[:, None])


# ----------------------------------------------------------------------
# Fast‑path for the “no‑NoPE” configuration (d_nope == 0)
# ----------------------------------------------------------------------
def _fast_forward_nope_opt(
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
    Optimised forward for the benchmarked configuration where
    qk_nope_head_dim == 0 (i.e. no‑NoPE).  The heavy attention
    computation is performed with a fused Triton kernel.
    """
    # ------------------------------------------------------------------
    # 0️⃣  Basic shapes / shortcuts
    # ------------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    drope = config.qk_rope_head_dim   # rope dimension D
    dv = config.v_head_dim

    # ------------------------------------------------------------------
    # 1️⃣  Down‑projection for Q and KV (both have seq_len == 1)
    # ------------------------------------------------------------------
    x_flat = x.squeeze(1)                           # (B, dim)

    q_lora = F.linear(x_flat, wDQ)                   # (B, dq)
    kv_lora = F.linear(x_flat, wDKV)                 # (B, dkv + drope)

    # ------------------------------------------------------------------
    # 2️⃣  Insert the new KV entry into the cache (rotate the key)
    # ------------------------------------------------------------------
    cur_len = kv_cache.seq_len                     # length before insertion

    kv_latent_new = kv_lora[:, :dkv]                # (B, dkv)
    k_rope_raw    = kv_lora[:, dkv:]                # (B, drope)

    # apply RoPE to the new key (position = cur_len)
    cos_k = cos_tbl[cur_len]                        # (drope,)
    sin_k = sin_tbl[cur_len]                        # (drope,)
    cos_k = cos_k.view(1, drope)
    sin_k = sin_k.view(1, drope)
    k_rope_rot = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k   # (B, drope)

    # write into cache
    kv_cache.data[:, cur_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len, dkv:] = k_rope_rot
    kv_cache.seq_len = cur_len + 1
    kv_len = kv_cache.seq_len                       # new total length

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries and apply RoPE (position = kv_len‑1)
    # ------------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                    # (B, drope * nh)
    q_up = q_up.view(bs, nh, drope)                # (B, nh, drope)

    query_pos = kv_len - 1
    cos_q = cos_tbl[query_pos].view(1, 1, drope)   # (1,1,drope)
    sin_q = sin_tbl[query_pos].view(1, 1, drope)   # (1,1,drope)

    q_rope = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # ------------------------------------------------------------------
    # 4️⃣  Prepare K (rotated) and V (latent) tensors from the cache
    # ------------------------------------------------------------------
    K_rot = kv_cache.data[..., dkv:]               # (B, max_seq_len, drope)
    V_latent = kv_cache.data[..., :dkv]            # (B, max_seq_len, dkv)

    # ------------------------------------------------------------------
    # 5️⃣  Allocate output buffer for per‑head values (before final wo)
    # ------------------------------------------------------------------
    Y_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # ------------------------------------------------------------------
    # 6️⃣  Reshape wV_T into the per‑head projection tensor expected by the kernel
    # ------------------------------------------------------------------
    # wUKV shape: (nh * dv, dkv) --> (nh, dv, dkv) --> (nh, dkv, dv)
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # 7️⃣  Triton launch configuration (tuned)
    # ------------------------------------------------------------------
    HEADS_PER_BLOCK = 16          # fits a warp nicely, reduces grid size
    BLOCK_K = 128                 # larger KV tile → fewer loop iterations
    # grid = (#batch * #head‑tiles)
    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    _triton_attn_nope_opt[grid](
        # pointers
        Q_ptr = q_rope,
        K_ptr = K_rot,
        V_ptr = V_latent,
        wV_T_ptr = wV_T,
        Y_ptr = Y_head,
        # strides
        stride_q_batch = q_rope.stride(0),
        stride_q_head  = q_rope.stride(1),
        stride_q_dim   = q_rope.stride(2),

        stride_k_batch = K_rot.stride(0),
        stride_k_len   = K_rot.stride(1),
        stride_k_dim   = K_rot.stride(2),

        stride_v_batch = V_latent.stride(0),
        stride_v_len   = V_latent.stride(1),
        stride_v_dim   = V_latent.stride(2),

        stride_wV_T_head = wV_T.stride(0),
        stride_wV_T_lat  = wV_T.stride(1),
        stride_wV_T_out  = wV_T.stride(2),

        stride_y_batch = Y_head.stride(0),
        stride_y_head  = Y_head.stride(1),
        stride_y_dv    = Y_head.stride(2),
        # compile‑time constants
        B = bs,
        H = nh,
        L = kv_len,
        D = drope,
        Dv_lat = dkv,
        Dv = dv,
        scale = 1.0 / math.sqrt(drope),
        HEADS_PER_BLOCK = HEADS_PER_BLOCK,
        BLOCK_K = BLOCK_K,
        # launch config
        num_warps = 8,
        num_stages = 3,
    )

    # ------------------------------------------------------------------
    # 8️⃣  Final linear projection (wo)
    # ------------------------------------------------------------------
    y_head_flat = Y_head.view(bs, nh * dv)            # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                   # (B, dim)
    out = out.unsqueeze(1)                            # (B, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# General (fallback) implementation – compiled with torch.compile
# ----------------------------------------------------------------------
_compiled_forward = None


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
        # -------------------- reference logic (unchanged) --------------------
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
# Entry point expected by the benchmark harness
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes a single forward step of the Multi‑Head Latent Attention (MLA) module.
    For the common “no‑NoPE” configuration (qk_nope_head_dim == 0) a highly‑optimised
    Triton kernel is used.  All other configurations fall back to a compiled reference.
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Extract scalar configuration values
    # --------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # --------------------------------------------------------------
    # Weight tensors (already on the correct device and dtype)
    # --------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight        # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight          # ((d_nope + dv) * nh, dkv)
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
    # Fast‑path: d_nope == 0  (benchmark configuration)
    # --------------------------------------------------------------
    if d_nope == 0:
        return _fast_forward_nope_opt(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin,
        )

    # --------------------------------------------------------------
    # General case – use compiled fallback (unchanged)
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

    # Output already has shape (B, 1, Dim)
    return out, kv_cache.data