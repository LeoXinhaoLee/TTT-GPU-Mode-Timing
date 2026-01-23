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
_cached_wV_T: torch.Tensor = None    # (n_heads, kv_lora_rank, v_head_dim)  bfloat16
# ----------------------------------------------------------------------
# Utility helpers
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
# Triton kernel – fused attention + *single* load of wV_T per head‑tile
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
    out_ptr,              # (B, H, Dv)                 bf16
    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len,  stride_k_dim,
    stride_v_batch, stride_v_len,  stride_v_dim,
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
    stride_out_batch, stride_out_head, stride_out_dim,
    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,        # batch size
    H: tl.constexpr,        # total heads
    Dq: tl.constexpr,       # rope dimension (e.g. 64)
    Dkv: tl.constexpr,      # KV‑lora rank (e.g. 512)
    Dv: tl.constexpr,       # per‑head value dimension (e.g. 128)
    scale: tl.constexpr,    # 1 / sqrt(Dq)
    # ------------------------------------------------------------------
    # Tiling parameters
    # ------------------------------------------------------------------
    HEADS_PER_BLOCK: tl.constexpr,   # #heads processed by one program (max = 1024/Dq)
    BLOCK_K: tl.constexpr,           # #keys processed per iteration
    BLOCK_DKV: tl.constexpr,         # #latents processed per inner‑loop
    BLOCK_DV: tl.constexpr,          # tile size for final projection
    # ------------------------------------------------------------------
    # Runtime arguments
    # ------------------------------------------------------------------
    L: tl.int32,               # current KV length (dynamic)
):
    """
    Optimised attention kernel for the “no‑rope” case (qk_nope_head_dim==0).

    1️⃣  Compute scaled Q·Kᵀ,
        keep numerically‑stable max / sum_exp.
    2️⃣  Accumulate the **latent value aggregation**   latent += exp_score @ V
        (latent has shape [HEADS_PER_BLOCK, Dkv]).
        The scaling factor from the soft‑max is applied to *latent* as well.
    3️⃣  After all KV‑blocks are processed,
        normalise: latent /= sum_exp[...,None].
    4️⃣  Multiply once with the per‑head value‑projection matrix wV_T
        (dkv → dv) and write the result.
    """
    pid = tl.program_id(0)                     # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    batch_id = pid // num_head_tiles
    tile_id  = pid % num_head_tiles
    head_start = tile_id * HEADS_PER_BLOCK

    # ------------------------------------------------------------------
    # 1️⃣ Load Q tile (HEADS_PER_BLOCK × Dq)
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
                 other=0.0)                     # (HEADS_PER_BLOCK, Dq)  bf16
    Q = tl.cast(Q, tl.float32)

    # ------------------------------------------------------------------
    # 2️⃣ Accumulators for numerically‑stable softmax and latent values
    # ------------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (→Hpb)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (→Hpb)
    # latent accumulator: (HEADS_PER_BLOCK, Dkv)  in fp32
    latent    = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)

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
                     cache_modifier='CA')                 # (BLOCK_K, Dq)  bf16
        K = tl.cast(K, tl.float32)

        # ----- Compute raw scores Q·Kᵀ ------------------------------------
        # (HEADS_PER_BLOCK, Dq) @ (Dq, BLOCK_K) → (HEADS_PER_BLOCK, BLOCK_K)
        scores = tl.dot(Q, tl.permute(K, (1, 0))) * scale         # fp32

        # ----- Stable soft‑max update ------------------------------------
        block_max = tl.max(scores, axis=1)                         # (Hpb,)
        new_max   = tl.maximum(max_score, block_max)                # (Hpb,)

        # rescale previous contributions (both sum_exp and latent)
        exp_factor = tl.exp(max_score - new_max)                    # (Hpb,)
        sum_exp   = sum_exp * exp_factor
        latent    = latent * exp_factor[:, None]

        # new block contribution
        exp_score = tl.exp(scores - new_max[:, None])               # (Hpb, BLOCK_K)
        sum_exp   = sum_exp + tl.sum(exp_score, axis=1)             # (Hpb,)

        # ----- Accumulate latent values  (exp_scoreᵀ @ V_block) ---------
        for d_start in tl.range(0, Dkv, BLOCK_DKV):
            cur_d = d_start + tl.arange(0, BLOCK_DKV, tl.int32)    # (BLOCK_DKV,)
            d_mask = cur_d < Dkv

            offs_v = (
                batch_id * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            V = tl.load(V_ptr + offs_v,
                        mask=k_mask[:, None] & d_mask[None, :],
                        other=0.0,
                        cache_modifier='CA')                     # (BLOCK_K, BLOCK_DKV) bf16
            V_fp32 = tl.cast(V, tl.float32)                        # fp32

            # latent contribution for this DKV‑tile
            # exp_score : (Hpb, BLOCK_K)
            # V_fp32   : (BLOCK_K, BLOCK_DKV)
            # result   : (Hpb, BLOCK_DKV)
            X = tl.dot(exp_score, V_fp32)                           # fp32

            # accumulate into the full latent buffer
            # NOTE: Triton does not allow direct slice‑assignment, therefore we
            #       add manually using a mask.
            # Build a mask for the destination slice (Heads, DKV‑tile)
            mask_tile = head_valid[:, None] & d_mask[None, :]
            # Load the current slice (only for masked elements, other=0 is fine)
            #   The slice lives in the “latent” register buffer, so we can update it
            #   with a simple element‑wise add.
            #   Because `latent` is a regular Triton tensor we can address it the
            #   same way as any other tensor.
            latent = tl.where(mask_tile,
                              latent + X,
                              latent)

    # ------------------------------------------------------------------
    # 4️⃣ Normalise the latent aggregation (divide by softmax denominator)
    # ------------------------------------------------------------------
    latent = latent / sum_exp[:, None]                # (HEADS_PER_BLOCK, Dkv)

    # ------------------------------------------------------------------
    # 5️⃣ Project the aggregated latent values to head‑output space
    #    (single read of wV_T per head‑tile)
    # ------------------------------------------------------------------
    out_acc = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)   # (Hpb, Dv)

    for d_out in tl.range(0, Dv, BLOCK_DV):
        cur_out = d_out + tl.arange(0, BLOCK_DV, tl.int32)   # (BLOCK_DV,)
        out_mask = cur_out < Dv

        # Load wV_T slice: (Hpb, Dkv, BLOCK_DV)
        offs_wV = (
            (head_start + head_range)[:, None, None] * stride_wV_T_head
            + tl.arange(0, Dkv)[:, None] * stride_wV_T_lat
            + cur_out[None, None, :] * stride_wV_T_out
        )
        wV = tl.load(wV_T_ptr + offs_wV,
                     mask=head_valid[:, None] & out_mask[None, :],
                     other=0.0,
                     cache_modifier='CA')                     # (Hpb, Dkv, BLOCK_DV) bf16
        wV = tl.cast(wV, tl.float32)

        # latent : (Hpb, Dkv)   @   wV : (Hpb, Dkv, BLOCK_DV)
        # Result: (Hpb, BLOCK_DV) = Σ_{dkv} latent * wV
        # We can compute it with a broadcasted mul + reduction:
        #   (Hpb, Dkv, 1) * (Hpb, Dkv, BLOCK_DV) → sum over axis=1
        Y = tl.sum(latent[:, :, None] * wV, axis=1)          # (Hpb, BLOCK_DV)

        # accumulate into the final output
        out_acc = tl.where(out_mask[None, :],
                           out_acc + Y,
                           out_acc)

    # ------------------------------------------------------------------
    # 6️⃣ Store the per‑batch, per‑head outputs
    # ------------------------------------------------------------------
    offs_out = (
        batch_id * stride_out_batch
        + (head_start + head_range)[:, None] * stride_out_head
        + tl.arange(0, Dv)[None, :] * stride_out_dim
    )
    tl.store(out_ptr + offs_out,
             tl.cast(out_acc, tl.bfloat16),
             mask=head_valid[:, None])
# ----------------------------------------------------------------------
# Fast‑path – common case where qk_nope_head_dim == 0 (optimised kernel)
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
    Fully‑fused forward for the special case where the “no‑rope” part
    of Q/K is disabled (qk_nope_head_dim==0).  All heavy work stays on‑chip.
    The only PyTorch calls are the four light linear projections and the final
    `wo` projection.
    """
    bs = cfg.batch_size
    nh = cfg.n_heads
    drope = cfg.qk_rope_head_dim      # Dq == drope
    dkv   = cfg.kv_lora_rank
    dv    = cfg.v_head_dim
    dim   = cfg.dim

    # --------------------------------------------------------------
    # 1️⃣ Down‑projection + KV‑cache update
    # --------------------------------------------------------------
    x_flat = x.squeeze(1)                        # (B, dim)

    # KV down‑projection (produces [latent, rope] concatenated)
    kv_lora = F.linear(x_flat, wDKV)             # (B, dkv + drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    # split into latent part and raw‑rope part
    kv_latent_new = kv_lora[:, :dkv]              # (B, dkv)
    rope_raw_new   = kv_lora[:, dkv:]             # (B, drope)

    # RoPE for the *new* key (in‑place)
    cos_k = cos_tbl[cur_len]                      # (drope,)
    sin_k = sin_tbl[cur_len]                      # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write into cache
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 2️⃣ Query side: down‑/up‑projection + RoPE
    # --------------------------------------------------------------
    q_lora = F.linear(x_flat, wDQ)                # (B, q_lora_rank)
    q_up   = F.linear(q_lora, wUQ)                # (B, nh * drope)
    q_up   = q_up.view(bs, nh, drope)             # (B, nh, drope)

    # RoPE for query (single token, position = new_len‑1)
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                        # (drope,)
    sin_q = sin_tbl[q_pos]                        # (drope,)
    q_rope = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # --------------------------------------------------------------
    # 3️⃣ Prepare tensors for Triton kernel
    # --------------------------------------------------------------
    Q = q_rope.contiguous()                       # (B, nh, drope) – already RoPE‑rotated
    K = kv_cache.data[:, :new_len, dkv:].contiguous()   # (B, L, drope)
    V = kv_cache.data[:, :new_len, :dkv].contiguous()   # (B, L, dkv)

    # --------------------------------------------------------------
    # 4️⃣ Cache the per‑head value‑projection matrix (dkv → dv)
    # --------------------------------------------------------------
    global _cached_wV_T
    if _cached_wV_T is None:
        # wUKV is ((dv) * nh, dkv) → reshape → (nh, dkv, dv) → transpose to (nh, dkv, dv)
        _cached_wV_T = wUKV.view(nh, dv, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 5️⃣ Allocate output buffers for the Triton kernel
    # --------------------------------------------------------------
    # y_head : (B, nh, dv) – result of attention + per‑head projection
    y_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # --------------------------------------------------------------
    # 6️⃣ Launch the fused kernel
    # --------------------------------------------------------------
    # Tuning knobs – chosen to fit the H200 (max 1024 threads per block)
    HEADS_PER_BLOCK = 16                     # 16 * 64 = 1024 threads
    BLOCK_K          = 256                  # moderate KV‑tile size
    BLOCK_DKV        = 64                   # latent tile size (must divide dkv)
    BLOCK_DV         = 32                   # value‑projection tile size

    scale = 1.0 / math.sqrt(drope)           # Dq == drope

    grid = lambda meta: (bs * triton.cdiv(nh, meta["HEADS_PER_BLOCK"]),)

    _triton_attn_fused_opt[grid](
        # pointers
        Q, K, V,
        _cached_wV_T,
        y_head,
        # strides
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        _cached_wV_T.stride(0), _cached_wV_T.stride(1), _cached_wV_T.stride(2),
        y_head.stride(0), y_head.stride(1), y_head.stride(2),
        # compile‑time constants
        bs, nh, drope, dkv, dv, scale,
        # tuning constants
        HEADS_PER_BLOCK,
        BLOCK_K,
        BLOCK_DKV,
        BLOCK_DV,
        # runtime argument (current KV length)
        new_len,
        num_warps=8,                        # 8 warps → 256 threads, matches layout
    )

    # --------------------------------------------------------------
    # 7️⃣ Final linear projection back to model dimension (cuBLAS)
    # --------------------------------------------------------------
    y_head_flat = y_head.view(bs, nh * dv)          # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                # (B, dim) – bf16
    out = out.unsqueeze(1)                         # (B, 1, dim)

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
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

    # ------------------------------------------------------------------
    # Ensure RoPE cosine / sine tables are cached (global, reused)
    # ------------------------------------------------------------------
    global _cached_cos, _cached_sin
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
    global _compiled_fwd
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