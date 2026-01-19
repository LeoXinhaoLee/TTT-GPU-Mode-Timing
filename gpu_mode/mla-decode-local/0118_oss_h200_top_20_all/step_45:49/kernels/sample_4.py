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
# Global caches (kept across calls) – never re‑allocated
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None          # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None          # (max_seq_len, rope_dim)  bfloat16
_cached_wq_fused: torch.Tensor = None     # (nh*rope_dim, dim)     bfloat16
_cached_wV_T_bf16: torch.Tensor = None    # (nh, dkv, dv)          bfloat16

# ----------------------------------------------------------------------
# Utility – same rotate_half used by RoPE
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# Pre‑compute cosine / sine tables for RoPE (bfloat16)
# ----------------------------------------------------------------------
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
        torch.bfloat16
    )
    pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)
    idx = pos * theta                     # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)   # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel – fused attention + per‑head value projection
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config(
            {"HEADS_PER_BLOCK": 128, "BLOCK_K": 4096, "BLOCK_DV": 64},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"HEADS_PER_BLOCK": 128, "BLOCK_K": 2048, "BLOCK_DV": 64},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"HEADS_PER_BLOCK": 128, "BLOCK_K": 1024, "BLOCK_DV": 64},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=[
        "B",
        "H",
        "L",
        "Dq",
        "Dv_lat",
        "Dv_out",
    ],
)
@triton.jit
def triton_attn_fused(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bf16
    K_ptr,                # (B, L, Dq)                 bf16
    V_ptr,                # (B, L, Dv_lat)             bf16
    wV_ptr,               # (H, Dv_lat, Dv_out)        bf16
    Out_ptr,              # (B, H, Dv_out)             bf16
    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,          # Q
    stride_k_batch, stride_k_len,  stride_k_dim,          # K
    stride_v_batch, stride_v_len,  stride_v_dim,          # V
    stride_wv_head, stride_wv_lat, stride_wv_out,        # wV
    stride_out_batch, stride_out_head, stride_out_dim,    # Out
    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    L: tl.constexpr,          # KV length
    Dq: tl.constexpr,         # rope_dim (e.g. 64)
    Dv_lat: tl.constexpr,     # latent value dim (kv_lora_rank, e.g. 512)
    Dv_out: tl.constexpr,     # head value dim (v_head_dim, e.g. 128)
    scale: tl.constexpr,      # 1/sqrt(Dq)
    # ------------------------------------------------------------------
    # Tuning parameters
    # ------------------------------------------------------------------
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Fully‑fused multi‑head attention for the *latent* path.
    Performs stable soft‑max, accumulates the weighted sum of the latent values
    and projects them to the head‑output space – all in registers.
    """
    pid = tl.program_id(0)                     # one program per batch

    # ------------------------------------------------------------------
    # 1️⃣ Identify which batch and which head‑tile we are handling
    # ------------------------------------------------------------------
    b = pid // ((H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK)
    tile = pid % ((H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK)
    head_start = tile * HEADS_PER_BLOCK
    head_end = tl.minimum(head_start + HEADS_PER_BLOCK, H)
    heads_in_tile = head_end - head_start    # ≤ HEADS_PER_BLOCK

    # ------------------------------------------------------------------
    # 2️⃣ Load queries for this tile (heads_in_tile × Dq)
    # ------------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_valid = (head_start + head_range) < H

    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(
        Q_ptr + offs_q,
        mask=head_valid[:, None],
        other=0.0,
    )  # (HEADS_PER_BLOCK, Dq)

    # ------------------------------------------------------------------
    # 3️⃣ Soft‑max state (max & sum) and latent accumulator (FP32)
    # ------------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dv_lat], dtype=tl.float32)

    # ------------------------------------------------------------------
    # 4️⃣ Main loop over KV blocks
    # ------------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)
        k_mask = cur_k < L

        # --------------------------------------------------------------
        #   Load K block (read‑only cache)
        # --------------------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(
            K_ptr + offs_k,
            mask=k_mask[:, None],
            other=0.0,
            cache_modifier="CG",
        )   # (BLOCK_K, Dq)

        # --------------------------------------------------------------
        #   Q·K → scores (float32)
        # --------------------------------------------------------------
        prod = tl.dot(q, tl.permute(k_block, (1, 0)))          # (HEADS_PER_BLOCK, BLOCK_K) bf16
        score_f32 = tl.cast(prod, tl.float32) * scale

        # --------------------------------------------------------------
        #   Numerically‑stable soft‑max update
        # --------------------------------------------------------------
        block_max = tl.max(score_f32, axis=1)                  # (HEADS_PER_BLOCK,)
        new_max   = tl.maximum(max_score, block_max)

        exp_factor = tl.exp(max_score - new_max)               # (HEADS_PER_BLOCK,)
        sum_exp   = sum_exp * exp_factor
        latent_acc = latent_acc * exp_factor[:, None]

        exp_score = tl.exp(score_f32 - new_max[:, None])      # (HEADS_PER_BLOCK, BLOCK_K)
        sum_exp   = sum_exp + tl.sum(exp_score, axis=1)       # (HEADS_PER_BLOCK,)

        # --------------------------------------------------------------
        #   Load V slice (read‑only cache)
        # --------------------------------------------------------------
        col_offs = tl.arange(0, Dv_lat)
        offs_v = (
            b * stride_v_batch
            + cur_k[:, None] * stride_v_len
            + col_offs[None, :] * stride_v_dim
        )
        v_slice = tl.load(
            V_ptr + offs_v,
            mask=k_mask[:, None],
            other=0.0,
            cache_modifier="CG",
        )   # (BLOCK_K, Dv_lat) bf16
        v_fp32 = tl.cast(v_slice, tl.float32)               # (BLOCK_K, Dv_lat)

        # weighted sum Σ exp_score[h,k] * v[k,d] → (HEADS_PER_BLOCK, Dv_lat)
        latent_tile = tl.dot(exp_score, v_fp32)              # (HEADS_PER_BLOCK, Dv_lat)
        latent_acc = latent_acc + latent_tile

        # keep the new maximum for next iteration
        max_score = new_max

    # ------------------------------------------------------------------
    # 5️⃣ Normalise accumulated latent vectors (soft‑max denominator)
    # ------------------------------------------------------------------
    norm_factor = tl.reciprocal(sum_exp)[:, None]                 # (HEADS_PER_BLOCK, 1)
    latent_norm = latent_acc * norm_factor                        # (HEADS_PER_BLOCK, Dv_lat) fp32

    # ------------------------------------------------------------------
    # 6️⃣ Project latent → head‑output (dv_out) — broadcast wV per tile
    # ------------------------------------------------------------------
    for start_d in range(0, Dv_out, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV, tl.int32)
        d_mask = cur_d < Dv_out

        # --------------------------------------------------------------
        #   Load the slice of wV for this tile (read‑only cache)
        # --------------------------------------------------------------
        head_off = (head_start + head_range)[:, None, None] * stride_wv_head
        lat_off  = tl.arange(0, Dv_lat)[None, :, None] * stride_wv_lat
        d_off    = cur_d[None, None, :] * stride_wv_out
        offs_wv = head_off + lat_off + d_off

        wv_slice = tl.load(
            wV_ptr + offs_wv,
            mask=head_valid[:, None, None] & d_mask[None, None, :],
            other=0.0,
            cache_modifier="CG",
        )   # (HEADS_PER_BLOCK, Dv_lat, BLOCK_DV) bf16
        wv_fp32 = tl.cast(wv_slice, tl.float32)

        # --------------------------------------------------------------
        #   latent_norm (HEADS_PER_BLOCK, Dv_lat) × wv_fp32 → (HEADS_PER_BLOCK, BLOCK_DV)
        # --------------------------------------------------------------
        out_tile = tl.dot(latent_norm, wv_fp32)                # (HEADS_PER_BLOCK, BLOCK_DV)

        # --------------------------------------------------------------
        #   Store the projected values
        # --------------------------------------------------------------
        offs_out = (
            b * stride_out_batch
            + (head_start + head_range)[:, None] * stride_out_head
            + cur_d[None, :] * stride_out_dim
        )
        tl.store(
            Out_ptr + offs_out,
            tl.cast(out_tile, tl.bfloat16),
            mask=head_valid[:, None] & d_mask[None, :],
        )
    # ------------------------------------------------------------------
    # END OF KERNEL
    # ------------------------------------------------------------------



# ----------------------------------------------------------------------
# Fast‑path – d_nope == 0 (the overwhelmingly common configuration)
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
    Optimised forward for the case `qk_nope_head_dim == 0`.
    The only heavy operation (attention) is performed by a fused Triton kernel.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim
    dim = config.dim

    # --------------------------------------------------------------
    # 1️⃣ Down‑projection + KV‑cache update (single‑token case)
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                     # (B, dim)
    kv_lora = F.linear(x2, wDKV)          # (B, dkv+drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    # split latent & rope part
    kv_latent_new = kv_lora[:, :dkv]               # (B, dkv)
    rope_raw_new   = kv_lora[:, dkv:]              # (B, drope)

    # RoPE rotation for the *new* key (in‑place)
    cos_k = cos_tbl[cur_len]                       # (drope,)
    sin_k = sin_tbl[cur_len]                       # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write into the cache (the cache lives in bf16)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 2️⃣ Q up‑projection + RoPE (single fused GEMM)
    # --------------------------------------------------------------
    global _cached_wq_fused
    if _cached_wq_fused is None or _cached_wq_fused.shape != (nh * drope, dim):
        # wUQ: (nh*drope, dq)   wDQ: (dq, dim)
        _cached_wq_fused = torch.matmul(wUQ, wDQ)          # (nh*drope, dim)
    q = F.linear(x2, _cached_wq_fused)                     # (B, nh*drope)
    q = q.view(bs, nh, drope)                             # (B, nh, drope)

    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                                 # (drope,)
    sin_q = sin_tbl[q_pos]                                 # (drope,)
    q = q * cos_q + _rotate_half(q) * sin_q                # (B, nh, drope)

    # --------------------------------------------------------------
    # 3️⃣ Assemble K (rope‑rotated) and V (latent) from KV‑cache
    # --------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]                 # (B, L, dkv+drope)
    k_rope = kv_all[..., dkv:]                             # (B, L, drope) – already rope‑rotated
    v_latent = kv_all[..., :dkv]                           # (B, L, dkv)

    # --------------------------------------------------------------
    # 4️⃣ Triton fused attention + per‑head value projection
    # --------------------------------------------------------------
    # ensure contiguous layout (required by the kernel)
    q_kernel = q.contiguous()                              # (B, NH, drope)
    k_kernel = k_rope.contiguous()                         # (B, L, drope)
    v_kernel = v_latent.contiguous()                       # (B, L, dkv)

    # allocate output buffer for head‑output values
    v_head_buf = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # prepare transposed value‑projection matrix wV_T (H, dkv, dv)
    global _cached_wV_T_bf16
    if _cached_wV_T_bf16 is None or _cached_wV_T_bf16.shape != (nh, dkv, dv):
        # wUKV has shape ((d_nope+dv)*nh, dkv) . With d_nope==0 we have (nh*dv, dkv)
        _cached_wV_T_bf16 = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()

    scale = 1.0 / math.sqrt(drope)   # Dq == drope

    # One program per batch (the kernel tiles all heads internally)
    grid = (bs,)

    triton_attn_fused[grid](
        # pointers
        q_kernel,
        k_kernel,
        v_kernel,
        _cached_wV_T_bf16,
        v_head_buf,
        # strides
        q_kernel.stride(0),
        q_kernel.stride(1),
        q_kernel.stride(2),
        k_kernel.stride(0),
        k_kernel.stride(1),
        k_kernel.stride(2),
        v_kernel.stride(0),
        v_kernel.stride(1),
        v_kernel.stride(2),
        _cached_wV_T_bf16.stride(0),
        _cached_wV_T_bf16.stride(1),
        _cached_wV_T_bf16.stride(2),
        v_head_buf.stride(0),
        v_head_buf.stride(1),
        v_head_buf.stride(2),
        # compile‑time arguments
        bs,
        nh,
        new_len,
        drope,
        dkv,
        dv,
        scale,
        HEADS_PER_BLOCK=128,
        BLOCK_K=4096,
        BLOCK_DV=64,
    )

    # --------------------------------------------------------------
    # 5️⃣ Final output projection – BF16 GEMM
    # --------------------------------------------------------------
    y_head_flat = v_head_buf.view(bs, nh * dv)          # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                     # (B, dim) bf16
    out = out.unsqueeze(1)                             # (B, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Compiled fallback (d_nope > 0) – unchanged from reference
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
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

    # --------------------------------------------------------------
    # Pre‑compute RoPE tables (cached globally)
    # --------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # --------------------------------------------------------------
    # Fast‑path when there is no‑PE dimension
    # --------------------------------------------------------------
    if d_nope == 0:
        out, new_kv = _fast_forward_multihead(
            config,
            x,
            kv_cache,
            wDQ,
            wDKV,
            wUQ,
            wUKV,
            wO,
            _cached_cos,
            _cached_sin,
        )
        # kv_cache has already been updated inside the fast‑path
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