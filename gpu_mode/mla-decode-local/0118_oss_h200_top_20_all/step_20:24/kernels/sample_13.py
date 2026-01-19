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
# Global caches for RoPE tables and compiled fallback
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16
_combined_qkv_weight = {}          # (id_up, id_down, id_kv) -> fused weight
_compiled_forward = None           # generic fallback (torch‑compile)

# ----------------------------------------------------------------------
# Helper utilities (same as the reference)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables for rotary embeddings, cached globally."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)  # (half,)
    pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                         # (max_seq_len, 1)
    idx = pos * theta  # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)  # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)

# ----------------------------------------------------------------------
# Triton kernel – fused attention + per‑head value projection (no wO)
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_vhead_kernel(
    # Pointers
    Q_ptr,          # (B, H, Dq)                bf16
    K_ptr,          # (B, L, Dq)                bf16
    V_ptr,          # (B, L, Dv_lat)            bf16  (Dv_lat = kv_lora_rank)
    wV_T_ptr,       # (H, Dv_lat, Dv)           bf16
    Y_ptr,          # (B, H, Dv)                bf16  (output per‑head value vector)

    # Strides
    stride_q_batch, stride_q_head, stride_q_dim,      # Q   (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,      # K   (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,      # V   (B, L, Dv_lat)
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,  # wV_T (H, Dv_lat, Dv)
    stride_y_batch, stride_y_head, stride_y_dim,      # Y   (B, H, Dv)

    # Compile‑time constants
    B: tl.constexpr,      # batch size
    H: tl.constexpr,      # #heads
    L: tl.constexpr,      # KV length
    Dq: tl.constexpr,     # rope head dim
    Dv_lat: tl.constexpr, # kv_lora_rank (latent dim of values)
    Dv: tl.constexpr,     # v_head_dim
    scale: tl.constexpr,  # 1 / sqrt(Dq)
    BLOCK_K: tl.constexpr,   # size of KV block processed per iteration
    BLOCK_DV: tl.constexpr,  # block size for latent accumulation
):
    """
    For each (batch, head) pair:
        1. Compute numerically stable soft‑max over K.
        2. Accumulate the weighted sum of V (latent dim Dv_lat).
        3. Multiply the resulting latent vector with per‑head value‑projection
           matrix wV_T → per‑head output vector of size Dv.
        4. Write that per‑head output to Y.
    """

    pid = tl.program_id(0)                # 0 … B*H-1
    b = pid // H
    h = pid % H

    # ------------------------------------------------------------------
    # 1️⃣ Load query vector q (Dq)
    # ------------------------------------------------------------------
    offs_q = b * stride_q_batch + h * stride_q_head + tl.arange(0, Dq) * stride_q_dim
    q = tl.load(Q_ptr + offs_q)           # (Dq,)

    # ------------------------------------------------------------------
    # 2️⃣ First pass – compute max score for numerically‑stable softmax
    # ------------------------------------------------------------------
    max_score = tl.full([1], -float('inf'), tl.float32)
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)          # (BLOCK_K,)
        mask_k = cur_k < L
        offs_k = b * stride_k_batch + cur_k * stride_k_len + tl.arange(0, Dq) * stride_k_dim
        k_block = tl.load(K_ptr + offs_k, mask=mask_k, other=0.0)   # (BLOCK_K, Dq)
        prod = tl.sum(q[None, :] * k_block, axis=1)                # (BLOCK_K,)
        max_score = tl.maximum(max_score, tl.cast(prod, tl.float32))

    # ------------------------------------------------------------------
    # 3️⃣ Second pass – compute exp(scores), sum_exp and weighted V sum
    # ------------------------------------------------------------------
    sum_exp = tl.full([1], 0.0, tl.float32)
    # accumulator for the weighted latent vector (size Dv_lat)
    acc = tl.zeros([BLOCK_DV], dtype=tl.bfloat16)

    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = cur_k < L

        # ----- keys ---------------------------------------------------
        offs_k = b * stride_k_batch + cur_k * stride_k_len + tl.arange(0, Dq) * stride_k_dim
        k_block = tl.load(K_ptr + offs_k, mask=mask_k, other=0.0)
        prod = tl.sum(q[None, :] * k_block, axis=1)          # (BLOCK_K,)

        # ----- soft‑max numerator (exp) -------------------------------
        score_f32 = tl.cast(prod, tl.float32) * scale
        score_f32 = score_f32 - max_score
        exp_score = tl.exp(score_f32)                        # (BLOCK_K,)
        sum_exp += tl.sum(exp_score, axis=0)                 # scalar

        # ----- load the latent values ---------------------------------
        # V has shape (B, L, Dv_lat)
        # We will accumulate over Dv_lat in BLOCK_DV sized chunks
        for start_d in range(0, Dv_lat, BLOCK_DV):
            cur_d = start_d + tl.arange(0, BLOCK_DV)
            dmask = cur_d < Dv_lat

            # (BLOCK_K, BLOCK_DV) offset matrix
            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(
                V_ptr + offs_v,
                mask=mask_k[:, None] & dmask[None, :],
                other=0.0
            )                                                   # (BLOCK_K, BLOCK_DV)

            weighted = v_slice * exp_score[:, None]            # broadcast exp_score
            acc += tl.sum(weighted, axis=0)                    # (BLOCK_DV,)

    # ------------------------------------------------------------------
    # 4️⃣ Normalise the latent accumulator
    # ------------------------------------------------------------------
    latent = acc / tl.cast(sum_exp, tl.bfloat16)               # (BLOCK_DV,)

    # ------------------------------------------------------------------
    # 5️⃣ Multiply with per‑head value‑projection matrix wV_T → v_head
    # ------------------------------------------------------------------
    v_head = tl.zeros([Dv], dtype=tl.bfloat16)                # (Dv,)

    for start_d in range(0, Dv_lat, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV)
        dmask = cur_d < Dv_lat

        # wV_T layout: (H, Dv_lat, Dv)
        offs_wV = (
            h * stride_wV_T_head
            + cur_d[:, None] * stride_wV_T_lat
            + tl.arange(0, Dv) * stride_wV_T_out
        )
        wV_block = tl.load(
            wV_T_ptr + offs_wV,
            mask=dmask[:, None] & (tl.arange(0, Dv) < Dv)[None, :],
            other=0.0
        )                                                       # (BLOCK_DV, Dv)

        lat_slice = latent[start_d:start_d + BLOCK_DV]         # (BLOCK_DV,)
        v_head += tl.sum(wV_block * lat_slice[:, None], axis=0)   # (Dv,)

    # ------------------------------------------------------------------
    # 6️⃣ Store per‑head output vector
    # ------------------------------------------------------------------
    offs_y = b * stride_y_batch + h * stride_y_head + tl.arange(0, Dv) * stride_y_dim
    tl.store(Y_ptr + offs_y, v_head, mask=(tl.arange(0, Dv) < Dv))


# ----------------------------------------------------------------------
# Fast‑path implementation (d_nope == 0)
# ----------------------------------------------------------------------
def _fast_forward(
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
    Fast‑path for the common case where the "no‑PE" head dimension is zero.
    Computes the MLM‑style attention using a custom Triton kernel that
    returns per‑head value vectors; the final linear projection (wO) is
    performed with a regular torch GEMM.
    """
    # ------------------------------------------------------------------
    # 0️⃣ Shape / convenience
    # ------------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    d  = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # ------------------------------------------------------------------
    # 1️⃣ Down‑project Q and KV (low‑rank)
    # ------------------------------------------------------------------
    x2 = x.squeeze(1)                     # (B, Dim)
    q_lora = F.linear(x2, wDQ)            # (B, dq)
    kv_lora0 = F.linear(x2, wDKV)         # (B, dkv + drope)

    # ------------------------------------------------------------------
    # 2️⃣ Update KV‑cache with the newly‑generated token
    # ------------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora0[:, :dkv]       # (B, dkv)
    rope_raw_new  = kv_lora0[:, dkv:]      # (B, drope)

    # RoPE rotation for the newly‑added key (position = cur_len)
    cos_k = cos_tbl[cur_len]               # (drope,)
    sin_k = sin_tbl[cur_len]               # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k   # (B, drope)

    # Store latent part and rotated rope part in the cache
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # ------------------------------------------------------------------
    # 3️⃣ Up‑project Q (low‑rank) and apply RoPE
    # ------------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                   # (B, nh*drope)
    q_up = q_up.view(bs, nh, drope)                # (B, nh, drope)

    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                         # (drope,)
    sin_q = sin_tbl[q_pos]                         # (drope,)
    q_rot = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # ------------------------------------------------------------------
    # 4️⃣ Gather K (rope‑rotated) and V (latent) from cache
    # ------------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]          # (B, L, dkv+drope)
    k_rope = kv_all[..., dkv:]                       # (B, L, drope)   – already rotated
    v_latent = kv_all[..., :dkv]                     # (B, L, dkv)

    # ------------------------------------------------------------------
    # 5️⃣ Prepare per‑head value‑projection matrix wV_T
    # ------------------------------------------------------------------
    # wUKV has shape (dv*nh, dkv)  →  (nh, dkv, dv) after reshape+permute
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # 6️⃣ Launch Triton kernel: compute per‑head value vectors
    # ------------------------------------------------------------------
    v_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # strides
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

    stride_y_batch = v_head.stride(0)
    stride_y_head  = v_head.stride(1)
    stride_y_dim   = v_head.stride(2)

    # launch configuration
    grid = (bs * nh,)

    BLOCK_K = 64          # positions per iteration (tuned for H200)
    BLOCK_DV = 32         # block size for latent accumulation

    scale = 1.0 / math.sqrt(drope)

    _triton_attn_vhead_kernel[grid](
        # pointers
        q_rot, k_rope, v_latent, wV_T, v_head,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len,   stride_k_dim,
        stride_v_batch, stride_v_len,   stride_v_dim,
        stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
        stride_y_batch, stride_y_head, stride_y_dim,
        # compile‑time constants
        bs, nh, new_len, drope, dkv, dv,
        scale,
        BLOCK_K, BLOCK_DV,
        # meta‑config
        num_warps=8, num_stages=4,
    )

    # ------------------------------------------------------------------
    # 7️⃣ Final output projection (single GEMM)
    # ------------------------------------------------------------------
    out = F.linear(v_head.view(bs, nh * dv), wO)   # (B, Dim)
    out = out.unsqueeze(1)                         # (B, 1, Dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Generic fallback (torch‑compile) – unchanged from the reference
# ----------------------------------------------------------------------
def _build_compiled_forward():
    """Compiled fallback implementation used when d_nope > 0."""
    import torch.nn.functional as F

    def _inner(x: torch.Tensor,
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
               dv: int):
        # Reference implementation (unchanged)
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
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Extract scalar config values (plain python ints)
    # --------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    d  = config.dim
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
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)  → (drope*nh, dq) when d_nope=0
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)    → (dv*nh, dkv) when d_nope=0
    wO   = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # Load / build RoPE tables (cached globally)
    # --------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope, config.max_seq_len, x.device)

    # --------------------------------------------------------------
    # Fast‑path when there is **no** "No‑PE" head dimension
    # --------------------------------------------------------------
    if d_nope == 0:
        return _fast_forward(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin
        )

    # --------------------------------------------------------------
    # General case – fall back to compiled reference implementation
    # --------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,                                   # (bs, 1, dim)
        kv_cache.data,                       # (bs, max_seq_len, dkv+drope)
        kv_cache.seq_len,                    # current cache length
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