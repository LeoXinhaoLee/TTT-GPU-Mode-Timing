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
# Utilities – RoPE rotate‑half & cached trig tables
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half (identical to reference)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


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
# Triton kernel – latent attention (one‑pass, stable softmax)
# ----------------------------------------------------------------------
@triton.jit
def _triton_latent_onepass(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bfloat16
    K_ptr,                # (B, L, Dq)                 bfloat16
    V_ptr,                # (B, L, Dkv)                bfloat16
    LAT_ptr,              # (B, H, Dkv)                bfloat16   (latent output)

    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,   # Q    (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,   # K    (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,   # V    (B, L, Dkv)

    stride_lat_batch, stride_lat_head, stride_lat_dim, # LAT  (B, H, Dkv)

    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    L: tl.constexpr,          # current KV length
    Dq: tl.constexpr,         # rope head dimension (e.g. 64)
    Dkv: tl.constexpr,        # KV‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value dim (unused here but kept for API compatibility)
    scale: tl.constexpr,      # 1/sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Computes the **latent** attention output:
        latent = softmax(Q·Kᵀ) @ V
    No projection is performed inside the kernel – the per‑head value projection
    and final linear projection are handled outside for maximum efficiency.
    """
    pid = tl.program_id(0)

    # ---------------------------------------------------------------
    # 0️⃣ Identify batch and head‑tile this program works on
    # ---------------------------------------------------------------
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles
    tile = pid % num_head_tiles
    head_start = tile * HEADS_PER_BLOCK

    # ---------------------------------------------------------------
    # 1️⃣ Load Q‑vectors for the heads in this tile (once)
    # ---------------------------------------------------------------
    hs = tl.arange(0, HEADS_PER_BLOCK)                     # (HEADS_PER_BLOCK,)
    head_valid = head_start + hs < H                        # mask for the last tile

    offs_q = (
        b * stride_q_batch
        + (head_start + hs)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)               # (HEADS_PER_BLOCK, Dq)
    q = tl.cast(q, tl.float32)

    # ---------------------------------------------------------------
    # 2️⃣ Initialise running max, normaliser and latent accumulator
    # ---------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)

    # ---------------------------------------------------------------
    # 3️⃣ Scan over K / V blocks (single pass)
    # ---------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)          # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- load K block -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0)                 # (BLOCK_K, Dq)
        k_block = tl.cast(k_block, tl.float32)

        # ----- Q·K dot‑product (scaled) ------------------------------------
        prod = tl.dot(q, tl.trans(k_block), out_dtype=tl.float32)   # (HEADS_PER_BLOCK, BLOCK_K)
        prod = prod * scale

        # ----- Update running max (stable softmax) -------------------------
        block_max = tl.max(prod, axis=1)                # (HEADS_PER_BLOCK,)
        new_max   = tl.maximum(max_score, block_max)

        # factor to rescale the previous accumulators
        scale_prev = tl.exp(max_score - new_max)        # (HEADS_PER_BLOCK,)

        # ----- Rescale previous state ---------------------------------------
        sum_exp   = sum_exp * scale_prev
        latent_acc = latent_acc * tl.broadcast_to(scale_prev[:, None], [HEADS_PER_BLOCK, Dkv])

        # ----- exponentials of the current block -----------------------------
        exp_scores = tl.exp(prod - new_max[:, None])   # (HEADS_PER_BLOCK, BLOCK_K)

        # ----- update normaliser ---------------------------------------------
        sum_exp = sum_exp + tl.sum(exp_scores, axis=1)   # (HEADS_PER_BLOCK,)

        # ----- weighted accumulation of V (latent vectors) -------------------
        for start_d in range(0, Dkv, BLOCK_DV):
            cur_d = start_d + tl.arange(0, BLOCK_DV)
            d_mask = cur_d < Dkv

            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(V_ptr + offs_v,
                              mask=k_mask[:, None] & d_mask[None, :],
                              other=0.0)               # (BLOCK_K, BLOCK_DV)
            v_slice = tl.cast(v_slice, tl.float32)

            # exp_scores (HEADS, BLOCK_K)  ×  v_slice (BLOCK_K, BLOCK_DV)
            # → (HEADS, BLOCK_DV)   – tensor‑core accelerated
            acc = tl.dot(exp_scores, v_slice, out_dtype=tl.float32)

            latent_acc[:, start_d:start_d + BLOCK_DV] = \
                latent_acc[:, start_d:start_d + BLOCK_DV] + acc

        # ----- store new max for the next iteration -------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # 4️⃣ Normalise the latent accumulator (still FP32)
    # ------------------------------------------------------------------
    latent = latent_acc / sum_exp[:, None]

    # ------------------------------------------------------------------
    # 5️⃣ Store latent (cast back to bfloat16)
    # ------------------------------------------------------------------
    offs_lat = (
        b * stride_lat_batch
        + (head_start + hs)[:, None] * stride_lat_head
        + tl.arange(0, Dkv)[None, :] * stride_lat_dim
    )
    tl.store(LAT_ptr + offs_lat,
             tl.cast(latent, tl.bfloat16),
             mask=head_valid[:, None])


# ----------------------------------------------------------------------
# Compiled fallback – unchanged (used when d_nope > 0)
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

    # ------------------------------------------------------------------
    # Extract scalar config values (plain python ints)
    # ------------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    d_nope = config.qk_nope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim

    # ------------------------------------------------------------------
    # Grab raw weight tensors (already on device, bfloat16)
    # ------------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    # Build / fetch RoPE tables (cached globally)
    # ------------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # ------------------------------------------------------------------
    # Fast‑path – d_nope == 0 (the common configuration)
    # ------------------------------------------------------------------
    if d_nope == 0:
        # --------------------------------------------------------------
        # Pre‑compute merged Q‑up+down + KV‑down weight (if not present)
        # --------------------------------------------------------------
        if not hasattr(config, "combined_Q_proj_weight"):
            # Q_up @ Q_down  →  (nh*drope, dim)
            with torch.no_grad():
                config.combined_Q_proj_weight = torch.mm(wUQ.float(),
                                                        wDQ.float()).to(torch.bfloat16)

        if not hasattr(config, "combined_all_weight"):
            # Concatenate Q (rope‑only) and KV‑down together
            config.combined_all_weight = torch.cat(
                [config.combined_Q_proj_weight, wDKV], dim=0
            ).contiguous()

        w_all = config.combined_all_weight

        # --------------------------------------------------------------
        # ONE GEMM for Q + KV‑down
        # --------------------------------------------------------------
        with torch.no_grad():
            x2 = x.squeeze(1)                      # (bs, dim)
            all_proj = F.linear(x2, w_all)         # (bs, nh*drope + dkv + drope)

            total_q_dim = nh * drope
            q_all = all_proj[:, :total_q_dim]                 # (bs, nh*drope)
            kv_all = all_proj[:, total_q_dim:]                # (bs, dkv + drope)

            # ----------------------------------------------------------
            # Reshape queries (rope part only) – shape (bs, nh, drope)
            # ----------------------------------------------------------
            q_all = q_all.view(bs, nh, drope)                 # (bs, nh, drope)

            # ----------------------------------------------------------
            # KV split – latent part + raw rope part
            # ----------------------------------------------------------
            kv_latent_new = kv_all[:, :dkv]          # (bs, dkv)
            rope_raw_new   = kv_all[:, dkv:]         # (bs, drope)

            # ----------------------------------------------------------
            # Write new token into KV cache (latent + rotated key)
            # ----------------------------------------------------------
            cur_len = kv_cache.seq_len
            new_len = cur_len + 1

            # RoPE rotation for the **new key**
            cos_k = _cached_cos[cur_len]            # (drope,)
            sin_k = _cached_sin[cur_len]            # (drope,)
            rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k   # (bs, drope)

            kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
            kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
            kv_cache.seq_len = new_len
            query_pos = new_len - 1

            # ----------------------------------------------------------
            # RoPE rotation for queries (current position)
            # ----------------------------------------------------------
            cos_q = _cached_cos[query_pos]          # (drope,)
            sin_q = _cached_sin[query_pos]          # (drope,)
            q_rot = q_all * cos_q + _rotate_half(q_all) * sin_q   # (bs, nh, drope)

            # ----------------------------------------------------------
            # Prepare tensors for scaled‑dot‑product attention
            # ----------------------------------------------------------
            # Q: (bs, nh, 1, drope)
            q = q_rot[:, :, None, :]   # adds seq_len=1 dimension

            # K & V share the same cache across heads – broadcast on head dim
            # K: (bs, 1, L, drope)
            k = kv_cache.data[..., dkv:].unsqueeze(1)   # (bs, 1, L, drope)
            # V: (bs, 1, L, dkv)
            v = kv_cache.data[..., :dkv].unsqueeze(1)   # (bs, 1, L, dkv)

            # ----------------------------------------------------------
            # Flash‑style attention (no causal mask – all past keys are visible)
            # ----------------------------------------------------------
            latent = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )   # (bs, nh, 1, dkv)
            latent = latent.squeeze(2)   # (bs, nh, dkv)

            # ----------------------------------------------------------
            # Per‑head value projection (latent → dv)
            # ----------------------------------------------------------
            latent_flat = latent.reshape(bs * nh, dkv)          # (bs*nh, dkv)
            v_proj = F.linear(latent_flat, wUKV)               # (bs*nh, dv)
            v_proj = v_proj.view(bs, nh, dv)                  # (bs, nh, dv)

            # ----------------------------------------------------------
            # Final linear projection
            # ----------------------------------------------------------
            v_proj_flat = v_proj.reshape(bs, nh * dv)         # (bs, nh*dv)
            out = F.linear(v_proj_flat, wO)                   # (bs, dim)
            out = out.unsqueeze(1)                             # (bs, 1, dim)

        return out, kv_cache.data

    # ------------------------------------------------------------------
    # General case – fall back to compiled reference implementation
    # ------------------------------------------------------------------
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

    # The compiled fallback already returns shape [B, 1, Dim].
    return out, kv_cache.data