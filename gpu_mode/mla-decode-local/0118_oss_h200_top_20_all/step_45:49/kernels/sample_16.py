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
#  Global RoPE tables – lazily created on first use (cached for the whole process)
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)  bf16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)  bf16


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Create cosine / sine tables for the RoPE operation (identical to the reference)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
        torch.bfloat16
    )
    pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len, 1)
    idx = pos * theta  # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)  # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Utility: rotate the last dimension by half, exactly like the Python‐side fallback."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ----------------------------------------------------------------------
#  Triton kernel – fused scaled‑dot‑product, numerically‑stable softmax,
#  latent accumulation and per‑head value projection.
#
#  The kernel follows the “decode‑style” FlashAttention pattern:
#   * Q is loaded once per head‑tile (kept in registers)
#   * K / V are streamed block‑wise from global memory
#   * Stable softmax (max / sum‑exp) is performed on‑the‑fly
#   * The latent vector (size d_kv) is accumulated in FP32, then projected
#     with the per‑head output matrix (wV_T) inside the same kernel.
#
#  Autotuning chooses a good combination of HEADS_PER_BLOCK (how many heads
#  are processed together) and BLOCK_K (KV‑block size) for the given problem size.
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 256}, num_warps=8, num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 512}, num_warps=8, num_stages=4),
    ],
    key=["B", "H", "L", "Dq", "Dv_lat"],
)
@triton.jit
def _triton_mla_kernel(
    # ------------------------------------------------------------------
    #  Pointers
    # ------------------------------------------------------------------
    Q_ptr,               # (B, H, Dq)                 bf16
    K_ptr,               # (B, L, Dq)                 bf16
    V_ptr,               # (B, L, Dv_lat)             bf16
    W_ptr,               # (H, Dv_lat, Dv)            bf16
    Out_ptr,             # (B, H, Dv)                 bf16
    # ------------------------------------------------------------------
    #  Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len,  stride_k_dim,
    stride_v_batch, stride_v_len,  stride_v_dim,
    stride_w_head, stride_w_in,    stride_w_out,
    stride_out_batch, stride_out_head, stride_out_dim,
    # ------------------------------------------------------------------
    #  Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    L: tl.constexpr,          # KV length (current cache length)
    Dq: tl.constexpr,         # rope dimension (e.g. 64)
    Dv_lat: tl.constexpr,     # latent dimension (kv_lora_rank, e.g. 512)
    Dv: tl.constexpr,         # value‑head dimension (e.g. 128)
    SCALE: tl.constexpr,      # 1 / sqrt(Dq)   (d_nope == 0)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    1️⃣ Load Q‑vectors for all heads in the current tile (bf16 → fp32).
    2️⃣ Scan the KV cache block‑wise:
          • Compute scaled dot‑product Q·K.
          • Update numerically‑stable softmax statistics.
          • Accumulate the latent vector Σ softmax·V.
    3️⃣ After the scan, normalise the latent vector.
    4️⃣ Multiply the normalised latent vector with the per‑head value‑projection matrix.
    """
    # ------------------------------------------------------------------
    #  Program‑ID layout  →  (batch, head‑tile)
    # ------------------------------------------------------------------
    pid = tl.program_id(0)                       # linear id over (B * ceil(H/HEADS_PER_BLOCK))
    tiles_per_batch = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // tiles_per_batch                    # batch index
    tile = pid % tiles_per_batch                  # which head‑tile inside this batch
    head_start = tile * HEADS_PER_BLOCK           # first head handled by this program

    # ------------------------------------------------------------------
    #  Load Q for every head in the tile (once, kept in registers)
    # ------------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)               # (HEADS_PER_BLOCK,)
    head_valid = head_start + head_range < H                 # mask for the last incomplete tile
    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q, mask=head_valid[:, None], other=0.0)   # bf16
    q = tl.cast(q, tl.float32)                                      # fp32 for all further work

    # ------------------------------------------------------------------
    #  Running statistics for the stable soft‑max
    # ------------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)

    # accumulator for the latent vector (fp32) – one row per head
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dv_lat], dtype=tl.float32)

    # ------------------------------------------------------------------
    #  Main scan over the KV cache (BLOCK_K sized tiles)
    # ------------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)          # (BLOCK_K,)
        mask_k = cur_k < L

        # ----------- K  --------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[None, :] * stride_k_len                         # (1, BLOCK_K)
            + tl.arange(0, Dq)[:, None] * stride_k_dim              # (Dq, 1)
        )
        k = tl.load(K_ptr + offs_k,
                    mask=mask_k[None, :],
                    other=0.0,
                    cache_modifier='CA')                              # bf16 → fp32 internally

        # ----------- Q·K -------------------------------------------------
        score_mat = tl.dot(q, k, out_dtype=tl.float32)             # (HEADS_PER_BLOCK, BLOCK_K)
        score_f32 = score_mat * SCALE

        # ----------- stable soft‑max update -------------------------------
        block_max = tl.max(score_f32, axis=1)                       # (HEADS_PER_BLOCK)
        new_max = tl.maximum(max_score, block_max)                  # (HEADS_PER_BLOCK)

        # re‑scale the previously accumulated terms
        exp_factor = tl.exp(max_score - new_max)                    # (HEADS_PER_BLOCK)
        sum_exp = sum_exp * exp_factor
        latent_acc = latent_acc * exp_factor[:, None]

        exp_score = tl.exp(score_f32 - new_max[:, None])            # (HEADS_PER_BLOCK, BLOCK_K)

        # ----------- V --------------------------------------------------
        offs_v = (
            b * stride_v_batch
            + cur_k[:, None] * stride_v_len                         # (BLOCK_K, 1)
            + tl.arange(0, Dv_lat)[None, :] * stride_v_dim          # (1, Dv_lat)
        )
        v = tl.load(V_ptr + offs_v,
                    mask=mask_k[:, None],
                    other=0.0,
                    cache_modifier='CA')                              # bf16 → fp32 internally

        # ----------- latent accumulation -------------------------------
        # latent_acc (HEADS_PER_BLOCK, Dv_lat)  +=  exp_score (HEADS_PER_BLOCK, BLOCK_K) @ v (BLOCK_K, Dv_lat)
        latent_acc = latent_acc + tl.dot(exp_score, v, out_dtype=tl.float32)

        # Update running max
        max_score = new_max

    # ------------------------------------------------------------------
    #  Normalise the latent vector (soft‑max denominator)
    # ------------------------------------------------------------------
    latent = latent_acc / sum_exp[:, None]               # (HEADS_PER_BLOCK, Dv_lat)

    # ------------------------------------------------------------------
    #  Final per‑head projection  latent (Dv_lat) → out_head (Dv)
    # ------------------------------------------------------------------
    BLOCK_DV_OUT = 32                                    # 2 tiles for dv=128
    for d_start in tl.range(0, Dv, BLOCK_DV_OUT):
        d_end = d_start + BLOCK_DV_OUT
        d_mask = d_end < Dv

        # Load a slice of the per‑head weight matrix
        offs_w = (
            (head_start + head_range)[:, None] * stride_w_head
            + tl.arange(0, Dv_lat)[:, None] * stride_w_in
            + tl.arange(0, BLOCK_DV_OUT)[None, :] * stride_w_out
        )
        w_block = tl.load(
            W_ptr + offs_w,
            mask=head_valid[:, None] & d_mask[None, :],
            other=0.0,
        )                                                   # (HEADS_PER_BLOCK, Dv_lat, BLOCK_DV_OUT)

        # out_chunk = latent @ w_block  (batched GEMM)
        # Equivalent to sum(latent[:, :, None] * w_block, axis=1)
        out_chunk = tl.sum(latent[:, :, None] * w_block, axis=1)   # (HEADS_PER_BLOCK, BLOCK_DV_OUT)

        # Store the result
        offs_out = (
            b * stride_out_batch
            + (head_start + head_range)[:, None] * stride_out_head
            + tl.arange(0, BLOCK_DV_OUT)[None, :] * stride_out_dim
        )
        tl.store(
            Out_ptr + offs_out,
            tl.cast(out_chunk, tl.bfloat16),
            mask=head_valid[:, None] & d_mask[None, :],
        )


# ----------------------------------------------------------------------
#  Fast‑path for the common configuration (qk_nope_head_dim == 0)
# ----------------------------------------------------------------------
def _fast_forward_triton(
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
    This implements the forward pass for the case d_nope == 0.
    All heavy work (attention + per‑head output projection) is processed
    by the fused Triton kernel above.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim      # Dq
    dkv = config.kv_lora_rank            # Dv_lat
    dv = config.v_head_dim                # Dv
    dim = config.dim

    # ------------------------------------------------------------------
    #  1️⃣  Down‑project + KV‑cache update (still pure torch – cheap)
    # ------------------------------------------------------------------
    # x : (B, 1, dim) → squeeze temporal axis because linear layers expect 2‑D input
    x2 = x.squeeze(1)                                 # (B, dim)

    # project to the KV‑space (latent + rope part)
    kv_lora = F.linear(x2, wDKV)                     # (B, dkv + drope)
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    # split latent / rope part
    kv_lat_new   = kv_lora[:, :dkv]                  # (B, dkv)
    rope_raw_new = kv_lora[:, dkv:]                  # (B, drope)

    # ------------------------------------------------------------------
    #  2️⃣  Rotate the freshly created key‑rope part (position = cur_len)
    # ------------------------------------------------------------------
    cos_k = cos_tbl[cur_len]                         # (drope,)
    sin_k = sin_tbl[cur_len]                         # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k   # (B, drope)

    # write latent and rotated rope into the KV cache
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_lat_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # ------------------------------------------------------------------
    #  3️⃣  Query up‑projection + RoPE
    # ------------------------------------------------------------------
    q_lora = F.linear(x2, wDQ)                       # (B, dq)
    q = F.linear(q_lora, wUQ)                        # (B, nh * drope)
    q = q.view(bs, nh, drope)                       # (B, nh, drope)

    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                           # (drope,)
    sin_q = sin_tbl[q_pos]                           # (drope,)
    q = q * cos_q + _rotate_half(q) * sin_q          # (B, nh, drope)

    # ------------------------------------------------------------------
    #  4️⃣  Gather K and V from the cache (contiguous layout)
    # ------------------------------------------------------------------
    K = kv_cache.data[:, :new_len, dkv:]             # (B, L, drope)   bf16
    V = kv_cache.data[:, :new_len, :dkv]             # (B, L, dkv)     bf16

    # ------------------------------------------------------------------
    #  5️⃣  Call the fused Triton kernel
    # ------------------------------------------------------------------
    out_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # reshape per‑head value‑projection matrix:
    #   wUKV : ((d_nope+dv)*nh, dkv) → (nh, dv, dkv)  (note: d_nope == 0)
    wV_T = wUKV.view(nh, dv, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    # ------------------------------------------------------------------
    #  Strides – everything is contiguous, so we can reuse .stride()
    # ------------------------------------------------------------------
    stride_q_batch = q.stride(0)
    stride_q_head  = q.stride(1)
    stride_q_dim   = q.stride(2)

    stride_k_batch = K.stride(0)
    stride_k_len   = K.stride(1)
    stride_k_dim   = K.stride(2)

    stride_v_batch = V.stride(0)
    stride_v_len   = V.stride(1)
    stride_v_dim   = V.stride(2)

    stride_w_head = wV_T.stride(0)
    stride_w_in   = wV_T.stride(1)
    stride_w_out  = wV_T.stride(2)

    stride_out_batch = out_head.stride(0)
    stride_out_head  = out_head.stride(1)
    stride_out_dim   = out_head.stride(2)

    # ------------------------------------------------------------------
    #  Grid layout (one program per (batch, head‑tile))
    # ------------------------------------------------------------------
    grid = (bs * ((nh + 31) // 32),)   # fallback grid size – Triton will pick the best autotuned variant

    _triton_mla_kernel[grid](
        Q_ptr=q,
        K_ptr=K,
        V_ptr=V,
        W_ptr=wV_T,
        Out_ptr=out_head,
        stride_q_batch=stride_q_batch,
        stride_q_head=stride_q_head,
        stride_q_dim=stride_q_dim,
        stride_k_batch=stride_k_batch,
        stride_k_len=stride_k_len,
        stride_k_dim=stride_k_dim,
        stride_v_batch=stride_v_batch,
        stride_v_len=stride_v_len,
        stride_v_dim=stride_v_dim,
        stride_w_head=stride_w_head,
        stride_w_in=stride_w_in,
        stride_w_out=stride_w_out,
        stride_out_batch=stride_out_batch,
        stride_out_head=stride_out_head,
        stride_out_dim=stride_out_dim,
        B=bs,
        H=nh,
        L=new_len,
        Dq=drope,
        Dv_lat=dkv,
        Dv=dv,
        SCALE=1.0 / math.sqrt(drape if (drape := drope) else 1.0),  # sqrt(d_nope + d_rope) – d_nope == 0
        # NOTE: HEADS_PER_BLOCK and BLOCK_K are NOT passed explicitly;
        # Triton will select the best autotuned configuration.
    )

    # ------------------------------------------------------------------
    #  6️⃣  Final linear projection back to model dimension
    # ------------------------------------------------------------------
    out_head_flat = out_head.view(bs, nh * dv)        # (B, nh*dv)
    out = F.linear(out_head_flat, wO)                 # (B, dim)
    out = out.unsqueeze(1)                            # (B, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
#  Helper: compiled fallback (used only when d_nope != 0)
# ----------------------------------------------------------------------
_compiled_forward = None
def _build_compiled_forward():
    """
    The reference implementation compiled with torch.compile – kept as a
    safety net for the rare case d_nope > 0.
    """
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
        # ----- identical to the reference implementation (unchanged) -----
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
#  Main entry point required by the benchmark harness
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    entry point: receives (config, x, kv_cache) and returns (output, updated_kv_cache)
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    #  Extract scalar config values (plain Python ints)
    # ------------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dim = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # ------------------------------------------------------------------
    #  Weight tensors – already on the correct device & dtype
    # ------------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    #  Build / fetch RoPE tables (cached globally)
    # ------------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope, config.max_seq_len, x.device)

    # ------------------------------------------------------------------
    #  Fast‑path – most common configuration (no‑PE part disabled)
    # ------------------------------------------------------------------
    if d_nope == 0:
        out, new_kv = _fast_forward_triton(
            config,
            x,
            kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos,
            _cached_sin,
        )
        # kv_cache is already updated inside the fast‑path function
        return out, new_kv

    # ------------------------------------------------------------------
    #  General case – fallback to the compiled reference implementation
    # ------------------------------------------------------------------
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