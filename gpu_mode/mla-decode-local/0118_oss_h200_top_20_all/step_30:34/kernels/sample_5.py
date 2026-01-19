### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config   # Must import this way
### END OF IMPORT STATEMENTS BLOCK ###

# ------------------------------------------------------------
# Helper – rotate‑half (same as the reference)
# ------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half (RoPE helper)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ------------------------------------------------------------
# RoPE tables – cached globally (cos / sin)
# ------------------------------------------------------------
_cached_cos: torch.Tensor = None          # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None          # (max_seq_len, rope_dim)  bfloat16


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Create (cos, sin) tables for rotary embeddings (bfloat16)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half))
    theta = theta.to(torch.bfloat16)                       # (half,)
    pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
    idx = pos * theta                                       # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                     # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ------------------------------------------------------------
# Triton kernel – fused attention + per‑head value‑projection (fast‑path)
# ------------------------------------------------------------
@triton.jit
def _triton_attn_vhead_kernel_multihead_opt(
    # --------------------------------------------------------------
    #   Pointers
    # --------------------------------------------------------------
    Q_ptr,                # (B, H, Dq)                 bfloat16
    K_ptr,                # (B, L, Dq)                 bfloat16
    V_ptr,                # (B, L, Dv_lat)             bfloat16
    wV_T_ptr,             # (H, Dv_lat, Dv)            bfloat16
    Y_ptr,                # (B, H, Dv)                 bfloat16

    # --------------------------------------------------------------
    #   Strides
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,   # Q    (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,   # K    (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,   # V    (B, L, Dv_lat)

    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,  # wV_T (H, Dv_lat, Dv)

    stride_y_batch, stride_y_head, stride_y_dv,           # Y    (B, H, Dv)

    # --------------------------------------------------------------
    #   Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    Dq: tl.constexpr,         # rope dimension (e.g. 64)
    Dv_lat: tl.constexpr,     # latent value dimension (kv‑lora rank, e.g. 512)
    Dv: tl.constexpr,         # per‑head value dim (e.g. 128)
    scale: tl.constexpr,      # 1/sqrt(D_tot)
    HEADS_PER_BLOCK: tl.constexpr,    # heads processed per program
    BLOCK_K: tl.constexpr,    # positions processed per iteration (power‑of‑2)
    BLOCK_DV: tl.constexpr,   # block size for the latent value accumulation (set to Dv_lat)

    # --------------------------------------------------------------
    #   Runtime argument
    # --------------------------------------------------------------
    L                      # current KV length (runtime)
):
    """
    Triton implementation for the common case ``qk_nope_head_dim == 0``.
    Performs a numerically‑stable softmax, accumulates the latent
    values and finally projects them to the per‑head output space.
    """
    pid = tl.program_id(0)

    # --------------------------------------------------------------
    #   Tile indexing
    # --------------------------------------------------------------
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # head‑tile inside the batch
    head_start = tile * HEADS_PER_BLOCK             # first head handled by this program

    # --------------------------------------------------------------
    #   Load Q for the handled heads (once per program)
    # --------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_mask = (head_start + head_range) < H

    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_mask[:, None],
                 other=0.0)                         # (HEADS_PER_BLOCK, Dq)

    # --------------------------------------------------------------
    #   Accumulators for stable‑softmax
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HPB,)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (HPB,)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, BLOCK_DV], dtype=tl.bfloat16)

    # --------------------------------------------------------------
    #   Main loop over K‑blocks (single‑pass softmax)
    # --------------------------------------------------------------
    for start_k in tl.range(0, L, BLOCK_K, num_stages=2):
        cur_k = start_k + tl.arange(0, BLOCK_K)           # (BLOCK_K,)
        k_mask = cur_k < L

        # ---- load K block -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0)                     # (BLOCK_K, Dq)

        # ---- compute raw scores via dot product ----------------------------
        prod = tl.dot(q, k_block, trans_b=True)          # bf16, (HPB, BLOCK_K)
        score_f32 = tl.cast(prod, tl.float32) * scale    # apply sqrt scaling

        # ---- block‑wise max ------------------------------------------------
        block_max = tl.max(score_f32, axis=1)            # (HPB,)

        # ---- update running max -------------------------------------------
        new_max = tl.maximum(max_score, block_max)       # (HPB,)

        # ---- scaling factor for the old accumulated values ---------------
        scale_factor = tl.exp(max_score - new_max)       # (HPB,)
        scale_factor_f16 = tl.cast(scale_factor, tl.bfloat16)

        sum_exp = sum_exp * scale_factor
        latent_acc = latent_acc * scale_factor_f16[:, None]

        # ---- exponentiate scores with the new max -------------------------
        exp_score_f32 = tl.exp(score_f32 - new_max[:, None])   # (HPB, BLOCK_K)
        exp_score = tl.cast(exp_score_f32, tl.bfloat16)        # bf16

        # ---- update denominator of softmax --------------------------------
        sum_exp = sum_exp + tl.sum(exp_score, axis=1)     # (HPB,)

        # ---- load V block (full Dv_lat) and accumulate --------------------
        offs_v = (
            b * stride_v_batch
            + cur_k[:, None] * stride_v_len
            + tl.arange(0, Dv_lat)[None, :] * stride_v_dim
        )
        v_slice = tl.load(V_ptr + offs_v,
                          mask=k_mask[:, None],
                          other=0.0)                # (BLOCK_K, Dv_lat)

        # weighted accumulation: (HPB, BLOCK_K) @ (BLOCK_K, Dv_lat)
        latent_acc += tl.dot(exp_score, v_slice)         # (HPB, Dv_lat)

        # ---- commit new max for the next iteration -----------------------
        max_score = new_max

    # --------------------------------------------------------------
    #   Normalise the latent accumulator (divide by Σexp)
    # --------------------------------------------------------------
    latent = latent_acc / tl.cast(sum_exp[:, None], tl.bfloat16)   # (HPB, Dv_lat)

    # --------------------------------------------------------------
    #   Project latent → per‑head output (size Dv)
    # --------------------------------------------------------------
    y_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.bfloat16)

    # Load the per‑head projection matrix (H, Dv_lat, Dv)
    cur_d = tl.arange(0, BLOCK_DV)       # (Dv_lat,)
    offs_wV = (
        (head_start + head_range)[:, None, None] * stride_wV_T_head
        + cur_d[None, :, None] * stride_wV_T_lat
        + tl.arange(0, Dv)[None, None, :] * stride_wV_T_out
    )
    wV_block = tl.load(wV_T_ptr + offs_wV,
                       mask=head_mask[:, None] & (cur_d[None, :] < Dv_lat),
                       other=0.0)                     # (HPB, Dv_lat, Dv)

    # y_head += (latent @ wV)   → sum over latent dimension
    y_head += tl.sum(wV_block * latent[:, :, None], axis=1)

    # --------------------------------------------------------------
    #   Store per‑head output
    # --------------------------------------------------------------
    offs_y = (
        b * stride_y_batch
        + (head_start + head_range)[:, None] * stride_y_head
        + tl.arange(0, Dv)[None, :] * stride_y_dv
    )
    tl.store(Y_ptr + offs_y,
             y_head,
             mask=head_mask[:, None])


# ----------------------------------------------------------------------
#  Fast‑path (qk_nope_head_dim == 0) – fully‑fused implementation
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
    Optimised forward for the common case ``qk_nope_head_dim == 0``.
    All heavy work (KV‑cache handling, RoPE, attention + per‑head value
    projection) is performed by a custom Triton kernel.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim          # rope dimension (Dq)
    dkv  = config.kv_lora_rank                # latent value dimension (Dv_lat)
    dv   = config.v_head_dim                  # per‑head value dimension (Dv)
    dim  = config.dim

    # ------------------------------------------------------------------
    #   1️⃣ Down‑project (Q) + KV‑down
    # ------------------------------------------------------------------
    x2 = x.squeeze(1)                          # (B, dim)
    q_lora = F.linear(x2, wDQ)                 # (B, dq)
    kv_lora0 = F.linear(x2, wDKV)              # (B, dkv + drope)

    # ------------------------------------------------------------------
    #   2️⃣ KV‑cache update (store latent + rotated key)
    # ------------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora0[:, :dkv]               # (B, dkv)
    rope_raw_new   = kv_lora0[:, dkv:]              # (B, drope)

    # Rotate the new key using pre‑computed RoPE tables (position = cur_len)
    cos_k = cos_tbl[cur_len]                        # (drope,)
    sin_k = sin_tbl[cur_len]                        # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k   # (B, drope)

    # Write the new token into the cache (contiguous layout)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # ------------------------------------------------------------------
    #   3️⃣ Up‑project Q and apply RoPE
    # ------------------------------------------------------------------
    # Q_up: (B, nh*drope)
    q_up = F.linear(q_lora, wUQ)                      # (B, nh*drope)
    q_up = q_up.view(bs, nh, drope)                  # (B, nh, drope)

    # Rotate the query (position = new_len‑1)
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                            # (drope,)
    sin_q = sin_tbl[q_pos]                            # (drope,)
    q_rope = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # ------------------------------------------------------------------
    #   4️⃣ Gather full KV from the cache (latent + rotated keys)
    # ------------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]            # (B, L, dkv + drope)
    k_rope = kv_all[..., dkv:]                       # (B, L, drope)   ← already rotated
    v_latent = kv_all[..., :dkv]                     # (B, L, dkv)

    # ------------------------------------------------------------------
    #   5️⃣ Prepare per‑head value‑projection matrix (transpose)
    # ------------------------------------------------------------------
    # wUKV shape: (nh*dv, dkv)  because d_nope == 0
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)

    # ------------------------------------------------------------------
    #   6️⃣ Triton kernel – attention + per‑head value projection
    # ------------------------------------------------------------------
    y_head = torch.empty((bs, nh, dv),
                         dtype=torch.bfloat16,
                         device=x.device)

    # strides – all tensors are contiguous, reuse .stride()
    stride_q_batch = q_rope.stride(0)
    stride_q_head  = q_rope.stride(1)
    stride_q_dim   = q_rope.stride(2)

    stride_k_batch = k_rope.stride(0)
    stride_k_len   = k_rope.stride(1)
    stride_k_dim   = k_rope.stride(2)

    stride_v_batch = v_latent.stride(0)
    stride_v_len   = v_latent.stride(1)
    stride_v_dim   = v_latent.stride(2)

    stride_wV_T_head = wV_T.stride(0)
    stride_wV_T_lat  = wV_T.stride(1)
    stride_wV_T_out  = wV_T.stride(2)

    stride_y_batch = y_head.stride(0)
    stride_y_head  = y_head.stride(1)
    stride_y_dv    = y_head.stride(2)

    # --------------------------------------------------------------
    #   Tuning parameters – empirically chosen for the reference config.
    # --------------------------------------------------------------
    HEADS_PER_BLOCK = 64          # more heads per program (reduces grid size)
    BLOCK_K         = 512         # positions per block (power‑of‑2)
    BLOCK_DV        = dkv         # full latent dimension (no inner loop)

    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    scale = 1.0 / math.sqrt(drope)    # because d_nope == 0

    _triton_attn_vhead_kernel_multihead_opt[grid](
        # pointers
        q_rope, k_rope, v_latent,
        wV_T, y_head,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len, stride_k_dim,
        stride_v_batch, stride_v_len, stride_v_dim,
        stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
        stride_y_batch, stride_y_head, stride_y_dv,
        # compile‑time constants
        bs, nh, drope, dkv, dv,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV,
        # runtime KV length
        new_len,
        # launch config (tuned)
        num_warps=8, num_stages=4,
    )

    # ------------------------------------------------------------------
    #   7️⃣ Final output projection (WO) – standard GEMM
    # ------------------------------------------------------------------
    y_head_flat = y_head.view(bs, nh * dv)          # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                 # (B, dim)
    out = out.unsqueeze(1)                          # (B, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
#  Compiled fallback (general case – d_nope > 0)
# ----------------------------------------------------------------------
_compiled_forward: torch._dynamo.eval_frame.OptimizedModule = None


def _build_compiled_forward():
    """
    Build a Torch‑compile fallback for the case ``qk_nope_head_dim > 0``.
    The reference implementation is left unchanged – torch.compile
    will optimise the graph (operator fusion, layout‑optimisations,
    etc.).
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
        # ---------- Reference implementation (unchanged) ----------
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
#  Main entry point (custom_kernel)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Entry point used by the benchmark harness.
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    #   Extract scalar config values (plain python ints)
    # ------------------------------------------------------------------
    bs    = config.batch_size
    nh    = config.n_heads
    dim   = config.dim
    dq    = config.q_lora_rank
    dkv   = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv    = config.v_head_dim

    # ------------------------------------------------------------------
    #   Weight tensors – already stored as bfloat16 tensors on the device
    # ------------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh , dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope + dv) * nh , dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

    # ------------------------------------------------------------------
    #   Build / fetch RoPE tables (cached globally)
    # ------------------------------------------------------------------
    global _cached_cos, _cached_sin
    if (_cached_cos is None) or (_cached_cos.shape[0] < config.max_seq_len):
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # ------------------------------------------------------------------
    #   Fast‑path when ``qk_nope_head_dim == 0``
    # ------------------------------------------------------------------
    if d_nope == 0:
        # Fast path returns (output, updated_kv_cache)
        return _fast_forward_multihead_opt(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin,
        )

    # ------------------------------------------------------------------
    #   General case – compiled fallback (d_nope > 0)
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
    # update KVCache state
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    # The compiled fallback already returns shape [B, 1, Dim].
    return out, kv_cache.data