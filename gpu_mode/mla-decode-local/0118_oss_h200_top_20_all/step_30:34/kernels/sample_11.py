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

# --------------------------------------------------------------
# Global cache for rotary tables (cos / sin)
# --------------------------------------------------------------
_cached_cos: torch.Tensor = None          # (max_seq_len, rope_dim)   bfloat16
_cached_sin: torch.Tensor = None          # (max_seq_len, rope_dim)   bfloat16

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

# --------------------------------------------------------------
# Helper: rotate‑half (same as the reference)
# --------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# Triton kernel – fused attention + per‑head value‑projection (K already rot‑ated)
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_vhead_kernel_no_rope_k(
    # --------------------------------------------------------------
    # Pointers
    # --------------------------------------------------------------
    Q_ptr,            # (B, H, Dq)         bfloat16 – already rotated
    K_ptr,            # (B, L, Dq)         bfloat16 – already rotated
    V_ptr,            # (B, L, Dv_lat)     bfloat16
    wV_T_ptr,         # (H, Dv_lat, Dv)    bfloat16
    Y_ptr,            # (B, H, Dv)         bfloat16

    # --------------------------------------------------------------
    # Strides (in elements, not bytes)
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len , stride_k_dim,
    stride_v_batch, stride_v_len, stride_v_dim,
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
    stride_y_batch, stride_y_head, stride_y_dv,

    # --------------------------------------------------------------
    # Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total number of heads
    L: tl.constexpr,          # current KV length (after insert)
    Dq: tl.constexpr,         # rope dimension (e.g. 64)
    Dv_lat: tl.constexpr,     # kv‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value‑dim (e.g. 128)
    scale: tl.constexpr,      # 1/sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,    # KV positions per tile
    BLOCK_DV: tl.constexpr,   # latent‑value tile size
):
    """
    Attention + per‑head value projection.
    K is already rotated, so we only perform dot‑product.
    """
    pid = tl.program_id(0)           # one program per batch element

    b = pid                         # batch index
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_mask  = head_range < H

    # --------------------------------------------------------------
    # Load Q – already rotated (one thread per head, vector of Dq)
    # --------------------------------------------------------------
    offs_q = (
        b * stride_q_batch
        + head_range[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_mask[:, None],
                 other=0.0)                     # (HEADS_PER_BLOCK, Dq)

    # --------------------------------------------------------------
    # Stable‑softmax accumulators (float32 for numerical stability)
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HPB,)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (HPB,)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, BLOCK_DV], dtype=tl.float32)  # (HPB, BLOCK_DV)

    # --------------------------------------------------------------
    # Main loop over KV positions (single‑pass softmax)
    # --------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)            # (BLOCK_K,)
        k_mask = cur_k < L

        # --------------------------------------------------------------
        # Load K block (already rotated)
        # --------------------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k = tl.load(K_ptr + offs_k,
                     mask=k_mask[:, None],
                     other=0.0)                      # (BLOCK_K, Dq)

        # --------------------------------------------------------------
        # Compute scores = Q · Kᵀ
        # --------------------------------------------------------------
        # q: (HPB, Dq), k: (BLOCK_K, Dq) -> term (HPB, BLOCK_K)
        term = tl.dot(q, k, trans_b=True)                # (HEADS_PER_BLOCK, BLOCK_K)
        score_f32 = tl.cast(term, tl.float32) * scale    # (HPB, BLOCK_K)

        # --------------------------------------------------------------
        # Stable‑softmax bookkeeping
        # --------------------------------------------------------------
        block_max = tl.max(score_f32, axis=1)                 # (HPB,)
        new_max   = tl.maximum(max_score, block_max)          # (HPB,)

        # rescale previous accumulators
        scale_factor = tl.exp(max_score - new_max)            # (HPB,)
        sum_exp   = sum_exp * scale_factor
        latent_acc = latent_acc * tl.cast(scale_factor, tl.float32)[:, None]

        exp_score_f32 = tl.exp(score_f32 - new_max[:, None])  # (HPB, BLOCK_K)
        sum_exp      = sum_exp + tl.sum(exp_score_f32, axis=1)

        # --------------------------------------------------------------
        # Accumulate weighted V (latent value) – fast dot
        # --------------------------------------------------------------
        exp_score_f32 = tl.cast(exp_score_f32, tl.float32)   # (HPB, BLOCK_K)

        for start_d in range(0, Dv_lat, BLOCK_DV):
            cur_d = start_d + tl.arange(0, BLOCK_DV)
            d_mask = cur_d < Dv_lat

            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(V_ptr + offs_v,
                              mask=k_mask[:, None] & d_mask[None, :],
                              other=0.0)                     # (BLOCK_K, BLOCK_DV)

            v_slice_f32 = tl.cast(v_slice, tl.float32)       # (BLOCK_K, BLOCK_DV)

            # latent_acc (HPB, BLOCK_DV) += exp_score_f32 @ v_slice_f32
            latent_acc += tl.dot(exp_score_f32, v_slice_f32) # (HPB, BLOCK_DV)

        max_score = new_max

    # --------------------------------------------------------------
    # Normalise the latent accumulator (divide by Σexp)
    # --------------------------------------------------------------
    latent = latent_acc / sum_exp[:, None]                     # (HPB, Dv_lat)  float32

    # --------------------------------------------------------------
    # Project latent → per‑head output (size Dv)
    # --------------------------------------------------------------
    y_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.bfloat16)

    for start_d in range(0, Dv_lat, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV)
        d_mask = cur_d < Dv_lat

        # wV_T slice: (HPB, BLOCK_DV, Dv)
        offs_wV = (
            head_range[:, None, None] * stride_wV_T_head
            + cur_d[None, :, None] * stride_wV_T_lat
            + tl.arange(0, Dv)[None, None, :] * stride_wV_T_out
        )
        wV_block = tl.load(wV_T_ptr + offs_wV,
                           mask=head_mask[:, None] & d_mask[None, :],
                           other=0.0)                     # (HPB, BLOCK_DV, Dv)

        lat_slice = latent[:, start_d:start_d + BLOCK_DV]      # (HPB, BLOCK_DV)

        # multiply‑accumulate over the latent dimension
        y_head += tl.sum(wV_block * lat_slice[:, :, None], axis=1)   # (HPB, Dv)

    # --------------------------------------------------------------
    # Store per‑head outputs
    # --------------------------------------------------------------
    offs_y = (
        b * stride_y_batch
        + head_range[:, None] * stride_y_head
        + tl.arange(0, Dv)[None, :] * stride_y_dv
    )
    tl.store(Y_ptr + offs_y,
             y_head,
             mask=head_mask[:, None])


# ----------------------------------------------------------------------
# Fast‑path (d_nope == 0) – rotates K once on insert, then runs the
# streamlined Triton kernel that does *not* rotate K on‑the‑fly.
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
    The heavy work (attention + per‑head value projection) is performed by a
    highly‑optimised Triton kernel that receives *already rotated* keys.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv   = config.kv_lora_rank
    dv    = config.v_head_dim

    # --------------------------------------------------------------
    # 1️⃣ Down‑projection (Q) and KV down‑projection (raw KV)
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                # (B, Dim)
    q_lora  = F.linear(x2, wDQ)       # (B, dq)
    kv_lora = F.linear(x2, wDKV)      # (B, dkv + drope)

    # --------------------------------------------------------------
    # 2️⃣ KV‑cache update – store latent part + *rotated* rope part
    # --------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora[:, :dkv]       # (B, dkv)
    rope_raw_new  = kv_lora[:, dkv:]       # (B, drope)

    # Rotate the new rope token using absolute position = cur_len
    cos_k = cos_tbl[cur_len]               # (drope,)
    sin_k = sin_tbl[cur_len]               # (drope,)
    rope_rot_new = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k   # (B, drope)

    # Write into cache (latent then rotated rope)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot_new
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 3️⃣ Up‑project Q and apply RoPE (single token)
    # --------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)               # (B, nh*drope)
    q_up = q_up.view(bs, nh, drope)           # (B, nh, drope)

    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                     # (drope,)
    sin_q = sin_tbl[q_pos]                     # (drope,)
    q_rot = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # --------------------------------------------------------------
    # 4️⃣ Gather KV from cache (latent + *already* rotated rope part)
    # --------------------------------------------------------------
    kv_all      = kv_cache.data[:, :new_len, :]          # (B, L, dkv+drope)
    k_rot       = kv_all[..., dkv:]                      # (B, L, drope) – already rotated
    v_latent    = kv_all[..., :dkv]                      # (B, L, dkv)

    # --------------------------------------------------------------
    # 5️⃣ Per‑head value‑projection matrix (transposed)
    # --------------------------------------------------------------
    # wUKV shape: ((d_nope + dv) * nh , dkv)  ; d_nope==0
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 6️⃣ Allocate output for each head
    # --------------------------------------------------------------
    y_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # --------------------------------------------------------------
    # 7️⃣ Compute strides (contiguous tensors)
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim = q_rot.stride()
    stride_k_batch, stride_k_len , stride_k_dim  = k_rot.stride()
    stride_v_batch, stride_v_len, stride_v_dim = v_latent.stride()
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out = wV_T.stride()
    stride_y_batch, stride_y_head, stride_y_dv = y_head.stride()

    # --------------------------------------------------------------
    # 8️⃣ Triton launch configuration (tuned for the 128‑head config)
    # --------------------------------------------------------------
    HEADS_PER_BLOCK = nh                         # we pack all heads
    BLOCK_K = 1024                               # positions per tile
    BLOCK_DV = 256 if dkv >= 256 else 128       # latent‑value tile size

    grid = (bs, )                                # one program per batch element

    scale = 1.0 / math.sqrt(drope)               # d_nope == 0

    _triton_attn_vhead_kernel_no_rope_k[grid](
        # pointers
        q_rot,
        k_rot,
        v_latent,
        wV_T,
        y_head,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len , stride_k_dim,
        stride_v_batch, stride_v_len, stride_v_dim,
        stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
        stride_y_batch, stride_y_head, stride_y_dv,
        # compile‑time constants
        bs, nh, new_len, drope, dkv, dv,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV,
        # launch config
        num_warps=16, num_stages=5,
    )

    # --------------------------------------------------------------
    # 9️⃣ Final linear projection
    # --------------------------------------------------------------
    y_head_flat = y_head.view(bs, nh * dv)          # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                 # (B, Dim)
    out = out.unsqueeze(1)                          # (B, 1, Dim)

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
        # ↓ same as the reference implementation in the prompt
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

    # ------------------------------------------------------------------
    #  Extract scalar config values
    # ------------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dim = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight        # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight          # ((d_nope + dv) * nh, dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

    # ------------------------------------------------------------------
    #  Build / fetch RoPE tables (cached globally)
    # ------------------------------------------------------------------
    global _cached_cos, _cached_sin
    if (_cached_cos is None) or (_cached_cos.shape[0] < config.max_seq_len):
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # ------------------------------------------------------------------
    #  Fast‑path – d_nope == 0 (the very common configuration)
    # ------------------------------------------------------------------
    if d_nope == 0:
        return _fast_forward_multihead_opt(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin,
        )

    # ------------------------------------------------------------------
    #  General case – use compiled reference implementation
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
    # update KVCache state with new length
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    # output already has shape [B, 1, Dim]
    return out, kv_cache.data