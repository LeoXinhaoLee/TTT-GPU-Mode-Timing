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
# Global caches (shared across kernel calls) – never re‑allocated
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None          # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None          # (max_seq_len, rope_dim)  bfloat16
_cached_wq_fused: torch.Tensor = None     # (nh*drope, dim)          bfloat16
_cached_wV_T: torch.Tensor = None         # (nh, dkv, dv)            bfloat16

# ----------------------------------------------------------------------
# Helper functions (identical to the reference implementation)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half (used by RoPE)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Pre‑compute cosine / sine tables for rotary positional embeddings (bfloat16)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half,
                                        dtype=torch.float32,
                                        device=device) / half)).to(torch.bfloat16)
    pos = torch.arange(max_seq_len,
                       dtype=torch.int64,
                       device=device).unsqueeze_(1)   # (max_seq_len, 1)
    idx = pos * theta                                          # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                        # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)

# ----------------------------------------------------------------------
# Triton kernel – fused attention with stable softmax and per‑head
# value projection (dot‑product version, using Tensor‑cores where possible)
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        # (keep the original autotune configs – the best one will be selected)
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 512, "BLOCK_DV": 128},
                      num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 256, "BLOCK_DV": 128},
                      num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 256, "BLOCK_DV":  64},
                      num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 128, "BLOCK_DV":  64},
                      num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 512, "BLOCK_DV": 128},
                      num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 256, "BLOCK_DV": 128},
                      num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 256, "BLOCK_DV":  64},
                      num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 128, "BLOCK_DV":  64},
                      num_warps=8,  num_stages=4),
        # Extra configs – the autotuner may pick them on H200
        triton.Config({"HEADS_PER_BLOCK": 128, "BLOCK_K": 1024, "BLOCK_DV": 256},
                      num_warps=8, num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 128, "BLOCK_K":  512, "BLOCK_DV": 256},
                      num_warps=8, num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 128, "BLOCK_K":  512, "BLOCK_DV": 128},
                      num_warps=8, num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 128, "BLOCK_K":  256, "BLOCK_DV": 256},
                      num_warps=8, num_stages=4),
        triton.Config({"HEADS_PER_BLOCK":  64, "BLOCK_K": 1024, "BLOCK_DV": 256},
                      num_warps=8, num_stages=4),
    ],
    key=[
        "B", "H", "L", "Dq", "Dv_lat", "Dv"
    ],
)
@triton.jit
def _triton_attn_vhead_dot_kernel(
    # --------------------------------------------------------------
    # Pointers
    # --------------------------------------------------------------
    Q_ptr,               # (B, H, Dq)                     bf16
    K_ptr,               # (B, L, Dq)                     bf16
    V_ptr,               # (B, L, Dv_lat)                 bf16
    wV_T_ptr,            # (H, Dv_lat, Dv)                bf16
    Vhead_ptr,           # (B, H, Dv)                     bf16 (output)

    # --------------------------------------------------------------
    # Strides
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,      # Q   (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,      # K   (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,      # V   (B, L, Dv_lat)

    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,  # wV_T (H, Dv_lat, Dv)

    stride_vhead_batch, stride_vhead_head, stride_vhead_out,  # Vhead (B, H, Dv)

    # --------------------------------------------------------------
    # Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    L: tl.constexpr,          # KV length
    Dq: tl.constexpr,         # rope dim (e.g. 64)
    Dv_lat: tl.constexpr,     # kv‑lora rank (e.g. 512)
    Dv: tl.constexpr,         # per‑head value dim (e.g. 128)
    scale: tl.constexpr,      # 1/sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Stable‑softmax attention fused with per‑head latent → output projection.
    This version leverages `tl.dot` (Tensor‑cores) for both Q·K and
    (exp·V) matrix multiplications.
    """

    pid = tl.program_id(0)                         # linear id across (batch, head‑tiles)
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # head‑tile index inside batch
    head_start = tile * HEADS_PER_BLOCK              # first head handled by this program

    # --------------------------------------------------------------
    # Load the query vectors for this head‑tile (HEADS_PER_BLOCK × Dq)
    # --------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_valid = head_start + head_range < H

    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_valid[:, None],
                 other=0.0)                     # (HEADS_PER_BLOCK, Dq)  bf16

    # --------------------------------------------------------------
    # Buffers for stable‑softmax
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HEADS_PER_BLOCK)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)           # (HEADS_PER_BLOCK)
    acc       = tl.zeros([HEADS_PER_BLOCK, Dv_lat], dtype=tl.float32)   # (HEADS_PER_BLOCK, Dv_lat)

    # --------------------------------------------------------------
    # Main loop over KV blocks
    # --------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)               # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- K block -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0,
                          cache_modifier='CA')                     # (BLOCK_K, Dq) bf16

        # ----- dot(Q, K^T) ---------------------------------------------
        # q : (HEADS_PER_BLOCK, Dq)    bf16
        # k_block : (BLOCK_K, Dq)      bf16
        # tl.dot with trans_b=True treats k_block as (Dq, BLOCK_K)
        score = tl.dot(q, k_block, trans_b=True)                        # (HEADS_PER_BLOCK, BLOCK_K) bf16
        score_f32 = tl.cast(score, tl.float32) * scale                  # (HEADS_PER_BLOCK, BLOCK_K)

        # ----- update running max / normalisation ----------------------
        block_max = tl.max(score_f32, axis=1)                          # (HEADS_PER_BLOCK)
        new_max   = tl.maximum(max_score, block_max)                    # (HEADS_PER_BLOCK)

        exp_factor = tl.exp(max_score - new_max)                        # (HEADS_PER_BLOCK) ≤ 1
        sum_exp = sum_exp * exp_factor
        acc = acc * exp_factor[:, None]

        # ----- exponentials for the current block ----------------------
        exp_score = tl.exp(score_f32 - new_max[:, None])               # (HEADS_PER_BLOCK, BLOCK_K)

        # ----- V accumulation (latent) ---------------------------------
        for start_d in range(0, Dv_lat, BLOCK_DV):
            cur_d = start_d + tl.arange(0, BLOCK_DV, tl.int32)        # (BLOCK_DV,)
            d_mask = cur_d < Dv_lat

            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(V_ptr + offs_v,
                              mask=k_mask[:, None] & d_mask[None, :],
                              other=0.0,
                              cache_modifier='CA')                     # (BLOCK_K, BLOCK_DV) bf16
            v_fp32 = tl.cast(v_slice, tl.float32)                     # (BLOCK_K, BLOCK_DV) fp32

            # weighted = exp_score ⋅ V  (matrix‑multiply)
            weighted = tl.dot(exp_score, v_fp32)                       # (HEADS_PER_BLOCK, BLOCK_DV) fp32
            acc = acc + tl.where(d_mask[None, :], weighted, 0.0)      # (HEADS_PER_BLOCK, Dv_lat)

        # ----- advance running max --------------------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # Normalise latent accumulator (latent = acc / sum_exp)
    # ------------------------------------------------------------------
    latent = acc / sum_exp[:, None]               # (HEADS_PER_BLOCK, Dv_lat) fp32

    # ------------------------------------------------------------------
    # Project latent -> per‑head output (HEADS_PER_BLOCK, Dv)
    # ------------------------------------------------------------------
    v_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)   # (HEADS_PER_BLOCK, Dv)

    for start_d in range(0, Dv_lat, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV, tl.int32)               # (BLOCK_DV,)
        d_mask = cur_d < Dv_lat

        offs_wV = (
            (head_start + head_range)[:, None, None] * stride_wV_T_head
            + cur_d[None, :, None] * stride_wV_T_lat
            + tl.arange(0, Dv, tl.int32)[None, None, :] * stride_wV_T_out
        )
        wV_block = tl.load(wV_T_ptr + offs_wV,
                           mask=head_valid[:, None] & d_mask[None, :],
                           other=0.0)                                   # (HEADS_PER_BLOCK, BLOCK_DV, Dv) bf16
        wV_f32 = tl.cast(wV_block, tl.float32)                           # (HEADS_PER_BLOCK, BLOCK_DV, Dv)

        latent_slice = latent[:, start_d : start_d + BLOCK_DV]            # (HEADS_PER_BLOCK, BLOCK_DV)
        # matmul: (HEADS_PER_BLOCK, BLOCK_DV) × (BLOCK_DV, Dv)
        v_head += tl.dot(latent_slice, wV_f32)                           # (HEADS_PER_BLOCK, Dv)

    # ------------------------------------------------------------------
    # Store the per‑head values (cast back to bf16)
    # ------------------------------------------------------------------
    offs_vhead = (
        b * stride_vhead_batch
        + (head_start + head_range)[:, None] * stride_vhead_head
        + tl.arange(0, Dv, tl.int32)[None, :] * stride_vhead_out
    )
    tl.store(Vhead_ptr + offs_vhead,
             tl.cast(v_head, tl.bfloat16),
             mask=head_valid[:, None])

# ----------------------------------------------------------------------
# Fast‑path when qk_nope_head_dim == 0 (the common configuration)
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
    Fast forward when `qk_nope_head_dim == 0` – uses the custom Triton
    kernel for the attention + per‑head value projection.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim      # Dq
    dkv   = config.kv_lora_rank           # latent dim for V
    dv    = config.v_head_dim
    dim   = config.dim

    # --------------------------------------------------------------
    # 1️⃣  KV down‑projection + KV‑cache update (identical to reference)
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                     # (B, dim)
    kv_lora = F.linear(x2, wDKV)           # (B, dkv + drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_lat_new = kv_lora[:, :dkv]          # (B, dkv)
    rope_raw_new = kv_lora[:, dkv:]        # (B, drope)

    # rotate the newly generated key‑rope part (RoPE)
    cos_k = cos_tbl[cur_len]               # (drope,)
    sin_k = sin_tbl[cur_len]
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write into cache (latent + rotated rope)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_lat_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 2️⃣  Query up‑projection (fused weight) + RoPE
    # --------------------------------------------------------------
    global _cached_wq_fused
    if _cached_wq_fused is None or _cached_wq_fused.shape != (nh * drope, dim):
        # fuse the two linear layers once
        _cached_wq_fused = torch.matmul(wUQ, wDQ)          # (nh*drope, dim)

    q = F.linear(x2, _cached_wq_fused)                     # (B, nh*drope)
    q = q.view(bs, nh, drope)                             # (B, nh, drope)

    # apply RoPE at the query position (new_len‑1)
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                                # (drope,)
    sin_q = sin_tbl[q_pos]
    q = q * cos_q + _rotate_half(q) * sin_q               # (B, nh, drope)

    # --------------------------------------------------------------
    # 3️⃣  Gather K (already rotated) and V (latent) from the cache
    # --------------------------------------------------------------
    K = kv_cache.data[:, :new_len, dkv:]                   # (B, L, drope)
    V = kv_cache.data[:, :new_len, :dkv]                  # (B, L, dkv)

    # --------------------------------------------------------------
    # 4️⃣  Per‑head value‑projection matrix (prepare for kernel)
    # --------------------------------------------------------------
    global _cached_wV_T
    if _cached_wV_T is None:
        # wUKV: (nh*dv, dkv)  ->  (nh, dkv, dv)
        _cached_wV_T = wUKV.view(nh, dv, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 5️⃣  Run the Triton kernel (attention + latent → per‑head output)
    # --------------------------------------------------------------
    v_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # compute strides (all tensors are contiguous)
    stride_q_batch = q.stride(0)
    stride_q_head  = q.stride(1)
    stride_q_dim   = q.stride(2)

    stride_k_batch = K.stride(0)
    stride_k_len   = K.stride(1)
    stride_k_dim   = K.stride(2)

    stride_v_batch = V.stride(0)
    stride_v_len   = V.stride(1)
    stride_v_dim   = V.stride(2)

    stride_wV_T_head = _cached_wV_T.stride(0)
    stride_wV_T_lat  = _cached_wV_T.stride(1)
    stride_wV_T_out  = _cached_wV_T.stride(2)

    stride_vhead_batch = v_head.stride(0)
    stride_vhead_head  = v_head.stride(1)
    stride_vhead_out   = v_head.stride(2)

    # Grid: one program per (batch, head‑tile)
    # NOTE: we keep the classic 64‑head‑tile grid – safe for all autotune configs.
    grid = (bs * ((nh + 63) // 64),)

    _triton_attn_vhead_dot_kernel[grid](
        Q_ptr=q,
        K_ptr=K,
        V_ptr=V,
        wV_T_ptr=_cached_wV_T,
        Vhead_ptr=v_head,
        stride_q_batch=stride_q_batch,
        stride_q_head=stride_q_head,
        stride_q_dim=stride_q_dim,
        stride_k_batch=stride_k_batch,
        stride_k_len=stride_k_len,
        stride_k_dim=stride_k_dim,
        stride_v_batch=stride_v_batch,
        stride_v_len=stride_v_len,
        stride_v_dim=stride_v_dim,
        stride_wV_T_head=stride_wV_T_head,
        stride_wV_T_lat=stride_wV_T_lat,
        stride_wV_T_out=stride_wV_T_out,
        stride_vhead_batch=stride_vhead_batch,
        stride_vhead_head=stride_vhead_head,
        stride_vhead_out=stride_vhead_out,
        B=bs,
        H=nh,
        L=new_len,
        Dq=drope,
        Dv_lat=dkv,
        Dv=dv,
        scale=1.0 / math.sqrt(drope),
        # compile‑time consts injected by autotuner
        num_warps=8,
        num_stages=4,
    )

    # --------------------------------------------------------------
    # 6️⃣  Output projection (same as reference)
    # --------------------------------------------------------------
    v_head_flat = v_head.view(bs, nh * dv)                # (B, nh*dv)
    out = F.linear(v_head_flat, wO)                       # (B, dim)
    out = out.unsqueeze(1)                                # (B, 1, dim)

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
    wDKV = config.KV_proj_down_weight         # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # Build / fetch RoPE tables (cached globally)
    # --------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # --------------------------------------------------------------
    # Fast‑path – qk_nope_head_dim == 0 (the common configuration)
    # --------------------------------------------------------------
    if d_nope == 0:
        out, new_kv = _fast_forward_triton(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin,
        )
        # kv_cache is already updated inside the fast‑path function
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