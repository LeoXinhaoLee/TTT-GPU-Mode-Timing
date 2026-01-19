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

# -----------------------------------------------------------------------------
# Global caches (persist across calls)
# -----------------------------------------------------------------------------
_cached_cos: torch.Tensor = None          # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None          # (max_seq_len, rope_dim)  bfloat16
_cached_wq_fused: torch.Tensor = None     # (nh * rope_dim, dim)    bfloat16

# -----------------------------------------------------------------------------
# Helper functions (identical to the reference implementation)
# -----------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half (used by RoPE)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Pre‑compute cosine / sine tables for rotary positional embeddings (bfloat16)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
    pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)   # (max_seq_len, 1)
    idx = pos * theta                                          # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                        # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)

# ----------------------------------------------------------------------
# Triton kernel – fast latent attention using Tensor‑Core matmuls (dot)
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        # Base configuration – 64 heads per block, small K block
        triton.Config(
            {"HEADS_PER_BLOCK": 64, "BLOCK_K": 512, "BLOCK_DV": 128},
            num_warps=8,
            num_stages=4,
        ),
        # Wider K block
        triton.Config(
            {"HEADS_PER_BLOCK": 64, "BLOCK_K": 1024, "BLOCK_DV": 128},
            num_warps=8,
            num_stages=4,
        ),
        # Larger V‑tile – 256‑wide latent blocks
        triton.Config(
            {"HEADS_PER_BLOCK": 64, "BLOCK_K": 1024, "BLOCK_DV": 256},
            num_warps=8,
            num_stages=4,
        ),
        # Large K block + 256‑wide V‑tile
        triton.Config(
            {"HEADS_PER_BLOCK": 64, "BLOCK_K": 2048, "BLOCK_DV": 256},
            num_warps=8,
            num_stages=4,
        ),
        # *** New configuration: full latent dimension in a single V block ***
        triton.Config(
            {"HEADS_PER_BLOCK": 64, "BLOCK_K": 1024, "BLOCK_DV": 512},
            num_warps=8,
            num_stages=4,
        ),
        # Same as above but with a larger K block (useful for very long KV)
        triton.Config(
            {"HEADS_PER_BLOCK": 64, "BLOCK_K": 2048, "BLOCK_DV": 512},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=["B", "H", "L", "Dq", "Dv_lat"],
)
@triton.jit
def _triton_latent_attn_kernel_allcols(
    # --------------------------------------------------------------
    # Pointers
    # --------------------------------------------------------------
    Q_ptr,               # (B, H, Dq)                     bf16
    K_ptr,               # (B, L, Dq)                     bf16
    V_ptr,               # (B, L, Dv_lat)                 bf16
    Latent_ptr,          # (B, H, Dv_lat)                 bf16

    # --------------------------------------------------------------
    # Strides
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,      # Q   (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,      # K   (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,      # V   (B, L, Dv_lat)

    stride_lat_batch, stride_lat_head, stride_lat_dim, # Latent output (B, H, Dv_lat)

    # --------------------------------------------------------------
    # Compile‑time constants
    # --------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    L: tl.constexpr,          # KV length (current seq length)
    Dq: tl.constexpr,         # rope dimension (e.g. 64)
    Dv_lat: tl.constexpr,     # latent dimension (kv_lora_rank, e.g. 512)
    SCALE: tl.constexpr,      # 1/sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Stable‑softmax attention that accumulates the *latent* (value‑aggregated)
    representation for all column‑blocks in a single launch.
    """
    pid = tl.program_id(0)                         # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # head‑tile index inside batch
    head_start = tile * HEADS_PER_BLOCK              # first head handled by this program

    # --------------------------------------------------------------
    # Load query vectors for this head‑tile (fp32)
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
    q = tl.cast(q, tl.float32)                     # fp32 for precision

    # --------------------------------------------------------------
    # Allocate per‑column accumulators and soft‑max state (fp32)
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)

    # number of column blocks needed for the latent dimension
    COL_BLOCKS = (Dv_lat + BLOCK_DV - 1) // BLOCK_DV
    lat_acc = [tl.zeros([HEADS_PER_BLOCK, BLOCK_DV], dtype=tl.float32) for _ in range(COL_BLOCKS)]

    # --------------------------------------------------------------
    # Main loop over KV blocks (K & V)
    # --------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)               # (BLOCK_K,)
        k_mask = cur_k < L

        # ------------------------------------------------------------------
        # Load K block (B, BLOCK_K, Dq) – bf16 → fp32
        # ------------------------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k = tl.load(K_ptr + offs_k,
                     mask=k_mask[:, None],
                     other=0.0,
                     cache_modifier='CA')                     # (BLOCK_K, Dq) bf16
        k = tl.cast(k, tl.float32)                                 # fp32

        # ------------------------------------------------------------------
        # Compute scaled dot‑product scores (HEADS_PER_BLOCK, BLOCK_K)
        # ------------------------------------------------------------------
        scores = tl.dot(q, k, trans_b=True) * SCALE                 # fp32

        # ------------------------------------------------------------------
        # Stable‑softmax update (once per KV‑block)
        # ------------------------------------------------------------------
        block_max = tl.max(scores, axis=1)                          # (HEADS_PER_BLOCK)
        new_max   = tl.maximum(max_score, block_max)                # (HEADS_PER_BLOCK)

        exp_factor = tl.exp(max_score - new_max)                    # (HEADS_PER_BLOCK)
        sum_exp = sum_exp * exp_factor
        for i in range(COL_BLOCKS):
            lat_acc[i] = lat_acc[i] * exp_factor[:, None]

        exp_score = tl.exp(scores - new_max[:, None])               # (HEADS_PER_BLOCK, BLOCK_K)

        # ------------------------------------------------------------------
        # Load V slices for each column‑block and update its accumulator
        # ------------------------------------------------------------------
        for col_idx in range(COL_BLOCKS):
            cur_d = col_idx * BLOCK_DV + tl.arange(0, BLOCK_DV, tl.int32)  # (BLOCK_DV,)
            d_mask = cur_d < Dv_lat

            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v = tl.load(V_ptr + offs_v,
                        mask=k_mask[:, None] & d_mask[None, :],
                        other=0.0,
                        cache_modifier='CA')                     # (BLOCK_K, BLOCK_DV) bf16
            v = tl.cast(v, tl.float32)                             # fp32

            # Weighted sum → per‑column accumulator (Tensor‑Core dot)
            weighted = tl.dot(exp_score, v)                        # (HEADS_PER_BLOCK, BLOCK_DV)
            lat_acc[col_idx] = lat_acc[col_idx] + weighted

        # ------------------------------------------------------------------
        # Advance running max
        # ------------------------------------------------------------------
        max_score = new_max

    # ----------------------------------------------------------------------
    # Normalise and store each column‑block
    # ----------------------------------------------------------------------
    for col_idx in range(COL_BLOCKS):
        latent = lat_acc[col_idx] / sum_exp[:, None]               # (HEADS_PER_BLOCK, BLOCK_DV) fp32

        # Store to output (bf16)
        cur_d = col_idx * BLOCK_DV + tl.arange(0, BLOCK_DV, tl.int32)  # (BLOCK_DV,)
        d_mask = cur_d < Dv_lat

        offs_lat = (
            b * stride_lat_batch
            + (head_start + head_range)[:, None] * stride_lat_head
            + cur_d[None, :] * stride_lat_dim
        )
        tl.store(
            Latent_ptr + offs_lat,
            tl.cast(latent, tl.bfloat16),
            mask=head_valid[:, None] & d_mask[None, :],
        )

# ----------------------------------------------------------------------
# Fast‑path – qk_nope_head_dim == 0 (the common configuration)
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
    Fast forward when `qk_nope_head_dim == 0`. This implementation
    uses the optimized Triton kernel defined above.
    """
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim          # Dq
    dkv   = config.kv_lora_rank               # latent dimension
    dim   = config.dim

    # --------------------------------------------------------------
    # 1️⃣  KV down‑projection + KV‑cache update
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                                 # (B, dim)
    kv_lora = F.linear(x2, wDKV)                      # (B, dkv + drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_lat_new = kv_lora[:, :dkv]                     # (B, dkv)
    rope_raw_new = kv_lora[:, dkv:]                   # (B, drope)

    # rotate the newly generated key‑rope part (single position)
    cos_k = cos_tbl[cur_len]                           # (drope,)
    sin_k = sin_tbl[cur_len]
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k

    # write into cache (latent + rotated rope)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_lat_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 2️⃣  Query up‑projection + RoPE
    # --------------------------------------------------------------
    global _cached_wq_fused
    if _cached_wq_fused is None or _cached_wq_fused.shape != (nh * drope, dim):
        # fuse the two linear layers once (Q_up @ Q_down)
        _cached_wq_fused = torch.matmul(wUQ, wDQ)          # (nh*drope, dim)

    q = F.linear(x2, _cached_wq_fused)                     # (B, nh*drope)
    q = q.view(bs, nh, drope)                             # (B, nh, drope)

    # apply RoPE at the query position (new_len‑1)
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                                 # (drope,)
    sin_q = sin_tbl[q_pos]
    q = q * cos_q + _rotate_half(q) * sin_q               # (B, nh, drope)

    # --------------------------------------------------------------
    # 3️⃣  Gather K (rotated) and V (latent) from the cache
    # --------------------------------------------------------------
    K = kv_cache.data[:, :new_len, dkv:]                   # (B, L, drope)
    V = kv_cache.data[:, :new_len, :dkv]                  # (B, L, dkv)

    # --------------------------------------------------------------
    # 4️⃣  Compute latent attention with the optimized Triton kernel
    # --------------------------------------------------------------
    latent = torch.empty((bs, nh, dkv), dtype=torch.bfloat16, device=x.device)

    # Grid configuration – each program processes HEADS_PER_BLOCK heads.
    # All autotuned configs use HEADS_PER_BLOCK = 64.
    head_tiles = (nh + 63) // 64
    grid = (bs * head_tiles,)

    _triton_latent_attn_kernel_allcols[grid](
        Q_ptr=q,
        K_ptr=K,
        V_ptr=V,
        Latent_ptr=latent,
        stride_q_batch=q.stride(0),
        stride_q_head=q.stride(1),
        stride_q_dim=q.stride(2),
        stride_k_batch=K.stride(0),
        stride_k_len=K.stride(1),
        stride_k_dim=K.stride(2),
        stride_v_batch=V.stride(0),
        stride_v_len=V.stride(1),
        stride_v_dim=V.stride(2),
        stride_lat_batch=latent.stride(0),
        stride_lat_head=latent.stride(1),
        stride_lat_dim=latent.stride(2),
        B=bs,
        H=nh,
        L=new_len,
        Dq=drope,
        Dv_lat=dkv,
        SCALE=1.0 / math.sqrt(drope),   # qk_nope_head_dim == 0
    )

    # --------------------------------------------------------------
    # 5️⃣  Per‑head value projection (batched matmul via einsum)
    # --------------------------------------------------------------
    # wUKV shape: ((d_nope + dv) * nh, dkv) -> (nh*dv, dkv) because d_nope==0
    wV_T = wUKV.view(nh, config.v_head_dim, dkv).transpose(1, 2).contiguous()   # (nh, dkv, dv)

    v_head = torch.einsum('bhd,hdv->bhv', latent, wV_T)           # (B, nh, dv)

    # --------------------------------------------------------------
    # 6️⃣  Output projection (same as reference)
    # --------------------------------------------------------------
    v_head_flat = v_head.view(bs, nh * config.v_head_dim)                # (B, nh*dv)
    out = F.linear(v_head_flat, wO)                       # (B, dim)
    out = out.unsqueeze(1)                                 # (B, 1, dim)

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
    dv    = config.v_head_dim

    # --------------------------------------------------------------
    # Weight tensors (already on the correct device & dtype)
    # --------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

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