# --------------------------------------------------------------
# Triton‑based multi‑head latent attention (fast path)
# --------------------------------------------------------------
# NOTE:  This file must expose a single function called
#        `custom_kernel` with the signature described in the
#        prompt.  All other helpers are defined locally.
# --------------------------------------------------------------

### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from reference import KVCache, Config   # noqa: F401 – imported as required
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# Global (cached) RoPE tables – filled on‑first use
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)  bf16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)  bf16

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Exactly the same table as the reference implementation."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
        torch.bfloat16
    )
    pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
    idx = pos * theta  # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)  # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate‑half helper – identical to the reference version."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ----------------------------------------------------------------------
# Triton kernel – fuses the softmax, weighted‑sum and value‑projection.
#            – the key/value buffers are stored transposed (D, L) for
#              co‑alesced loads.
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 256}, num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 512}, num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 1024}, num_warps=8, num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 256}, num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 512}, num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 1024}, num_warps=8, num_stages=4),
        # a larger BLOCK_K often helps on H200 (large L, modest D)
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 2048}, num_warps=8, num_stages=4),
    ],
    key=["B", "H", "L", "Dq", "Dv_lat"],
)
@triton.jit
def _triton_mla_kernel(
    # ------------------------------------------------------------------
    # Pointers (all tensors are bf16)
    # ------------------------------------------------------------------
    Q_ptr,                 # (B, H, Dq)
    K_ptr,                 # (B, Dq, L)   – transposed KV keys
    V_ptr,                 # (B, Dv_lat, L)  – transposed KV values (latent)
    W_ptr,                 # (H, Dv_lat, Dv) – per‑head value‑projection
    Out_ptr,               # (B, H, Dv)
    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_dim, stride_k_len,
    stride_v_batch, stride_v_dim, stride_v_len,
    stride_w_head, stride_w_in, stride_w_out,
    stride_out_batch, stride_out_head, stride_out_dim,
    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    L: tl.constexpr,          # current KV length
    Dq: tl.constexpr,         # rope (query/key) head dimension
    Dv_lat: tl.constexpr,     # latent dimension (kv_lora_rank)
    Dv: tl.constexpr,         # value dimension (v_head_dim)
    SCALE: tl.constexpr,      # 1/sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    1️⃣  Compute scaled dot‑product Q·K   (tensor‑core friendly)
    2️⃣  Stable soft‑max (running max / sumexp)
    3️⃣  Weighted sum of the latent V
    4️⃣  Apply per‑head value projection W
    """
    pid = tl.program_id(0)                         # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    # ------------------------------------------------------------------
    # Tile identification (batch + head‑tile)
    # ------------------------------------------------------------------
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # head‑tile inside the batch
    head_start = tile * HEADS_PER_BLOCK

    # ------------------------------------------------------------------
    # Load Q for the whole head‑tile (HEADS_PER_BLOCK × Dq)
    # ------------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_valid = head_start + head_range < H          # mask for the last tile

    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q, mask=head_valid[:, None], other=0.0)   # (HEADS_PER_BLOCK, Dq)

    # ------------------------------------------------------------------
    # Buffers for numerically stable soft‑max
    # ------------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)

    # accumulator for the latent vector (size Dv_lat)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dv_lat], dtype=tl.float32)

    # ------------------------------------------------------------------
    # Main loop over KV blocks (K & V)
    # ------------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)          # (BLOCK_K,)
        k_mask = cur_k < L

        # --------------------------------------------------------------
        # Load K block – shape (Dq, BLOCK_K) – contiguous thanks to the
        # transposed layout (B, Dq, L).  This is a co‑alesced load.
        # --------------------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + tl.arange(0, Dq)[:, None] * stride_k_dim          # dim axis
            + cur_k[None, :] * stride_k_len                     # token axis (stride = 1)
        )
        k_t = tl.load(K_ptr + offs_k,
                       mask=k_mask[None, :],
                       other=0.0,
                       cache_modifier='CA')                    # (Dq, BLOCK_K)  bf16

        # --------------------------------------------------------------
        # Scaled dot‑product scores   (HEADS_PER_BLOCK, BLOCK_K)
        # --------------------------------------------------------------
        prod = tl.dot(q, k_t)                     # bf16 → (HEADS_PER_BLOCK, BLOCK_K)
        score_f32 = tl.cast(prod, tl.float32) * SCALE

        # --------------------------------------------------------------
        # Stable‑softmax update
        # --------------------------------------------------------------
        block_max = tl.max(score_f32, axis=1)                # (HEADS_PER_BLOCK)
        new_max   = tl.maximum(max_score, block_max)          # (HEADS_PER_BLOCK)

        # renormalise previously accumulated terms
        exp_factor = tl.exp(max_score - new_max)              # (HEADS_PER_BLOCK)
        sum_exp = sum_exp * exp_factor
        latent_acc = latent_acc * exp_factor[:, None]

        exp_score = tl.exp(score_f32 - new_max[:, None])    # (HEADS_PER_BLOCK, BLOCK_K)

        # --------------------------------------------------------------
        # Load V slice – shape (BLOCK_K, Dv_lat), also contiguous thanks to
        # the transposed layout (B, Dv_lat, L).
        # --------------------------------------------------------------
        offs_v = (
            b * stride_v_batch
            + cur_k[:, None] * stride_v_len                     # token axis (stride = 1)
            + tl.arange(0, Dv_lat)[None, :] * stride_v_dim      # latent dim axis
        )
        v_slice = tl.load(V_ptr + offs_v,
                          mask=k_mask[:, None],
                          other=0.0,
                          cache_modifier='CA')                 # (BLOCK_K, Dv_lat) bf16
        v_fp32 = tl.cast(v_slice, tl.float32)                # (BLOCK_K, Dv_lat) fp32

        # --------------------------------------------------------------
        # Weighted sum → latent accumulator
        # --------------------------------------------------------------
        latent_acc = latent_acc + tl.dot(exp_score, v_fp32)   # (HEADS_PER_BLOCK, Dv_lat)

        # --------------------------------------------------------------
        # Update running max (for the next block)
        # --------------------------------------------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # Normalise the latent vector (soft‑max denominator)
    # ------------------------------------------------------------------
    latent = latent_acc / sum_exp[:, None]               # (HEADS_PER_BLOCK, Dv_lat) fp32

    # ------------------------------------------------------------------
    # Multiply with per‑head value‑projection matrix W
    # ------------------------------------------------------------------
    BLOCK_DV_OUT = 32
    for d_start in tl.range(0, Dv, BLOCK_DV_OUT):
        cur_d = d_start + tl.arange(0, BLOCK_DV_OUT, tl.int32)           # (BLOCK_DV_OUT,)
        d_mask = cur_d < Dv

        offs_w = (
            (head_start + head_range)[:, None] * stride_w_head
            + tl.arange(0, Dv_lat)[:, None] * stride_w_in
            + cur_d[None, :] * stride_w_out
        )
        w_block = tl.load(W_ptr + offs_w,
                          mask=head_valid[:, None] & d_mask[None, :],
                          other=0.0)                                   # (HEADS_PER_BLOCK, Dv_lat, BLOCK_DV_OUT)

        # (HEADS_PER_BLOCK, BLOCK_DV_OUT) ← (HEADS_PER_BLOCK, Dv_lat) @ (HEADS_PER_BLOCK, Dv_lat, BLOCK_DV_OUT)
        out_chunk = tl.sum(latent[:, :, None] * w_block, axis=1)         # (HEADS_PER_BLOCK, BLOCK_DV_OUT)

        offs_out = (
            b * stride_out_batch
            + (head_start + head_range)[:, None] * stride_out_head
            + cur_d[None, :] * stride_out_dim
        )
        tl.store(Out_ptr + offs_out,
                 tl.cast(out_chunk, tl.bfloat16),
                 mask=head_valid[:, None] & d_mask[None, :])


# ----------------------------------------------------------------------
# Fast‑path for the common configuration (qk_nope_head_dim == 0)
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
):
    """
    Optimised forward for the typical config where the “no‑PE” part
    is disabled.  This version stores the KV cache in a *transposed*
    layout (dim, seq_len) to obtain co‑alesced loads inside the Triton
    kernel.  The kernel fuses the soft‑max, latent accumulation and
    per‑head value projection; the final linear projection is performed
    with a regular torch GEMM.
    """
    # ------------------------------------------------------------------
    # 0️⃣  Allocate transposed cache buffers on first use
    # ------------------------------------------------------------------
    if not hasattr(kv_cache, "trans_k"):
        # (batch, rope_dim, max_seq_len)
        kv_cache.trans_k = torch.empty(
            (config.batch_size, config.qk_rope_head_dim, config.max_seq_len),
            dtype=torch.bfloat16,
            device=x.device,
        )
        # (batch, kv_lora_rank, max_seq_len)
        kv_cache.trans_v = torch.empty(
            (config.batch_size, config.kv_lora_rank, config.max_seq_len),
            dtype=torch.bfloat16,
            device=x.device,
        )
        # zero‑initialise – easier than calling .zero_() later
        kv_cache.trans_k.zero_()
        kv_cache.trans_v.zero_()

    bs = config.batch_size
    dkv = config.kv_lora_rank
    drope = config.qk_rope_head_dim

    # --------------------------------------------------------------
    # 1️⃣  Down‑project + KV‑cache update
    # --------------------------------------------------------------
    x2 = x.squeeze(1)                    # (B, dim)
    q_lora = F.linear(x2, wDQ)           # (B, q_lora_rank)
    kv_lora = F.linear(x2, wDKV)         # (B, dkv + drope)

    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    # split latent / rope part
    kv_lat_new = kv_lora[:, :dkv]                 # (B, dkv)
    rope_raw_new = kv_lora[:, dkv:]               # (B, drope)

    # rotate key‑rope at position = cur_len
    cos_k = cos_tbl[cur_len]                      # (drope,)
    sin_k = sin_tbl[cur_len]                      # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k   # (B, drope)

    # --------------------------------------------------------------
    # Write into the original cache (required by the fallback path)
    # --------------------------------------------------------------
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_lat_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot

    # --------------------------------------------------------------
    # Write into the transposed buffers (fast‑path only)
    # --------------------------------------------------------------
    kv_cache.trans_v[..., cur_len] = kv_lat_new          # (B, dkv)
    kv_cache.trans_k[..., cur_len] = rope_rot           # (B, drope)

    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 2️⃣  Query up‑projection + RoPE (still in torch, negligible cost)
    # --------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)               # (B, nh * drope)
    q = q_up.view(bs, config.n_heads, drope)   # (B, nh, drope)

    # rotate queries (position = new_len‑1)
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                     # (drope,)
    sin_q = sin_tbl[q_pos]                     # (drope,)
    q = q * cos_q + _rotate_half(q) * sin_q    # (B, nh, drope)

    # --------------------------------------------------------------
    # 3️⃣  Gather transposed K / V for the attention kernel
    # --------------------------------------------------------------
    K = kv_cache.trans_k[:, :, :new_len]       # (B, drope, L)
    V = kv_cache.trans_v[:, :, :new_len]       # (B, dkv, L)

    # --------------------------------------------------------------
    # 4️⃣  Run the fused Triton kernel
    # --------------------------------------------------------------
    out_head = torch.empty(
        (bs, config.n_heads, config.v_head_dim), dtype=torch.bfloat16, device=x.device
    )

    # reshape wUKV to (nh, dkv, dv) – per‑head value projection matrix
    wV_T = wUKV.view(config.n_heads, config.v_head_dim, dkv).transpose(1, 2).contiguous()  # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # Strides – all tensors are contiguous → we can just read .stride()
    # ------------------------------------------------------------------
    stride_q_batch = q.stride(0)
    stride_q_head  = q.stride(1)
    stride_q_dim   = q.stride(2)

    stride_k_batch = K.stride(0)
    stride_k_dim   = K.stride(1)    # stride over rope dim (Dq)
    stride_k_len   = K.stride(2)    # stride over token dimension (contiguous)

    stride_v_batch = V.stride(0)
    stride_v_dim   = V.stride(1)    # stride over latent dim
    stride_v_len   = V.stride(2)    # stride over token dimension (contiguous)

    stride_w_head = wV_T.stride(0)
    stride_w_in   = wV_T.stride(1)
    stride_w_out  = wV_T.stride(2)

    stride_out_batch = out_head.stride(0)
    stride_out_head  = out_head.stride(1)
    stride_out_dim   = out_head.stride(2)

    # ------------------------------------------------------------------
    # Grid launch – one program per (batch, head‑tile)
    # ------------------------------------------------------------------
    grid = (bs * ((config.n_heads + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

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
        stride_k_dim=stride_k_dim,
        stride_k_len=stride_k_len,
        stride_v_batch=stride_v_batch,
        stride_v_dim=stride_v_dim,
        stride_v_len=stride_v_len,
        stride_w_head=stride_w_head,
        stride_w_in=stride_w_in,
        stride_w_out=stride_w_out,
        stride_out_batch=stride_out_batch,
        stride_out_head=stride_out_head,
        stride_out_dim=stride_out_dim,
        B=bs,
        H=config.n_heads,
        L=new_len,
        Dq=drope,
        Dv_lat=dkv,
        Dv=config.v_head_dim,
        SCALE=1.0 / math.sqrt(drope),
        HEADS_PER_BLOCK=HEADS_PER_BLOCK,
        BLOCK_K=BLOCK_K,
    )

    # ------------------------------------------------------------------
    # 5️⃣  Final linear projection back to model dimension
    # ------------------------------------------------------------------
    out_head_flat = out_head.view(bs, -1)          # (B, nh*dv)
    out = F.linear(out_head_flat, wO)              # (B, dim)
    out = out.unsqueeze(1)                         # (B, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Fallback (general) – unchanged, compiled with torch.compile
# ----------------------------------------------------------------------
_compiled_forward: torch.nn.Module = None
def _build_compiled_forward():
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
        # ------------------------------------------------------------------
        # Reference implementation – unchanged (see original description)
        # ------------------------------------------------------------------
        q_lora = F.linear(x, wDQ)                                 # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)                              # (bs, 1, dkv + d_rope)

        new_len = cur_len + kv_lora0.shape[1]
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)

        kv_lora = kv_data[:, :new_len, :]                         # (bs, kv_len, dkv + d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        q_up = F.linear(q_lora.squeeze(1), wUQ)                   # (bs, nh*d_nope+d_rope)
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)        # (bs, nh, d_nope+d_rope)
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

    # ------------------------------------------------------------------
    # Extract scalar config values (plain python ints)
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
    # Weight tensors (already on the correct device & dtype)
    # ------------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

    # ------------------------------------------------------------------
    # Build / fetch RoPE tables (cached globally)
    # ------------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # ------------------------------------------------------------------
    # Fast‑path – qk_nope_head_dim == 0 (the common configuration)
    # ------------------------------------------------------------------
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
        return out, new_kv

    # ------------------------------------------------------------------
    # General case – fallback to the compiled reference implementation
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