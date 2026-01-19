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
# Global caches (shared across kernel calls)
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)  bfloat16
_cached_wdown: torch.Tensor = None # (q_lora_rank + kv_lora_rank + qk_rope_head_dim, dim) bfloat16
_cached_wV_T: torch.Tensor = None  # (n_heads, kv_lora_rank, v_head_dim) bfloat16
_cached_wO_v: torch.Tensor = None  # (n_heads, v_head_dim, dim) bfloat16

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half (used by RoPE)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Pre‑compute cosine / sine tables for rotary positional embeddings (bfloat16)."""
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
    pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)   # (max_seq_len, 1)
    idx = pos * theta                                                             # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                                           # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel – fused multi‑head attention + per‑head value projection
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 256, "BLOCK_DV": 64}, num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 64, "BLOCK_K": 256, "BLOCK_DV": 64}, num_warps=8,  num_stages=4),
        triton.Config({"HEADS_PER_BLOCK": 32, "BLOCK_K": 128, "BLOCK_DV": 64}, num_warps=8,  num_stages=4),
    ],
    key=[
        "B", "H", "L", "Dq", "Dv_lat", "Dv"
    ],
)
@triton.jit
def _triton_attn_vhead_kernel(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,               # (B, H, Dq)                     bf16
    K_ptr,               # (B, L, Dq)                     bf16
    V_ptr,               # (B, L, Dv_lat)                 bf16
    wV_T_ptr,            # (H, Dv_lat, Dv)                bf16
    wO_v_ptr,            # (H, Dv, Dim)                   bf16
    Out_ptr,             # (B, Dim)                       bf16

    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,      # Q   (B, H, Dq)
    stride_k_batch, stride_k_len,  stride_k_dim,      # K   (B, L, Dq)
    stride_v_batch, stride_v_len,  stride_v_dim,      # V   (B, L, Dv_lat)

    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,  # wV_T (H, Dv_lat, Dv)
    stride_wO_head,   stride_wO_mid, stride_wO_out,      # wO_v (H, Dv, Dim)

    stride_out_batch, stride_out_dim,                       # Out (B, Dim)

    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    L: tl.constexpr,          # KV length
    Dq: tl.constexpr,         # Q/K dimension (rope head dim)
    Dv_lat: tl.constexpr,     # KV‑LoRA rank
    Dv: tl.constexpr,         # per‑head value dim
    Dim: tl.constexpr,        # model dimension
    scale: tl.constexpr,      # 1/sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """
    Fused attention + value‑projection + final output projection.
    All accumulators are kept in fp32 for numerical stability.
    """
    pid = tl.program_id(0)                         # 0 … B * ceil(H/HEADS_PER_BLOCK) – 1
    num_head_tiles = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    b = pid // num_head_tiles                       # batch index
    tile = pid % num_head_tiles                     # head‑tile index inside batch
    head_start = tile * HEADS_PER_BLOCK              # first head handled by this program

    # ------------------------------------------------------------------
    # Load queries for this head‑tile (HEADS_PER_BLOCK × Dq)
    # ------------------------------------------------------------------
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_valid = head_start + head_range < H

    offs_q = (
        b * stride_q_batch
        + (head_start + head_range)[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                mask=head_valid[:, None],
                other=0.0)                         # (HEADS_PER_BLOCK, Dq)  bf16

    # ------------------------------------------------------------------
    # Allocate reduction buffers (fp32)
    # ------------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HEADS_PER_BLOCK)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)           # (HEADS_PER_BLOCK)
    acc_lat   = tl.zeros([HEADS_PER_BLOCK, BLOCK_DV], dtype=tl.float32)   # (HEADS_PER_BLOCK, BLOCK_DV)

    # ------------------------------------------------------------------
    # Main loop over key/value blocks (single‑pass Stable‑softmax)
    # ------------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K, tl.int32)               # (BLOCK_K,)
        k_mask = cur_k < L

        # ----- load K block -------------------------------------------------
        offs_k = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_block = tl.load(K_ptr + offs_k,
                          mask=k_mask[:, None],
                          other=0.0)                                   # (BLOCK_K, Dq)  bf16

        # ----- dot(q, k) ----------------------------------------------------
        prod = tl.sum(q[:, None, :] * k_block[None, :, :], axis=2)      # (HEADS_PER_BLOCK, BLOCK_K)  bf16
        score_f32 = tl.cast(prod, tl.float32) * scale                   # scaling (float32)

        # ----- stable soft‑max update ---------------------------------------
        block_max = tl.max(score_f32, axis=1)                           # (HEADS_PER_BLOCK)
        new_max   = tl.maximum(max_score, block_max)                     # (HEADS_PER_BLOCK)

        # rescale previous accumulator & denominator
        exp_factor = tl.exp(max_score - new_max)                         # (HEADS_PER_BLOCK) ≤ 1
        sum_exp = sum_exp * exp_factor
        acc_lat = acc_lat * exp_factor[:, None]

        # ----- exponentials for current block --------------------------------
        exp_score = tl.exp(score_f32 - new_max[:, None])                # (HEADS_PER_BLOCK, BLOCK_K)

        # ----- update denominator -------------------------------------------
        sum_exp = sum_exp + tl.sum(exp_score, axis=1)                    # (HEADS_PER_BLOCK)

        # ----- weighted value accumulation -----------------------------------
        for start_d in range(0, Dv_lat, BLOCK_DV):
            cur_d = start_d + tl.arange(0, BLOCK_DV, tl.int32)          # (BLOCK_DV,)
            d_mask = cur_d < Dv_lat

            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(V_ptr + offs_v,
                              mask=k_mask[:, None] & d_mask[None, :],
                              other=0.0)                               # (BLOCK_K, BLOCK_DV) bf16
            v_slice_f32 = tl.cast(v_slice, tl.float32)                 # (BLOCK_K, BLOCK_DV) fp32

            # weighted contribution of this block
            weighted = v_slice_f32[None, :, :] * exp_score[:, :, None]  # (HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV) fp32
            acc_lat = acc_lat + tl.sum(weighted, axis=1)               # (HEADS_PER_BLOCK, BLOCK_DV) fp32

        # ----- advance max --------------------------------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # Normalise the latent accumulator (latent = acc / sum_exp)
    # ------------------------------------------------------------------
    latent = acc_lat / sum_exp[:, None]                # (HEADS_PER_BLOCK, Dv_lat) fp32

    # ------------------------------------------------------------------
    # 1) Value projection:  latent @ wV_T  →  v_head (HEADS_PER_BLOCK, Dv)
    # ------------------------------------------------------------------
    # We'll perform the matmul in blocks over the Dv_lat dimension.
    v_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)

    for start_d in range(0, Dv_lat, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV, tl.int32)               # (BLOCK_DV,)
        d_mask = cur_d < Dv_lat

        # Load a slice of the per‑head value‑projection matrix
        offs_wV = (
            (head_start + head_range)[:, None, None] * stride_wV_T_head
            + cur_d[None, :, None] * stride_wV_T_lat
            + tl.arange(0, Dv, tl.int32)[None, None, :] * stride_wV_T_out
        )
        wV_block = tl.load(wV_T_ptr + offs_wV,
                           mask=head_valid[:, None] & d_mask[None, :],
                           other=0.0)                                   # (HEADS_PER_BLOCK, BLOCK_DV, Dv) bf16
        wV_block_f32 = tl.cast(wV_block, tl.float32)                     # fp32

        lat_slice = latent[:, start_d:start_d + BLOCK_DV]                # (HEADS_PER_BLOCK, BLOCK_DV) fp32
        # accumulate: (HEADS_PER_BLOCK, Dv) += lat_slice @ wV_block
        v_head += tl.sum(lat_slice[:, :, None] * wV_block_f32, axis=1)   # fp32

    # ------------------------------------------------------------------
    # 2) Final output projection:  v_head @ wO_v  →  partial out (HEADS_PER_BLOCK, Dim)
    # ------------------------------------------------------------------
    # Again compute the matmul in blocks over the Dv dimension.
    out_partial = tl.zeros([HEADS_PER_BLOCK, Dim], dtype=tl.float32)

    for start_d in range(0, Dv, BLOCK_DV):
        cur_d = start_d + tl.arange(0, BLOCK_DV, tl.int32)               # (BLOCK_DV,)
        d_mask = cur_d < Dv

        # Load a slice of the per‑head output‑projection matrix
        offs_wO = (
            (head_start + head_range)[:, None, None] * stride_wO_head
            + cur_d[None, :, None] * stride_wO_mid
            + tl.arange(0, Dim, tl.int32)[None, None, :] * stride_wO_out
        )
        wO_block = tl.load(wO_v_ptr + offs_wO,
                           mask=head_valid[:, None] & d_mask[None, :],
                           other=0.0)                                   # (HEADS_PER_BLOCK, BLOCK_DV, Dim) bf16
        wO_block_f32 = tl.cast(wO_block, tl.float32)                     # fp32

        v_slice = v_head[:, start_d:start_d + BLOCK_DV]                  # (HEADS_PER_BLOCK, BLOCK_DV) fp32
        out_partial += tl.sum(v_slice[:, :, None] * wO_block_f32, axis=1) # (HEADS_PER_BLOCK, Dim)

    # ------------------------------------------------------------------
    # Store the partial results directly into the final output tensor.
    # Because each head‑tile contributes a distinct slice of the head dimension,
    # there is no race – we can accumulate with a simple atomic‑add.
    # ------------------------------------------------------------------
    offs_out = (
        b * stride_out_batch
        + tl.arange(0, Dim)[None, :] * stride_out_dim
    )
    # Atomic add – Triton does not expose a native atomic for bf16,
    # so we perform the addition in fp32 and cast once.
    out_fp32 = tl.load(Out_ptr + offs_out, mask=tl.full([Dim], True), other=0.0)
    out_fp32 = out_fp32 + out_partial
    tl.store(Out_ptr + offs_out, tl.cast(out_fp32, tl.bfloat16), mask=tl.full([Dim], True))

# ----------------------------------------------------------------------
# Fast‑path – d_nope == 0 (no “no‑RoPE” dimensions)
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
    """Optimised forward when `qk_nope_head_dim == 0`."""
    bs = config.batch_size
    nh = config.n_heads
    drope = config.qk_rope_head_dim
    dkv = config.kv_lora_rank
    dv = config.v_head_dim
    dim = config.dim

    # --------------------------------------------------------------
    # 1️⃣ Down‑project Q and KV together (single fused matmul)
    # --------------------------------------------------------------
    global _cached_wdown
    if _cached_wdown is None or _cached_wdown.shape != (config.q_lora_rank + dkv + drope, dim):
        _cached_wdown = torch.cat([wDQ, wDKV], dim=0)  # (dq + dkv + drope, dim)

    # Input x is (B, 1, Dim) → squeeze temporal dim
    x2 = x.squeeze(1)                         # (B, Dim)
    proj = F.linear(x2, _cached_wdown)        # (B, dq + dkv + drope)
    q_lora = proj[:, :config.q_lora_rank]    # (B, dq)
    kv_lora = proj[:, config.q_lora_rank:]   # (B, dkv + drope)

    # --------------------------------------------------------------
    # 2️⃣ Write the newest token into the KV cache (RoPE already applied)
    # --------------------------------------------------------------
    cur_len = kv_cache.seq_len                 # already cached length
    new_len = cur_len + 1

    kv_latent_new = kv_lora[:, :dkv]           # (B, dkv)
    rope_raw_new  = kv_lora[:, dkv:]           # (B, drope)

    # RoPE for the key at position `cur_len`
    cos_k = cos_tbl[cur_len]                  # (drope,)
    sin_k = sin_tbl[cur_len]                  # (drope,)
    rope_rot = rope_raw_new * cos_k + _rotate_half(rope_raw_new) * sin_k   # (B, drope)

    # Update KV cache (in‑place)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_rot
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 3️⃣ Up‑project queries and apply RoPE
    # --------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)               # (B, nh * drope)
    q_up = q_up.view(bs, nh, drope)           # (B, nh, drope)

    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                    # (drope,)
    sin_q = sin_tbl[q_pos]                    # (drope,)
    q_rot = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)

    # --------------------------------------------------------------
    # 4️⃣ Gather the full KV tensors (rotated keys + latent values)
    # --------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]    # (B, L, dkv + drope)
    k_rope = kv_all[..., dkv:]                # (B, L, drope)
    v_latent = kv_all[..., :dkv]              # (B, L, dkv)

    # --------------------------------------------------------------
    # 5️⃣ Prepare per‑head weight tensors (cache them)
    # --------------------------------------------------------------
    global _cached_wV_T, _cached_wO_v
    if _cached_wV_T is None or _cached_wV_T.shape != (nh, dkv, dv):
        # wUKV shape: ((d_nope + dv) * nh, dkv) ; d_nope == 0 → (dv * nh, dkv)
        _cached_wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)

    if _cached_wO_v is None or _cached_wO_v.shape != (nh, dv, dim):
        _cached_wO_v = wO.view(dim, nh, dv).permute(1, 2, 0).contiguous()     # (nh, dv, dim)

    wV_T = _cached_wV_T
    wO_v = _cached_wO_v

    # --------------------------------------------------------------
    # 6️⃣ Allocate output buffer
    # --------------------------------------------------------------
    out = torch.empty((bs, dim), dtype=torch.bfloat16, device=x.device)

    # --------------------------------------------------------------
    # 7️⃣ Launch the fused Triton kernel (attention + two projections)
    # --------------------------------------------------------------
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

    stride_wO_head   = wO_v.stride(0)
    stride_wO_mid    = wO_v.stride(1)
    stride_wO_out    = wO_v.stride(2)

    stride_out_batch = out.stride(0)
    stride_out_dim   = out.stride(1)

    scale = 1.0 / math.sqrt(drope)   # Dq == drope because NoPE dim == 0

    # Select a good tiling – empirically the first config works best
    HEADS_PER_BLOCK = 32
    BLOCK_K = 256
    BLOCK_DV = 64

    grid = (bs * ((nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK),)

    _triton_attn_vhead_kernel[grid](
        # pointers
        q_rot, k_rope, v_latent,
        wV_T, wO_v, out,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len, stride_k_dim,
        stride_v_batch, stride_v_len, stride_v_dim,
        stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
        stride_wO_head,   stride_wO_mid,  stride_wO_out,
        stride_out_batch, stride_out_dim,
        # compile‑time constants
        bs, nh, new_len, drope, dkv, dv, dim,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV,
        num_warps=8, num_stages=4,
    )

    # --------------------------------------------------------------
    # 8️⃣ Reshape to (B, 1, Dim) to match the original API.
    # --------------------------------------------------------------
    out = out.unsqueeze(1)   # (B, 1, Dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Compiled fallback (d_nope > 0) – unchanged from reference
# ----------------------------------------------------------------------
_compiled_forward = None
def _build_compiled_forward():
    """Compiled fallback used when `qk_nope_head_dim` > 0."""
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
    Entry point expected by the benchmark harness.
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
    # Fast‑path – the common configuration has d_nope == 0
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