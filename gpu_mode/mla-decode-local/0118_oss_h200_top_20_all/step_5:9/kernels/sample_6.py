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
# Helper – rotate‑half (used for RoPE)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ----------------------------------------------------------------------
# RoPE cache – cosine / sine tables (lazy, shared across calls)
# ----------------------------------------------------------------------
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Returns (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    `dim` must be even.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                     # (max_seq_len, 1)
        idx = pos * theta[None, :]                                          # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                 # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# Triton row‑wise softmax (unchanged from reference)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,              # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # -------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # -------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # -------- normalize ----------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)


def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape

    if n_cols <= 32:
        BLOCK_SIZE = 32
    elif n_cols <= 64:
        BLOCK_SIZE = 64
    elif n_cols <= 128:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    out = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_kernel[grid](
        out,
        x,
        out.stride(0),
        x.stride(0),
        N=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------------
# Main kernel – fast MLA forward (works for any `qk_nope_head_dim`)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward pass.
    Returns
    -------
    output: torch.Tensor       # shape (batch, seq_len, dim) – bf16
    kv_cache.data: torch.Tensor   # up‑to‑date cache (raw kv‑lora)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Local aliases (avoid repeated attribute look‑ups)
    # --------------------------------------------------------------
    bs   = config.batch_size          # e.g. 128
    sl   = config.seq_len             # always 1 in the benchmark / generation
    nh   = config.n_heads             # 128
    d    = config.dim                 # 7168
    dq   = config.q_lora_rank         # 1536
    dkv  = config.kv_lora_rank        # 512
    d_nope = config.qk_nope_head_dim  # may be 0
    drope = config.qk_rope_head_dim   # 64
    dv   = config.v_head_dim          # 128
    msl  = config.max_seq_len         # 8192

    # --------------------------------------------------------------
    # Extract weight tensors (already on CUDA, bfloat16)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight            # (dq, dim)
    wDKV  = config.KV_proj_down_weight           # (dkv + drope, dim)
    wUQ   = config.Q_proj_up_weight              # ((d_nope + drope) * nh, dq)
    wUKV  = config.KV_proj_up_weight             # ((d_nope + dv) * nh, dkv)
    wO    = config.wo_weight                     # (dim, nh * dv)

    # --------------------------------------------------------------
    # 0️⃣ Down‑projections (Linear layers without bias)
    # --------------------------------------------------------------
    # (bs, 1, dq)
    q_lora = F.linear(x, wDQ)
    # (bs, 1, dkv + drope)
    kv_lora0 = F.linear(x, wDKV)

    # --------------------------------------------------------------
    # 1️⃣ KV‑cache update (in‑place)
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora0)          # kv_lora : (bs, kv_len, dkv + drope)
    query_pos = kv_len - 1                         # absolute position of the current query token

    # --------------------------------------------------------------
    # 2️⃣ Up‑projection of queries (NoPE + RoPE)
    # --------------------------------------------------------------
    # (bs, (d_nope + drope) * nh) -> reshape -> (bs, nh, d_nope + drope)
    q_up = F.linear(q_lora.squeeze(1), wUQ)                 # (bs, (d_nope+drope)*nh)
    q_up = q_up.view(bs, nh, d_nope + drope)              # (bs, nh, d_nope+drope)

    if d_nope > 0:
        q_nope, q_rope = torch.split(q_up, [d_nope, drope], dim=-1)   # (bs, nh, d_nope) & (bs, nh, drope)
    else:
        q_nope = None
        q_rope = q_up                                            # (bs, nh, drope)

    # --------------------------------------------------------------
    # 3️⃣ Split KV‑lora into latent (dkv) and rope (drope) parts
    # --------------------------------------------------------------
    kv_latent_raw = kv_lora[..., :dkv]          # (bs, kv_len, dkv)
    kv_rope_raw   = kv_lora[..., dkv:]          # (bs, kv_len, drope)

    # --------------------------------------------------------------
    # 4️⃣ Prepare RoPE tables (cached)
    # --------------------------------------------------------------
    cos_tbl, sin_tbl = _get_rope_tables(drope, msl, x.device)

    # --------------------------------------------------------------
    # 5️⃣ RoPE – queries
    # --------------------------------------------------------------
    cos_q = cos_tbl[query_pos].view(1, 1, drope)   # (1,1,drope)
    sin_q = sin_tbl[query_pos].view(1, 1, drope)
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, drope)

    # --------------------------------------------------------------
    # 6️⃣ RoPE – keys (apply to every cached position)
    # --------------------------------------------------------------
    # (kv_len, drope)
    cos_k = cos_tbl[:kv_len]          # (kv_len, drope)
    sin_k = sin_tbl[:kv_len]          # (kv_len, drope)

    # broadcast over batch dimension
    cos_k = cos_k.unsqueeze(0)        # (1, kv_len, drope)
    sin_k = sin_k.unsqueeze(0)        # (1, kv_len, drope)

    k_rope = kv_rope_raw * cos_k + _rotate_half(kv_rope_raw) * sin_k   # (bs, kv_len, drope)

    # --------------------------------------------------------------
    # 7️⃣ Compute attention scores
    # --------------------------------------------------------------
    # Rope part
    # q_rope : (bs, nh, drope)
    # k_rope : (bs, kv_len, drope)
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-1, -2))   # (bs, nh, kv_len)

    if d_nope > 0:
        # ------------------------------------------------------
        # 7a️⃣ Project query NoPE part into the latent space (dkv)
        # ------------------------------------------------------
        # wUKV has shape ((d_nope+dv)*nh, dkv)
        # reshape to (nh, d_nope+dv, dkv)
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)

        # weight that maps NoPE part to latent space
        wK = wUKV_view[:, :d_nope, :]                              # (nh, d_nope, dkv)

        # q_nope : (bs, nh, d_nope)  ->  (bs, nh, dkv)
        q_nope_latent = torch.einsum('bhd, hdk -> bhk', q_nope, wK)  # (bs, nh, dkv)

        # latent part of the scores
        scores_nope = torch.matmul(q_nope_latent, kv_latent_raw.transpose(-1, -2))   # (bs, nh, kv_len)

        # combine
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + drope))
        wV = wUKV_view[:, d_nope:, :].permute(0, 2, 1)            # (nh, dkv, dv)
    else:
        # NoPE part absent → only rope scores
        scores = scores_rope * (1.0 / math.sqrt(drope))
        # For the dv‑projection we only need the value‑part of wUKV
        wV = wUKV.view(nh, dv, dkv).permute(0, 2, 1)              # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 8️⃣ Softmax (row‑wise) – Triton implementation
    # --------------------------------------------------------------
    # reshape to 2‑D for the Triton kernel: (B*H, KV_len)
    scores_flat = scores.reshape(bs * nh, kv_len)
    attn_flat = _triton_softmax(scores_flat)
    attn = attn_flat.view(bs, nh, kv_len)          # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 9️⃣ Weighted sum over latent vectors (dkv)
    # --------------------------------------------------------------
    # attn : (bs, nh, kv_len)   kv_latent_raw : (bs, kv_len, dkv)
    latent_agg = torch.matmul(attn, kv_latent_raw)   # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 🔟 Project aggregated latents to the value space (dv)
    # --------------------------------------------------------------
    # latent_agg : (bs, nh, dkv)   wV : (nh, dkv, dv)
    # use einsum which is nicely batched over heads
    y_head = torch.einsum('bhd, hdf -> bhf', latent_agg, wV)   # (bs, nh, dv)

    # --------------------------------------------------------------
    # 1️⃣1️⃣ Final linear projection back to model dimension
    # --------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)      # (bs, nh*dv)
    output = F.linear(y_head_flat, wO)             # (bs, dim)
    output = output.unsqueeze(1)                    # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the (now‑updated) KV‑cache buffer
    # ------------------------------------------------------------------
    return output, kv_cache.data