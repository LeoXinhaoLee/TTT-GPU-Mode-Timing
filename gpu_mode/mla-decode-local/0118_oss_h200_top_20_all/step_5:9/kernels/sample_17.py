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
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


# ----------------------------------------------------------------------
# RoPE cache (cos / sin tables) – stored lazily
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
# Triton row‑wise softmax – unchanged (kept for the fallback path)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,              # number of columns
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
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
        NUM_STAGES=2,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------------
# Optimised MLA forward – custom_kernel
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
    sl   = config.seq_len             # always 1 in the benchmark
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
    wDKV  = config.KV_proj_down_weight           # (dkv+drope, dim)
    wUQ   = config.Q_proj_up_weight              # ((d_nope+drope)*nh, dq)
    wUKV  = config.KV_proj_up_weight             # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                     # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 0️⃣  Down‑projections (Linear layers without bias)
    # ------------------------------------------------------------------
    q_lora   = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora0 = F.linear(x, wDKV)                    # (bs, sl, dkv+drope)

    # ------------------------------------------------------------------
    # 1️⃣  KV‑cache update (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora0)            # kv_lora : (bs, kv_len, dkv+drope)
    query_pos = kv_len - 1                           # absolute position of the current query token

    # ------------------------------------------------------------------
    # Fast path when there is **no** NoPE part (d_nope == 0)
    # ------------------------------------------------------------------
    if d_nope == 0:
        # --------------------------------------------------------------
        # Up‑project queries (only RoPE part)
        # --------------------------------------------------------------
        # q_lora : (bs, 1, dq) → (bs, nh*drope)
        q_up = F.linear(q_lora.squeeze(1), wUQ)                 # (bs, nh*drope)
        q_up = q_up.view(bs, nh, drope)                         # (bs, nh, drope)

        # --------------------------------------------------------------
        # Split KV into latent (dkv) and rope (drope) parts
        # --------------------------------------------------------------
        kv_latent = kv_lora[..., :dkv]          # (bs, kv_len, dkv)
        kv_rope   = kv_lora[..., dkv:]          # (bs, kv_len, drope)

        # --------------------------------------------------------------
        # RoPE: pre‑compute cosine/sine tables (cached)
        # --------------------------------------------------------------
        cos_tbl, sin_tbl = _get_rope_tables(drope, msl, x.device)

        # --------------------------------------------------------------
        # Queries – apply RoPE at the current position
        # --------------------------------------------------------------
        cos_q = cos_tbl[query_pos].view(1, 1, drope)   # (1,1,drope)
        sin_q = sin_tbl[query_pos].view(1, 1, drope)
        q_rope = q_up * cos_q + _rotate_half(q_up) * sin_q   # (bs, nh, drope)

        # --------------------------------------------------------------
        # Keys – apply RoPE for every cached position
        # --------------------------------------------------------------
        cos_k = cos_tbl[:kv_len]    # (kv_len, drope)
        sin_k = sin_tbl[:kv_len]    # (kv_len, drope)
        # broadcast over batch dimension
        cos_k = cos_k.unsqueeze(0)   # (1, kv_len, drope)
        sin_k = sin_k.unsqueeze(0)
        k_rope = kv_rope * cos_k + _rotate_half(kv_rope) * sin_k   # (bs, kv_len, drope)

        # --------------------------------------------------------------
        # Flash‑Attention style fused Q‑K‑V computation
        # --------------------------------------------------------------
        # Shapes required by torch.nn.functional.scaled_dot_product_attention:
        #   Q : (B, 1, H, d_rope)
        #   K : (B, Kv, H, d_rope)
        #   V : (B, Kv, H, dkv)   (latent vectors)
        Q = q_rope.unsqueeze(1)                                   # (bs, 1, nh, drope)
        K = k_rope.unsqueeze(2).expand(-1, -1, nh, -1)            # (bs, kv_len, nh, drope)
        V = kv_latent.unsqueeze(2).expand(-1, -1, nh, -1)        # (bs, kv_len, nh, dkv)

        # scaled_dot_product_attention does 1/√d scaling internally
        latent_agg = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=0.0,
            is_causal=False
        )  # (bs, 1, nh, dkv)

        latent_agg = latent_agg.squeeze(1)                        # (bs, nh, dkv)

        # --------------------------------------------------------------
        # Project the aggregated latents into the value‑space
        # --------------------------------------------------------------
        # wUKV for this branch is (nh*dv, dkv)
        wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1)           # (nh, dkv, dv)
        y_head = torch.einsum('bhd, hdf -> bhf', latent_agg, wV_T)   # (bs, nh, dv)

        # --------------------------------------------------------------
        # Final output projection
        # --------------------------------------------------------------
        y_head_flat = y_head.reshape(bs, nh * dv)                 # (bs, nh*dv)
        output = F.linear(y_head_flat, wO)                        # (bs, dim)
        output = output.unsqueeze(1)                               # (bs, 1, dim)

    else:
        # ------------------------------------------------------------------
        # General path (fallback to the reference implementation – still correct)
        # ------------------------------------------------------------------
        # Up‑project queries (both NoPE + RoPE)
        q_up = F.linear(q_lora.squeeze(1), wUQ)                         # (bs, nh*(d_nope+drope))
        q_up = q_up.view(bs, nh, d_nope + drope)                       # (bs, nh, d_nope+drope)
        q_nope, q_rope_raw = torch.split(q_up, [d_nope, drope], dim=-1)

        # Split KV latent / rope parts
        kv_latent_raw = kv_lora[..., :dkv]                # (bs, kv_len, dkv)
        kv_rope_raw   = kv_lora[..., dkv:]                # (bs, kv_len, drope)

        # Split up‑projection weight into key‑ and value‑parts
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)      # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)       # (nh, dkv, dv)

        # ------------------------------------------------------------------
        # Project NoPE part of queries into latent space
        # ------------------------------------------------------------------
        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk', q_nope, wK)   # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((bs, nh, dkv),
                                         dtype=torch.bfloat16,
                                         device=x.device)

        # ------------------------------------------------------------------
        # RoPE tables
        # ------------------------------------------------------------------
        cos_tbl, sin_tbl = _get_rope_tables(drope, msl, x.device)

        # queries – RoPE
        cos_q = cos_tbl[query_pos].view(1, 1, drope)          # (1,1,drope)
        sin_q = sin_tbl[query_pos].view(1, 1, drope)
        q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, drope)

        # keys – RoPE (broadcast over heads)
        cos_k = cos_tbl[:kv_len].unsqueeze(0)                 # (1, kv_len, drope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)                 # (1, kv_len, drope)
        k_rope = kv_rope_raw * cos_k + _rotate_half(kv_rope_raw) * sin_k   # (bs, kv_len, drope)

        # ------------------------------------------------------------------
        # Compute scores (rope + nope) and apply scaling
        # ------------------------------------------------------------------
        scores_rope = torch.matmul(q_rope, k_rope.transpose(-1, -2))           # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent, kv_latent_raw.transpose(-1, -2))  # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + drope))

        # ------------------------------------------------------------------
        # Softmax – Triton implementation (row‑wise)
        # ------------------------------------------------------------------
        scores_flat = scores.reshape(bs * nh, kv_len)
        attn_flat = _triton_softmax(scores_flat)
        attn = attn_flat.view(bs, nh, kv_len)          # (bs, nh, kv_len)

        # ------------------------------------------------------------------
        # Weighted sum over latent vectors
        # ------------------------------------------------------------------
        latent_agg = torch.matmul(attn, kv_latent_raw)   # (bs, nh, dkv)

        # ------------------------------------------------------------------
        # Project to value space
        # ------------------------------------------------------------------
        y_head = torch.einsum('bhd, hdf -> bhf', latent_agg, wV_T)   # (bs, nh, dv)

        # ------------------------------------------------------------------
        # Final output projection
        # ------------------------------------------------------------------
        y_head_flat = y_head.reshape(bs, nh * dv)           # (bs, nh*dv)
        output = F.linear(y_head_flat, wO)                  # (bs, dim)
        output = output.unsqueeze(1)                        # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the (now‑updated) KV‑cache buffer
    # ------------------------------------------------------------------
    return output, kv_cache.data