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
# Helper utilities (RoPE tables, rotate‑half, Triton softmax)
# ----------------------------------------------------------------------
_rotate_half_cache = {}
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


_rope_cache: dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )                               # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
        idx = pos * theta[None, :]                         # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Row‑wise softmax for a 2‑D bf16 tensor."""
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)


def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax using the Triton kernel above."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
    if n_cols <= 32:
        BLOCK = 32
    elif n_cols <= 64:
        BLOCK = 64
    elif n_cols <= 128:
        BLOCK = 128
    else:
        BLOCK = 1 << (n_cols - 1).bit_length()
        BLOCK = min(BLOCK, 1024)

    out = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_kernel[grid](
        out,
        x,
        out.stride(0),
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------------
# Custom kernel – MLA forward (fully‑fused version)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor                 # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor       # updated KV‑cache tensor (the .data field)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Unpack config
    # --------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len                # always 1 for the caller
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # --------------------------------------------------------------
    # Weight tensors (already on device & bf16)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # 1️⃣ Down‑project → x → q_lora , kv_lora_input
    # --------------------------------------------------------------
    # x : (bs, 1, d)
    q_lora     = F.linear(x, wDQ)                      # (bs, 1, dq)
    kv_lora_in = F.linear(x, wDKV)                     # (bs, 1, dkv + d_rope)

    # --------------------------------------------------------------
    # 2️⃣ KV‑cache update (in‑place)
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_in)   # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                    # absolute position for the query token

    # --------------------------------------------------------------
    # 3️⃣ Up‑project Q  (no‑PE + RoPE parts)
    # --------------------------------------------------------------
    # squeeze the singleton seq‑dim before the up‑projection
    q_up = F.linear(q_lora.squeeze(1), wUQ)                # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)               # (bs, nh, d_nope+d_rope)

    q_nope = q_up[..., :d_nope]                             # (bs, nh, d_nope)
    q_rope_raw = q_up[..., d_nope:]                         # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 4️⃣ Split KV into latent (no‑PE) and RoPE components
    # --------------------------------------------------------------
    kv_latent   = kv_lora[..., :dkv]                         # (bs, kv_len, dkv)
    k_rope_raw  = kv_lora[..., dkv:]                        # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 5️⃣ RoPE tables (cached) – apply to Q & K
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ---- Q RoPE (single position) ----
    cos_q = cos_table[query_pos]          # (d_rope,)
    sin_q = sin_table[query_pos]          # (d_rope,)
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, d_rope)

    # ---- K RoPE (all cached positions) ----
    # slice the tables to the current cache length
    cos_k = cos_table[:kv_len]            # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]            # (kv_len, d_rope)

    # broadcast across batch dimension
    k_rope = k_rope_raw * cos_k[None, :, :] + _rotate_half(k_rope_raw) * sin_k[None, :, :]   # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 6️⃣ Project Q “no‑PE” part into the latent space (dkv)
    # --------------------------------------------------------------
    # wUKV is ((d_nope+dv)*nh, dkv). View as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)          # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                         # (nh, d_nope, dkv)

    if d_nope > 0:
        q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)
    else:
        q_nope_latent = torch.zeros(bs, nh, dkv, dtype=torch.bfloat16, device=x.device)

    # --------------------------------------------------------------
    # 7️⃣ Compute raw attention scores (latent + RoPE)
    # --------------------------------------------------------------
    # latent part
    scores_nope = torch.einsum('bhd,bkd->bhk', q_nope_latent, kv_latent)   # (bs, nh, kv_len)
    # RoPE part
    scores_rope = torch.einsum('bhd,bkd->bhk', q_rope, k_rope)             # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale
    scores = scores.to(torch.bfloat16)

    # --------------------------------------------------------------
    # 8️⃣ Softmax (Triton) → attention weights
    # --------------------------------------------------------------
    scores_flat = scores.reshape(bs * nh, kv_len)            # (B*H, L)
    attn_flat = _triton_softmax(scores_flat)                # (B*H, L)
    attn = attn_flat.view(bs, nh, kv_len)                   # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 9️⃣ Weighted sum of latent keys → M (bs, nh, dkv)
    # --------------------------------------------------------------
    M = torch.einsum('bhk,bkd->bhd', attn, kv_latent)       # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 🔟 Project M to per‑head values (dv)
    # --------------------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                           # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                               # (nh, dkv, dv)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)          # (bs, nh, dv)

    # --------------------------------------------------------------
    # 1️⃣1️⃣ Merge heads & final linear projection
    # --------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                         # (bs, nh*dv)
    output = F.linear(y, wO)                                 # (bs, dim)
    output = output.unsqueeze(1)                             # (bs, 1, dim)

    # --------------------------------------------------------------
    # Return result and the (now‑updated) KV‑cache tensor
    # --------------------------------------------------------------
    return output, kv_cache.data