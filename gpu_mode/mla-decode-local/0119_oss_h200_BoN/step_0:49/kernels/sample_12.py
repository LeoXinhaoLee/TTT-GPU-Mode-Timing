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

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# RoPE tables (cached per device/config)
# ----------------------------------------------------------------------
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta: (half,)
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len, 1)
        idx = pos * theta[None, :]          # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)  # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# Triton row‑wise softmax (bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
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
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ------------------------------------------------------------
# Optimised MLA forward kernel
# ------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_data : torch.Tensor   # the updated KV‑cache data tensor
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------
    # Unpack config for readability
    # ------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # always 1 in the provided configs
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------
    # Extract weight tensors (expected to be bf16 and on the correct device)
    # ------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight           # (dq, dim)
    wDKV  = config.KV_proj_down_weight          # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight             # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight            # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                    # (dim, nh*dv)

    # ------------------------------------------------------------
    # 1️⃣ Down‑projection
    # ------------------------------------------------------------
    # x: (bs, sl, dim)
    q_lora = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)             # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------
    # 2️⃣ KV‑cache update (in‑place)
    # ------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)    # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                         # absolute position of the *current* token

    # ------------------------------------------------------------
    # 3️⃣ Up‑project queries
    # ------------------------------------------------------------
    # squeeze seq_len (always 1) before the up‑projection
    q_up = F.linear(q_lora.squeeze(1), wUQ)       # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)    # (bs, nh, d_nope+d_rope)
    q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)   # each (bs, nh, *)

    # ------------------------------------------------------------
    # 4️⃣ RoPE for queries (single position)
    # ------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]      # (d_rope,)
    sin_q = sin_table[query_pos]      # (d_rope,)
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

    # ------------------------------------------------------------
    # 5️⃣ Split KV cache into latent & rope parts
    # ------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]          # (bs, kv_len, dkv)
    kv_rope_input = kv_lora[..., dkv:]          # (bs, kv_len, d_rope)

    # ------------------------------------------------------------
    # 6️⃣ RoPE for keys (all cached positions)
    # ------------------------------------------------------------
    cos_k = cos_table[:kv_len]      # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]      # (kv_len, d_rope)
    k_rope = kv_rope_input * cos_k + _rotate_half(kv_rope_input) * sin_k  # (bs, kv_len, d_rope)

    # ------------------------------------------------------------
    # 7️⃣ Prepare per‑head projection matrices from wUKV
    # ------------------------------------------------------------
    # wUKV shape: ((d_nope+dv)*nh, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)   # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                 # (nh, d_nope, dkv)
    wV = wUKV_view[:, d_nope:, :]                 # (nh, dv, dkv)

    # ------------------------------------------------------------
    # 8️⃣ Project query “no‑PE’’ part into latent space
    # ------------------------------------------------------------
    # q_nope: (bs, nh, d_nope), wK: (nh, d_nope, dkv) → (bs, nh, dkv)
    q_nope_latent = torch.einsum('bhd, hdk -> bhk', q_nope, wK)  # (bs, nh, dkv)

    # ------------------------------------------------------------
    # 9️⃣ Compute attention scores (latent + rotary)
    # ------------------------------------------------------------
    # latent part
    scores_nope = torch.matmul(q_nope_latent, kv_nope_input.transpose(-2, -1))   # (bs, nh, kv_len)
    # rope part
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))                  # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale   # (bs, nh, kv_len)

    # ------------------------------------------------------------
    # 🔟 Softmax (Triton implementation)
    # ------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)      # (B*H, kv_len)
    attn_flat = _triton_softmax(scores_flat)       # (B*H, kv_len) bf16
    attn = attn_flat.view(bs, nh, kv_len)         # (bs, nh, kv_len)

    # ------------------------------------------------------------
    # 1️⃣1️⃣ Weighted sum of latent keys
    # ------------------------------------------------------------
    M = torch.matmul(attn, kv_nope_input)         # (bs, nh, dkv)

    # ------------------------------------------------------------
    # 1️⃣2️⃣ Project aggregated latent representation to values
    # ------------------------------------------------------------
    wV_T = wV.permute(0, 2, 1)                    # (nh, dkv, dv)
    y_head = torch.einsum('bhd, hdk -> bhk', M, wV_T)   # (bs, nh, dv)

    # ------------------------------------------------------------
    # 1️⃣3️⃣ Final linear projection back to model dimension
    # ------------------------------------------------------------
    y = y_head.reshape(bs, -1)    # (bs, nh*dv)
    y = y.unsqueeze(1)            # (bs, 1, nh*dv)
    output = F.linear(y, wO)      # (bs, 1, dim) – bf16

    # ------------------------------------------------------------
    # Return output and updated KV‑cache data
    # ------------------------------------------------------------
    return output, kv_cache.data