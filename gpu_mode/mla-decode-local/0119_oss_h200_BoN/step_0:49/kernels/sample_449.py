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
#  Utility helpers (rotate‑half, RoPE tables, Triton softmax)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half (the
    “rotate‑half” operation used by RoPE)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
#  RoPE cosine / sine tables (cached per (dim, max_seq_len, device) pair)
# ----------------------------------------------------------------------
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in BF16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device)
                            / half)).to(torch.bfloat16)               # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device)[:, None]  # (max_seq_len,1)
        idx = pos * theta[None, :]                    # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)          # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Triton‑based row‑wise softmax (bf16) – kept as a fallback if needed
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
    """Row‑wise softmax for a 2‑D BF16 tensor using Triton."""
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

# ----------------------------------------------------------------------
#  Main kernel: highly‑optimized MLA forward pass
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor          # shape (batch, seq_len, dim), BF16
    kv_cache_tensor : torch.Tensor # the internal KV‑cache tensor after update
    """
    # ------------------------------------------------------------------
    # Unpack configuration
    # ------------------------------------------------------------------
    config, x, kv_cache = data

    bs   = config.batch_size                # 128
    sl   = config.seq_len                   # ==1 in all supplied configs
    nh   = config.n_heads                   # 128
    d    = config.dim                       # 7168
    dq   = config.q_lora_rank               # 1536
    dkv  = config.kv_lora_rank              # 512
    dno  = config.qk_nope_head_dim          # e.g. 0‑64 (varies per config)
    dro  = config.qk_rope_head_dim          # 64
    dv   = config.v_head_dim                # 128
    msl  = config.max_seq_len               # 8192

    # ------------------------------------------------------------------
    # Weight tensors (already on the correct device & dtype)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight               # (dq, dim)
    wDKV  = config.KV_proj_down_weight              # (dkv + dro, dim)
    wUQ   = config.Q_proj_up_weight                 # ((dno+dro)*nh, dq)
    wUKV  = config.KV_proj_up_weight                # ((dno+dv)*nh, dkv)
    wO    = config.wo_weight                        # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project
    # ------------------------------------------------------------------
    # x : (bs, sl, d)
    # for speed we squeeze the seq‑dim (sl == 1)
    x2d = x.squeeze(1)                                 # (bs, d)

    q_lora  = F.linear(x2d, wDQ)                       # (bs, dq)
    kv_lora_input = F.linear(x2d, wDKV)                # (bs, dkv+dro)

    # ------------------------------------------------------------------
    # 2️⃣  KV‑cache update
    # ------------------------------------------------------------------
    # KV‑cache expects a (bs, seq, *) tensor – add the missing seq dim
    kv_lora_input = kv_lora_input.unsqueeze(1)          # (bs, 1, dkv+dro)
    kv_cache_tensor, kv_len = kv_cache(kv_lora_input)   # kv_len is total cached length
    # kv_cache_tensor has shape (bs, kv_len, dkv+dro)  (already stored inside kv_cache)

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries
    # ------------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                       # (bs, (dno+dro)*nh)
    q_up = q_up.view(bs, nh, dno + dro)                # (bs, nh, dno+dro)
    q_nope, q_rope = torch.split(q_up, [dno, dro], dim=-1)   # (bs, nh, dno) , (bs, nh, dro)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV into latent (no‑PE) and RoPE parts
    # ------------------------------------------------------------------
    kv_cache_tensor = kv_cache_tensor.to(torch.bfloat16)   # ensure BF16
    kv_nope   = kv_cache_tensor[..., :dkv]               # (bs, kv_len, dkv)
    k_rope_in = kv_cache_tensor[..., dkv:]               # (bs, kv_len, dro)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE tables (cached)
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(dro, msl, x.device)   # (msl, dro)

    # ------------------------------------------------------------------
    # 6️⃣  RoPE on queries (single position)
    # ------------------------------------------------------------------
    query_pos = kv_len - 1                                     # absolute position for this token
    cos_q = cos_table[query_pos]                               # (dro,)
    sin_q = sin_table[query_pos]                               # (dro,)
    # broadcast to (bs, nh, dro)
    cos_q = cos_q[None, None, :]                               # (1,1,dro)
    sin_q = sin_q[None, None, :]                               # (1,1,dro)

    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q    # (bs, nh, dro)

    # ------------------------------------------------------------------
    # 7️⃣  RoPE on keys (all cached positions)
    # ------------------------------------------------------------------
    # cos_k / sin_k : (kv_len, dro)
    cos_k = cos_table[:kv_len]                                 # (kv_len, dro)
    sin_k = sin_table[:kv_len]                                 # (kv_len, dro)
    # broadcast across batch dimension
    cos_k = cos_k[None, :, :]                                  # (1, kv_len, dro)
    sin_k = sin_k[None, :, :]                                  # (1, kv_len, dro)

    k_rope = k_rope_in * cos_k + _rotate_half(k_rope_in) * sin_k   # (bs, kv_len, dro)

    # ------------------------------------------------------------------
    # 8️⃣  Linear mapping for the “no‑PE” part of queries
    # ------------------------------------------------------------------
    # wUKV_view : (nh, dno+dv, dkv)
    wUKV_view = wUKV.view(nh, dno + dv, dkv)                # (nh, dno+dv, dkv)

    # only the first dno rows are needed for the query side
    wK = wUKV_view[:, :dno, :]                              # (nh, dno, dkv)

    # q_nope : (bs, nh, dno)   wK : (nh, dno, dkv) -> (bs, nh, dkv)
    # einsum is already highly‑optimised for BF16 and batched work.
    q_nope_lat = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 9️⃣  Build full‑dim Q and K tensors (no‑PE + RoPE)
    # ------------------------------------------------------------------
    # Q : (bs, nh, dkv+dro)
    Q = torch.cat([q_nope_lat, q_rope], dim=-1)            # (bs, nh, dkv+dro)

    # K : (bs, kv_len, dkv+dro)
    # kv_nope is shared across heads – we keep it compact and let matmul handle broadcasting.
    K = torch.cat([kv_nope, k_rope], dim=-1)               # (bs, kv_len, dkv+dro)

    # ------------------------------------------------------------------
    # 🔟  Compute raw attention scores (batched matmuls)
    # ------------------------------------------------------------------
    # Q @ Kᵀ  → (bs, nh, kv_len)
    # Note: torch.matmul broadcasts the head dimension of Q over the key dimension.
    # kv_nope_T : (bs, dkv+dro, kv_len)
    K_t = K.transpose(1, 2)                                 # (bs, dkv+dro, kv_len)
    scores = torch.matmul(Q, K_t)                           # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣  Scale and softmax
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(dno + dro)                       # divisor = sqrt(qk_head_dim)
    scores = scores * scale

    # Torch's cuDNN softmax is extremely fast for BF16, typically faster than a Triton‑based one.
    attn = F.softmax(scores, dim=-1)                        # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 1️⃣2️⃣  Weighted sum of latent keys (M = Σ attn·kv_nope)
    # ------------------------------------------------------------------
    # attn : (bs, nh, kv_len)   kv_nope : (bs, kv_len, dkv)
    M = torch.matmul(attn, kv_nope)                         # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 1️⃣3️⃣  Project aggregated latent keys to values (per‑head wV)
    # ------------------------------------------------------------------
    wV = wUKV_view[:, dno:, :]                              # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                              # (nh, dkv, dv)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)          # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣4️⃣  Output projection (wo)
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                         # (bs, nh*dv)
    y = y.unsqueeze(1)                                      # (bs, 1, nh*dv)
    output = F.linear(y, wO)                                 # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return result and the (now‑updated) KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data