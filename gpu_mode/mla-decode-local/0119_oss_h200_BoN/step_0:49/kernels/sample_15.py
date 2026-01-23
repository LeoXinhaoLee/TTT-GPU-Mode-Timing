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

# ---------------------------------------------------------------
#  Helper utilities (RoPE tables, rotate‑half, Triton softmax)
# ---------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bf16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                     # (max_seq_len, 1)
        idx = pos * theta[None, :]                                                                       # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                                              # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_out = row * stride_out
    row_off_in = row * stride_in

    # ---------- max ----------
    col = tl.arange(0, BLOCK_SIZE)
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float("inf"))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float("inf"))
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

# ---------------------------------------------------------------
#  Custom kernel – MLA forward
# ---------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of Multi‑head Latent Attention (MLA).

    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # the internally updated KV‑cache tensor
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len          # is 1 for the inference path
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    max_seq_len = config.max_seq_len

    # ------------------------------------------------------------------
    # Weights (all bf16, already on the correct device)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, d)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, d)
    wUQ   = config.Q_proj_up_weight            # ((d_nope + d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope + dv)*nh, dkv)
    wO    = config.wo_weight                   # (d, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project
    # ------------------------------------------------------------------
    # x : (bs, sl, d)  with sl == 1
    q_lora = F.linear(x, wDQ)                      # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)              # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  KV‑cache update
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)      # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                          # position of the just‑added token

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries
    # ------------------------------------------------------------------
    q_lora = q_lora.squeeze(1)                     # (bs, dq)
    q_up = F.linear(q_lora, wUQ)                   # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)      # (bs, nh, d_total)

    q_nope = q_up[..., :d_nope]                    # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                    # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV cache tensor
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]             # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]             # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE for query and keys
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, max_seq_len, x.device)

    # query side (single position)
    cos_q = cos_table[query_pos].view(1, 1, d_rope)    # (1,1,d_rope)
    sin_q = sin_table[query_pos].view(1, 1, d_rope)    # (1,1,d_rope)
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

    # key side (all cached positions)
    cos_k = cos_table[:kv_len].unsqueeze(0)            # (1, kv_len, d_rope)
    sin_k = sin_table[:kv_len].unsqueeze(0)            # (1, kv_len, d_rope)
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Project the “no‑PE” query part into the latent space (dkv)
    # ------------------------------------------------------------------
    # reshape KV‑up weight for easy slicing
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)       # (nh, d_total_head, dkv)
    wK = wUKV_view[:, :d_nope, :]                     # (nh, d_nope, dkv)

    # q_nope : (bs, nh, d_nope)  -> (bs, nh, dkv)
    q_nope_proj = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 7️⃣  Compute attention scores (latent + RoPE)
    # ------------------------------------------------------------------
    # latent part
    scores_nope = torch.einsum('bhd,bkd->bhk', q_nope_proj, kv_nope_input)   # (bs, nh, kv_len)

    # RoPE part
    scores_rope = torch.einsum('bhd,bkd->bhk', q_rope, k_rope)               # (bs, nh, kv_len)

    # combine + scaling
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                           # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 8️⃣  Softmax (Triton implementation)
    # ------------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)        # (B*H, kv_len)
    attn_flat = _triton_softmax(scores_flat)          # (B*H, kv_len)
    attn = attn_flat.view(bs, nh, kv_len)             # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Weighted sum of latent keys (M)
    # ------------------------------------------------------------------
    M = torch.einsum('bhn,bnk->bhk', attn, kv_nope_input)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent keys to per‑head values
    # ------------------------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                    # (nh, dkv, dv)
    v_head = torch.einsum('bhd,hdk->bhk', M, wV_T)   # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣ Merge heads & final projection
    # ------------------------------------------------------------------
    y = v_head.reshape(bs, nh * dv)               # (bs, nh*dv)
    y = y.unsqueeze(1)                            # (bs, 1, nh*dv)
    output = F.linear(y, wO)                      # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return output and the updated KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data