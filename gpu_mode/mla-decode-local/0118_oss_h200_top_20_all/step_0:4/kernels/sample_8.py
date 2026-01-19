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
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Return cached cosine / sine tables for RoPE.
    Shape: (max_seq_len, dim) in bfloat16.
    """
    if not hasattr(_get_rope_tables, "_cache"):
        _get_rope_tables._cache = {}
    key = (dim, max_seq_len, device)
    if key not in _get_rope_tables._cache:
        half = dim // 2
        theta = (
            10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)
        ).to(torch.bfloat16)                     # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (L,1)
        idx = pos * theta[None, :]                # (L, half)
        idx = torch.cat([idx, idx], dim=-1)       # (L, dim)
        _get_rope_tables._cache[key] = (idx.cos().to(torch.bfloat16),
                                        idx.sin().to(torch.bfloat16))
    return _get_rope_tables._cache[key]

# ----------------------------------------------------------------------
# Triton row‑wise softmax (bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---- max (in fp32) ----
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---- exp & sum ----
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---- normalize ----
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
    # Choose a power‑of‑two block size
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
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# Optimised MLA forward (using batched matmul + Triton softmax)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Highly‑optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor          # (batch, seq_len, dim), dtype=bfloat16
    kv_cache.data : torch.Tensor  # updated raw KV‑cache tensor
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack config for readability (all values are python ints)
    # ------------------------------------------------------------------
    bs   = config.batch_size            # e.g. 128
    sl   = config.seq_len               # always 1 in the benchmark
    d    = config.dim                   # model dimension (7168)
    nh   = config.n_heads               # number of heads (128)
    dq   = config.q_lora_rank           # 1536
    dkv  = config.kv_lora_rank          # 512
    d_nope = config.qk_nope_head_dim    # e.g. 64
    d_rope = config.qk_rope_head_dim    # e.g. 64
    dv   = config.v_head_dim            # 128
    msl  = config.max_seq_len           # e.g. 8192

    # ------------------------------------------------------------------
    # Weights (already on the correct device, dtype = bfloat16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, d)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, d)
    wUQ   = config.Q_proj_up_weight            # ((d_nope + d_rope) * nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
    wO    = config.wo_weight                   # (d, nh * dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project (pure torch linear)
    # ------------------------------------------------------------------
    # x has shape (bs, 1, d) – squeeze the temporal dim for the linear ops
    x2d = x.squeeze(1)                           # (bs, d)

    q_lora   = F.linear(x2d, wDQ)                # (bs, dq)
    kv_lora_raw = F.linear(x2d, wDKV)            # (bs, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  Update KV‑cache (the cache holds the *raw* down‑projected values)
    # ------------------------------------------------------------------
    kv_lora_raw = kv_lora_raw.unsqueeze(1)       # (bs, 1, dkv + d_rope)
    kv_lora, kv_len = kv_cache(kv_lora_raw)      # kv_lora: (bs, kv_len, dkv + d_rope)
    query_pos = kv_len - 1                        # absolute position of the current token

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project query & split query into NOPE / RoPE parts
    # ------------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ).view(bs, nh, d_nope + d_rope)   # (bs, nh, d_nope+d_rope)
    q_nope = q_up[..., :d_nope]                                 # (bs, nh, d_nope)
    q_rope_raw = q_up[..., d_nope:]                             # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV‑cache into latent and RoPE streams
    # ------------------------------------------------------------------
    kv_nope_raw = kv_lora[..., :dkv]                # (bs, kv_len, dkv)
    k_rope_raw  = kv_lora[..., dkv:]                # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  Prepare RoPE tables (cached globally)
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ----- query side RoPE (single position) -----
    cos_q = cos_table[query_pos].unsqueeze(0).unsqueeze(0)   # (1,1,d_rope)
    sin_q = sin_table[query_pos].unsqueeze(0).unsqueeze(0)
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, d_rope)

    # ----- key side RoPE (all cached positions) -----
    cos_k = cos_table[:kv_len].unsqueeze(0)   # (1,kv_len,d_rope)
    sin_k = sin_table[:kv_len].unsqueeze(0)
    k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Extract per‑head projection matrices from wUKV
    # ------------------------------------------------------------------
    # wUKV shape: ((d_nope + dv) * nh, dkv)
    wUKV_ = wUKV.view(nh, d_nope + dv, dkv)   # (nh, d_nope+dv, dkv)
    wK = wUKV_[:, :d_nope, :]                 # (nh, d_nope, dkv)
    wV_T = wUKV_[:, d_nope:, :].permute(0, 2, 1)   # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # 7️⃣  Project the NOPE‑part of queries into the latent space (dkv)
    # ------------------------------------------------------------------
    # q_nope: (bs, nh, d_nope)   wK: (nh, d_nope, dkv)
    q_proj_dkv = torch.einsum('bhd,hdj->bhj', q_nope, wK)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 8️⃣  Compute attention scores (latent + RoPE) via batched matmul
    # ------------------------------------------------------------------
    # latent part
    scores_nope = torch.matmul(q_proj_dkv, kv_nope_raw.transpose(1, 2))   # (bs, nh, kv_len)
    # rope part
    scores_rope = torch.matmul(q_rope, k_rope.transpose(1, 2))           # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                         # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Softmax (row‑wise) → attention weights
    # ------------------------------------------------------------------
    attn = _triton_softmax(scores.view(bs * nh, kv_len)).view(bs, nh, kv_len)   # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 🔟  Weighted sum of latent KV vectors (produces aggregated dkv)
    # ------------------------------------------------------------------
    # Using batched matmul instead of einsum
    agg = torch.matmul(attn, kv_nope_raw)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣  Project aggregated latent vectors to values (per‑head linear)
    # ------------------------------------------------------------------
    v = torch.einsum('bhd,hdv->bhv', agg, wV_T)   # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣2️⃣  Merge heads & final output projection
    # ------------------------------------------------------------------
    y = v.reshape(bs, nh * dv)       # (bs, nh*dv)
    y = y.unsqueeze(1)               # (bs, 1, nh*dv)
    output = F.linear(y, wO)         # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data