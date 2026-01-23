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
#  RoPE helper utilities (cached cosine/sine tables)
# ----------------------------------------------------------------------
_rope_cache = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached cosine / sine tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta = 10000 ** (-(i/half))
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                       # (max_seq_len, 1)
        idx = pos * theta[None, :]                     # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)            # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Triton kernel: RoPE (rotate‑half) - works for both query and key tensors
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                # [B, T, D] bf16
    cos_ptr, sin_ptr,     # [T, D] or [max_seq_len, D] bf16
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,      # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    # each program instance processes one (b, t) pair
    b = pid // T
    t = pid - b * T

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # base pointer for the row (b, t, :)
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + offs * stride_xd                     # first half
    x1_ptr = x_base + (half + offs) * stride_xd            # second half

    # cosine / sine for this position (t)
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t
    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d

    # load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE (rotate‑half)
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    In‑place RoPE on query tensor of shape (bs, nh, d_rope).
    cos_q / sin_q are 1‑D tensors of length d_rope.
    """
    assert q_rope.is_cuda and q_rope.dtype == torch.bfloat16
    bs, nh, d_rope = q_rope.shape
    half = d_rope // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    BLOCK_HALF = min(BLOCK_HALF, 256)

    grid = (bs * nh,)
    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=bs, T=nh, D=d_rope,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        # broadcast across t (heads) -> stride = 0
        stride_cos_t=0, stride_cos_d=cos_q.stride(0),
        stride_sin_t=0, stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

# ----------------------------------------------------------------------
#  Triton softmax (row‑wise, bf16)
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
#  Custom kernel – MLA forward (optimised)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns a tuple (output, updated_kv_cache_tensor).
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration (readability)
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # always 1 for the supplied configs
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    # Extract weight tensors (already on device & bf16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight           # (dq, dim)
    wDKV  = config.KV_proj_down_weight          # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight             # ((d_nope + d_rope) * nh, dq)
    wUKV  = config.KV_proj_up_weight            # ((d_nope + dv) * nh, dkv)
    wO    = config.wo_weight                    # (dim, nh * dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project
    # ------------------------------------------------------------------
    q_lora = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)             # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  Update KV‑cache (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)    # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                         # absolute position for query RoPE

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries (splitting No‑PE / RoPE parts)
    # ------------------------------------------------------------------
    # sl == 1 ⇒ squeeze before linear
    q_up = F.linear(q_lora.squeeze(1), wUQ)        # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)     # (bs, nh, d_total)
    q_nope = q_up[..., :d_nope]                   # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                   # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV into latent (no‑PE) and RoPE parts
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]            # (bs, kv_len, dkv)
    k_rope_input = kv_lora[..., dkv:]            # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE – use cached cosine/sine tables
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ----- query side (single position) -----
    cos_q = cos_table[query_pos]                 # (d_rope,)
    sin_q = sin_table[query_pos]                 # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)    # in‑place rotation

    # ----- key side (all cached positions) -----
    # we must not overwrite the KV‑cache; create a fresh copy for rotation
    k_rope = k_rope_input.clone()
    half = d_rope // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    BLOCK_HALF = min(BLOCK_HALF, 256)

    grid = (bs * kv_len,)
    rope_swap_halves_kernel[grid](
        k_rope,
        cos_table,
        sin_table,
        B=bs,
        T=kv_len,
        D=d_rope,
        stride_xb=k_rope.stride(0),
        stride_xt=k_rope.stride(1),
        stride_xd=k_rope.stride(2),
        stride_cos_t=cos_table.stride(0),
        stride_cos_d=cos_table.stride(1),
        stride_sin_t=sin_table.stride(0),
        stride_sin_d=sin_table.stride(1),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )
    # ------------------------------------------------------------------
    # 6️⃣  Split the up‑projection weight for KV
    # ------------------------------------------------------------------
    # wUKV shape: ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)   # (nh, total, dkv)
    wK = wUKV_view[:, :d_nope, :]                # (nh, d_nope, dkv)
    wV = wUKV_view[:, d_nope:, :]                # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                   # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # 7️⃣  Compute attention scores (no‑PE + RoPE)
    # ------------------------------------------------------------------
    # scores_nope:   (bs, nh, kv_len) = q_nope @ wK @ kv_nope_input^T
    # using two einsums for clarity / performance
    q_nope_lat = torch.einsum('bhi,hij->bhj', q_nope, wK)             # (bs, nh, dkv)
    scores_nope = torch.einsum('bhj,bkj->bhk', q_nope_lat, kv_nope_input)  # (bs, nh, kv_len)

    # scores_rope:   (bs, nh, kv_len) = q_rope @ k_rope^T
    scores_rope = torch.einsum('bhi,bki->bhk', q_rope, k_rope)       # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale

    # ------------------------------------------------------------------
    # 8️⃣  Softmax (Triton) → attention weights
    # ------------------------------------------------------------------
    # flatten heads for better utilisation of row‑wise softmax
    attn_flat = _triton_softmax(scores.view(bs * nh, kv_len))      # (B*H, kv_len)
    attn = attn_flat.view(bs, nh, kv_len)                          # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Weighted sum of latent keys (M)
    # ------------------------------------------------------------------
    M = torch.einsum('bhk,bkd->bhd', attn, kv_nope_input)          # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent keys to per‑head values
    # ------------------------------------------------------------------
    y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)                 # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣ Merge heads & final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)          # (bs, nh*dv)
    y = y.unsqueeze(1)                       # (bs, 1, nh*dv)
    output = F.linear(y, wO)                 # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data