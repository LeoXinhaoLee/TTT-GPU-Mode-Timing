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
# 0️⃣  RoPE kernels (in‑place, half‑dim swap)
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                      # [B, T, D]  (bf16)
    cos_ptr, sin_ptr,           # [T, D] or [D] (bf16)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,           # D must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,  # processes D/2 elements per iteration
):
    pid = tl.program_id(0)
    bt = pid
    b = bt // T
    t = bt - b * T

    half = D // 2                     # size of each half
    offs = tl.arange(0, BLOCK_HALF)   # 0 … BLOCK_HALF‑1
    mask = offs < half

    # ------------------------------------------------------------------
    # pointers to the two halves of x
    # ------------------------------------------------------------------
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + offs * stride_xd                # first half
    x1_ptr = x_base + (half + offs) * stride_xd       # second half

    # ------------------------------------------------------------------
    # pointers to cos / sin (may be broadcasted)
    # ------------------------------------------------------------------
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t

    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d

    # ------------------------------------------------------------------
    # load (promote to fp32 for arithmetic)
    # ------------------------------------------------------------------
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # ------------------------------------------------------------------
    # RoPE with rotate‑half (swap‑halves)
    # out0 =  x0 * c - x1 * s
    # out1 =  x1 * c + x0 * s
    # ------------------------------------------------------------------
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # ------------------------------------------------------------------
    # store back in‑place
    # ------------------------------------------------------------------
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)


def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    Apply RoPE to a tensor of shape (B, H, D) where D is even.
    cos_q / sin_q are 1‑D tensors of length D (broadcasted over B·H).
    """
    assert q_rope.is_cuda
    assert q_rope.shape[-1] % 2 == 0
    B, H, D = q_rope.shape
    half = D // 2
    # nearest power‑of‑2 >= half
    BLOCK_HALF = 1 << (half - 1).bit_length()
    grid = (B * H,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=B,
        T=H,
        D=D,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        # broadcast cos/sin over the “T” dimension (H) → stride = 0
        stride_cos_t=0,
        stride_cos_d=cos_q.stride(0),
        stride_sin_t=0,
        stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )


def rope_inplace_keys(k_rope: torch.Tensor, cos_k: torch.Tensor, sin_k: torch.Tensor):
    """
    Apply RoPE to a tensor of shape (B, L, D) where D is even.
    cos_k / sin_k are (L, D) tables.
    """
    assert k_rope.is_cuda
    assert k_rope.shape[-1] % 2 == 0
    B, L, D = k_rope.shape
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    grid = (B * L,)

    rope_swap_halves_kernel[grid](
        k_rope,
        cos_k, sin_k,
        B=B,
        T=L,
        D=D,
        stride_xb=k_rope.stride(0),
        stride_xt=k_rope.stride(1),
        stride_xd=k_rope.stride(2),
        # cos/sin vary with the token index → non‑zero stride over T
        stride_cos_t=cos_k.stride(0),
        stride_cos_d=cos_k.stride(1),
        stride_sin_t=sin_k.stride(0),
        stride_sin_d=sin_k.stride(1),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )


# ----------------------------------------------------------------------
# 1️⃣  Cached RoPE tables (cos / sin) – build once per config
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bf16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )                     # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (L,1)
        idx = pos * theta[None, :]                 # (L, half)
        idx = torch.cat([idx, idx], dim=-1)        # (L, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# 2️⃣  Triton softmax (row‑wise, bf16 → fp32 for stability)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,                     # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
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
    # choose a power‑of‑2 block size (capped at 1024)
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
# 3️⃣  Optimised MLA forward kernel
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # the updated KV‑cache tensor
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration (readability)
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # always 1 in the provided configs
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
    wUQ   = config.Q_proj_up_weight             # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight            # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                    # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project
    # ------------------------------------------------------------------
    q_lora = F.linear(x, wDQ)                         # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)                 # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  Update KV‑cache (in‑place) and obtain absolute position of the query
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)         # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                             # absolute token index for the query

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries (seq_len == 1 → squeeze)
    # ------------------------------------------------------------------
    q_up = F.linear(q_lora.squeeze(1), wUQ)            # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)         # (bs, nh, d_nope+d_rope)
    q_nope = q_up[..., :d_nope]                       # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                       # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV into latent (no‑PE) and RoPE parts
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]                # (bs, kv_len, dkv)
    k_rope = kv_lora[..., dkv:].contiguous()          # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE tables (cos / sin) – cached per config
    # ------------------------------------------------------------------
    cos_tbl, sin_tbl = _get_rope_tables(d_rope, msl, x.device)

    # ---- query side (single position) ----
    cos_q = cos_tbl[query_pos]
    sin_q = sin_tbl[query_pos]
    rope_inplace_query(q_rope, cos_q, sin_q)          # in‑place

    # ---- key side (all cached positions) ----
    cos_k = cos_tbl[:kv_len]   # (kv_len, d_rope)
    sin_k = sin_tbl[:kv_len]   # (kv_len, d_rope)
    rope_inplace_keys(k_rope, cos_k, sin_k)           # in‑place

    # ------------------------------------------------------------------
    # 6️⃣  Latent projection for the “no‑PE” query part
    # ------------------------------------------------------------------
    # wUKV : ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)      # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                    # (nh, d_nope, dkv)

    # q_nope : (bs, nh, d_nope)   wK : (nh, d_nope, dkv) → (bs, nh, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)  # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 7️⃣  Compute attention scores (latent + RoPE)
    # ------------------------------------------------------------------
    # scores from latent part
    kv_nope_T = kv_nope_input.permute(0, 2, 1)                     # (bs, dkv, kv_len)
    scores_nope = torch.bmm(q_nope_latent, kv_nope_T)             # (bs, nh, kv_len)

    # scores from RoPE part
    k_rope_T = k_rope.permute(0, 2, 1)                             # (bs, d_rope, kv_len)
    scores_rope = torch.bmm(q_rope, k_rope_T)                     # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                  # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 8️⃣  Softmax (Triton) → attention weights
    # ------------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)                     # (B*H, kv_len)
    attn_flat = _triton_softmax(scores_flat)                       # (B*H, kv_len)
    attn = attn_flat.view(bs, nh, kv_len)                         # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Weighted sum of latent keys (M)
    # ------------------------------------------------------------------
    # M = Σ_t attn_{b,h,t} * kv_nope_input_{b,t,:}
    M = torch.einsum('bhl,bld->bhd', attn, kv_nope_input)        # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent keys to per‑head values
    # ------------------------------------------------------------------
    wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)            # (nh, dkv, dv)
    y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)               # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣ Merge heads & final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, -1).unsqueeze(1)                      # (bs, 1, nh*dv)
    output = F.linear(y, wO)                                      # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data