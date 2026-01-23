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
# 0️⃣  RoPE utilities (in‑place rotate‑half + triton kernel)
# ----------------------------------------------------------------------
@triton.jit
def _rope_swap_halves_kernel(
    x_ptr,                     # [B, H, D]  (bf16)
    cos_ptr, sin_ptr,          # [D]        (bf16) – broadcasted across B,H
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,           # D is even
    stride_xb, stride_xh, stride_xd,
    stride_c, stride_s,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    half = D // 2

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # ------------------------------------------------------------------
    # pointers for the two halves of the vector
    # ------------------------------------------------------------------
    x_base   = x_ptr + b * stride_xb + h * stride_xh
    x0_ptr = x_base + offs                * stride_xd           # first half
    x1_ptr = x_base + (half + offs)       * stride_xd           # second half

    # ------------------------------------------------------------------
    # pointers for cosine / sine (same for all B,H – stride set to 0)
    # ------------------------------------------------------------------
    c_ptr = cos_ptr + offs * stride_c
    s_ptr = sin_ptr + offs * stride_s

    # ------------------------------------------------------------------
    # loads
    # ------------------------------------------------------------------
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # ------------------------------------------------------------------
    # RoPE with rotate‑half (swap‑halves) in‑place
    #   out0 =  x0 * c - x1 * s
    #   out1 =  x1 * c + x0 * s
    # ------------------------------------------------------------------
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_inplace_query(q_rope: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """
    In‑place RoPE for the query tensor (shape: B × H × D, D even, dtype = bf16).
    ``cos`` / ``sin`` are 1‑D tensors of length D (same dtype).
    """
    assert q_rope.is_cuda
    B, H, D = q_rope.shape
    assert D % 2 == 0

    # choose a block size that is a power‑of‑2 and >= D//2
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    if BLOCK_HALF > 256:
        BLOCK_HALF = 256

    grid = (B * H,)
    _rope_swap_halves_kernel[grid](
        q_rope,
        cos, sin,
        B=B, H=H, D=D,
        stride_xb=q_rope.stride(0),
        stride_xh=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        stride_c=cos.stride(0), stride_s=sin.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )


# ----------------------------------------------------------------------
# 1️⃣  Cached RoPE cosine / sine tables (generated once per config)
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Return (cos, sin) tables of shape (max_seq_len, dim) in BF16.
    ``dim`` must be even.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta_i = 10000^{-i/half}
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (T,1)
        idx = pos * theta  # (T, half)
        idx = torch.cat([idx, idx], dim=-1)  # (T, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# 2️⃣  Triton row‑wise softmax (bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr,               # [R, C]  bf16
    in_ptr,                # [R, C]  bf16
    stride_out, stride_in,
    N: tl.constexpr,       # number of columns (C)
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)

    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---- max ----
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---- exp & sum ----
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---- normalize ----
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)


def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Row‑wise softmax for a 2‑D bf16 tensor using Triton.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    N, C = x.shape

    if C <= 32:
        BLOCK = 32
    elif C <= 64:
        BLOCK = 64
    elif C <= 128:
        BLOCK = 128
    else:
        BLOCK = 1 << (C - 1).bit_length()
        BLOCK = min(BLOCK, 1024)

    out = torch.empty_like(x)
    grid = (N,)
    _softmax_kernel[grid](
        out,
        x,
        out.stride(0),
        x.stride(0),
        C,
        BLOCK_SIZE=BLOCK,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------------
# 3️⃣  Optimised MLA forward (core kernel)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑Head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # updated KV‑cache tensor (same device)
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # unpack config (readability)
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # always 1 for the forward call
    nh   = config.n_heads
    dim  = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    # weight tensors (already on correct device & dtype)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight           # (dq, dim)
    wDKV  = config.KV_proj_down_weight          # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight             # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight            # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                    # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project
    # ------------------------------------------------------------------
    q_lora   = F.linear(x, wDQ)                       # (bs, sl, dq)
    kv_lora0 = F.linear(x, wDKV)                      # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  KV‑cache update (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora0)   # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                # absolute position for the current query

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project the query (Q) and split
    # ------------------------------------------------------------------
    # sl==1 ⇒ squeeze before the linear
    q_up = F.linear(q_lora.squeeze(1), wUQ)          # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)       # (bs, nh, d_nope+d_rope)

    q_nope = q_up[..., :d_nope]                     # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                     # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV into latent (no‑PE) and RoPE parts
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]               # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]               # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE for query (single position) & keys (all cached positions)
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # query side – use the tiny in‑place kernel
    cos_q = cos_table[query_pos]   # (d_rope,)
    sin_q = sin_table[query_pos]   # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)   # in‑place modifies q_rope

    # key side – broadcast over the whole cache
    cos_k = cos_table[:kv_len]                # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                # (kv_len, d_rope)
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Split the KV up‑projection weight into K‑latent & V parts
    # ------------------------------------------------------------------
    # wUKV shape is ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)        # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                      # (nh, d_nope, dkv)
    wV = wUKV_view[:, d_nope:, :]                      # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                         # (nh, dkv, dv)

    # ------------------------------------------------------------------
    # 7️⃣  Project the “no‑PE” query into the latent space (q_nope_latent)
    # ------------------------------------------------------------------
    # q_nope: (bs, nh, d_nope)   wK: (nh, d_nope, dkv) → (bs, nh, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # ------------------------------------------------------------------
    # 8️⃣  Compute attention scores (latent + RoPE)
    # ------------------------------------------------------------------
    # latent part
    kv_nope_T = kv_nope_input.transpose(1, 2)                 # (bs, dkv, kv_len)
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)       # (bs, nh, kv_len)

    # RoPE part
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))   # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale               # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Softmax (Triton) → attention weights
    # ------------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)                 # (B*H, kv_len)
    attn_flat = _triton_softmax(scores_flat)                   # (B*H, kv_len)
    attn = attn_flat.view(bs, nh, kv_len)                     # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 🔟  Weighted sum of latent keys (M)
    # ------------------------------------------------------------------
    # attn @ kv_nope_input → (bs, nh, dkv)
    M = torch.matmul(attn, kv_nope_input)                     # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣ Project aggregated latent keys to per‑head values
    # ------------------------------------------------------------------
    # M: (bs, nh, dkv)   wV_T: (nh, dkv, dv) → (bs, nh, dv)
    y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)           # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣2️⃣ Final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                          # (bs, nh*dv)
    y = y.unsqueeze(1)                                       # (bs, 1, nh*dv)
    output = F.linear(y, wO)                                 # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data