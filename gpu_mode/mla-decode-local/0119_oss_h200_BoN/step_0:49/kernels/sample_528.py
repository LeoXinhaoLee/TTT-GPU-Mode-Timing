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
# Helper kernels
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,            # [B, H, L, D] (flattened)
    cos_ptr, sin_ptr, # [D]   (broadcasted)
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,            # must be even
    stride_xb, stride_xh, stride_xl, stride_xd,
    stride_cos_d, stride_sin_d,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    # 1‑D launch: pid = b*H + h
    b = pid // H
    h = pid - b * H

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)          # indices inside each half
    mask = offs < half

    # base pointer for the vector we are rotating
    x_base = x_ptr + b * stride_xb + h * stride_xh
    # left half
    x0_ptr = x_base + offs * stride_xd
    # right half (half offset)
    x1_ptr = x_base + (half + offs) * stride_xd

    # load cosine / sine (same for all positions, broadcasted)
    cos_ptr = cos_ptr + offs * stride_cos_d
    sin_ptr = sin_ptr + offs * stride_sin_d
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(cos_ptr, mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(sin_ptr, mask=mask, other=0.0).to(tl.float32)

    # RoPE with rotate‑half (swap‑halves)
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    In‑place RoPE for the query half.
    q_rope : (B, H, D)  Bfloat16, D even
    cos_q / sin_q : (D,)   Bfloat16
    """
    assert q_rope.is_cuda and q_rope.dtype == torch.bfloat16
    B, H, D = q_rope.shape
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()          # next power‑of‑2
    grid = (B * H,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=B, H=H, L=1, D=D,
        stride_xb=q_rope.stride(0),
        stride_xh=q_rope.stride(1),
        stride_xl=0,           # not used because we launch per‑head
        stride_xd=q_rope.stride(2),
        stride_cos_d=cos_q.stride(0),
        stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )
    return q_rope

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

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
    # pick a power‑of‑2 block size (capped at 1024)
    if n_cols <= 32:
        BLOCK_SIZE = 32
    elif n_cols <= 64:
        BLOCK_SIZE = 64
    elif n_cols <= 128:
        BLOCK_SIZE = 128
    elif n_cols <= 256:
        BLOCK_SIZE = 256
    elif n_cols <= 512:
        BLOCK_SIZE = 512
    elif n_cols <= 1024:
        BLOCK_SIZE = 1024
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

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Cached cosine / sine tables for a given dimension.
    Returns (cos, sin) of shape (max_seq_len, dim) in bfloat16.
    """
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)  # (half,)
    pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                       # (max_seq_len, 1)
    idx = pos * theta[None, :]                              # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                    # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)

# ----------------------------------------------------------------------
# Optimised MLA kernel
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑Head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor      # shape (batch, seq_len, dim), bf16
    kv_cache : torch.Tensor    # updated cache tensor (shape from KVCache)
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len           # always 1 in the tests
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    # Weights (all are torch.bfloat16 on the correct device)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑projection (very cheap because sl == 1)
    # ------------------------------------------------------------------
    q_lora = F.linear(x, wDQ)                 # (bs, 1, dq)
    kv_lora_input = F.linear(x, wDKV)          # (bs, 1, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  Update KV‑cache (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input) # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries (q) and split the two parts
    # ------------------------------------------------------------------
    q_up = F.linear(q_lora.squeeze(1), wUQ)    # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope) # (bs, nh, d_total)
    q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV (no‑PE vs RoPE) and keep the latent part for later
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]         # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]        # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE tables (cached)
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ----- query side (single position) --------------------------------
    cos_q = cos_table[query_pos].contiguous()
    sin_q = sin_table[query_pos].contiguous()
    # apply RoPE in‑place, result stays in q_rope
    rope_inplace_query(q_rope, cos_q, sin_q)

    # ----- key side (all cached positions) -----------------------------
    #   k_rope = k * cos + rotate_half(k) * sin
    cos_k = cos_table[:kv_len]                # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                # (kv_len, d_rope)
    # broadcast across batch
    cos_k = cos_k.unsqueeze(0)                # (1, kv_len, d_rope)
    sin_k = sin_k.unsqueeze(0)                # (1, kv_len, d_rope)
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Project the “no‑PE” query part into the low‑rank space
    # ------------------------------------------------------------------
    # wUKV is ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)       # (nh, d_total_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                     # (nh, d_nope, dkv)

    # q_nope: (bs, nh, d_nope), wK: (nh, d_nope, dkv) → (bs, nh, dkv)
    # use batched matmul: reshape to (bs*nh, d_nope) and then bmm with wK expanded
    if d_nope == 0:
        # edge case – no “no‑PE” dimensions, create a zero tensor
        q_nope_latent = torch.zeros(bs, nh, dkv, dtype=torch.bfloat16, device=x.device)
    else:
        # reshape for bmm
        q_nope_flat = q_nope.reshape(bs * nh, d_nope)                # (B*H, d_nope)
        wK_flat = wK.reshape(nh, d_nope, dkv)                        # (H, d_nope, dkv)
        # expand wK across the batch dimension (broadcast)
        wK_exp = wK_flat.unsqueeze(0).expand(bs, nh, d_nope, dkv)    # (B, H, d_nope, dkv)
        wK_exp = wK_exp.reshape(bs * nh, d_nope, dkv)               # (B*H, d_nope, dkv)
        q_nope_latent = torch.bmm(q_nope_flat.unsqueeze(1), wK_exp) # (B*H, 1, dkv)
        q_nope_latent = q_nope_latent.squeeze(1).view(bs, nh, dkv)   # (B, H, dkv)

    # ------------------------------------------------------------------
    # 7️⃣  Concatenate the two halves and compute the full attention scores
    # ------------------------------------------------------------------
    #   Q   : [q_nope_latent, q_rope]  → (bs, nh, dkv + d_rope)
    #   K   : [kv_nope_input, k_rope]   → (bs, kv_len, dkv + d_rope)
    Q = torch.cat([q_nope_latent, q_rope], dim=-1)   # (bs, nh, d_total)
    K = torch.cat([kv_nope_input, k_rope], dim=-1)   # (bs, kv_len, d_total)

    # batch‑wise matmul: (bs, nh, d_total) @ (bs, d_total, kv_len) → (bs, nh, kv_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = scores * scale

    # ------------------------------------------------------------------
    # 8️⃣  Softmax (row‑wise) – Triton implementation
    # ------------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)          # (B*H, L)
    attn_flat = _triton_softmax(scores_flat)           # (B*H, L) bf16
    attn = attn_flat.view(bs, nh, kv_len)              # (bs, nh, L)

    # ------------------------------------------------------------------
    # 9️⃣  Weighted sum of the latent keys (M)
    # ------------------------------------------------------------------
    # attn: (bs, nh, L)   kv_nope_input: (bs, L, dkv)
    M = torch.matmul(attn, kv_nope_input)              # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent keys to per‑head values (V)
    # ------------------------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                      # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                         # (nh, dkv, dv)

    # M: (bs, nh, dkv)  wV_T: (nh, dkv, dv) → (bs, nh, dv)
    # use batched matmul similar to the query side
    M_flat = M.reshape(bs * nh, dkv)                    # (B*H, dkv)
    wV_T_flat = wV_T.reshape(nh, dkv, dv)              # (H, dkv, dv)
    wV_exp = wV_T_flat.unsqueeze(0).expand(bs, nh, dkv, dv)
    wV_exp = wV_exp.reshape(bs * nh, dkv, dv)          # (B*H, dkv, dv)
    y_head = torch.bmm(M_flat.unsqueeze(1), wV_exp)    # (B*H, 1, dv)
    y_head = y_head.squeeze(1).view(bs, nh, dv)        # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣  Merge heads and final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                    # (bs, nh*dv)
    y = y.unsqueeze(1)                                 # (bs, 1, nh*dv)
    output = F.linear(y, wO)                           # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the updated cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data