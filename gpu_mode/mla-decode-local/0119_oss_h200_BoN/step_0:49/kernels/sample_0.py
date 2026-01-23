### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way.
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# 1️⃣  RoPE – cached cos / sin tables
# ----------------------------------------------------------------------
_rope_cache = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len, 1)
        idx = pos * theta[None, :]          # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)  # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 2️⃣  In‑place RoPE for the query side (already present in the reference)
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                      # [B, H, D] bf16
    cos_ptr, sin_ptr,           # [D] (broadcasted)
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,            # must be even
    stride_xb, stride_xh, stride_xd,
    stride_cos_d,
    stride_sin_d,
    BLOCK_HALF: tl.constexpr,   # processes D/2 in blocks
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    half = D // 2
    off = tl.arange(0, BLOCK_HALF)
    mask = off < half

    # pointers to the two halves of x
    x_base = x_ptr + b * stride_xb + h * stride_xh
    x0_ptr = x_base + off * stride_xd                     # first half
    x1_ptr = x_base + (half + off) * stride_xd            # second half

    # cos / sin (broadcasted over B/H)
    c_ptr = cos_ptr + off * stride_cos_d
    s_ptr = sin_ptr + off * stride_sin_d

    # load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE with rotate‑half (swap‑halves)
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # store back
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """Applies RoPE *in‑place* to a tensor of shape (B, H, D)."""
    assert q_rope.is_cuda
    B, H, D = q_rope.shape
    assert D % 2 == 0

    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()               # next power‑of‑2
    grid = (B * H,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=B, H=H, D=D,
        stride_xb=q_rope.stride(0),
        stride_xh=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        stride_cos_d=cos_q.stride(0),
        stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )
# ----------------------------------------------------------------------
# 3️⃣  Triton softmax (row‑wise, bf16)
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

    # ---------- exponentiate & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalise ----------
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

# ----------------------------------------------------------------------
# 4️⃣  Custom kernel – fully‑fused MLA forward
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor      # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # the updated KV‑cache (bf16)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Hyper‑parameters (readability)
    # --------------------------------------------------------------
    bs = config.batch_size          # 128
    sl = config.seq_len             # 1 (always)
    nh = config.n_heads             # 128
    dq = config.q_lora_rank         # 1536
    dkv = config.kv_lora_rank       # 512
    d_nope = config.qk_nope_head_dim   # 64 (example)
    d_rope = config.qk_rope_head_dim    # 64
    dv = config.v_head_dim              # 128
    msl = config.max_seq_len            # 8192

    # --------------------------------------------------------------
    # Weight tensors (already on the right device and dtype)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # 1️⃣  Down‑projections
    # --------------------------------------------------------------
    q_lora   = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora_i = F.linear(x, wDKV)                  # (bs, sl, dkv + d_rope)

    # --------------------------------------------------------------
    # 2️⃣  KV‑cache handling
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_i)          # (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                          # absolute position of the new token

    # --------------------------------------------------------------
    # 3️⃣  Up‑project queries
    # --------------------------------------------------------------
    #   (bs, sl, (d_nope+d_rope)*nh)  →  (bs, nh, d_nope+d_rope)
    q_up = F.linear(q_lora.squeeze(1), wUQ)        # (bs, nh*(d_nope+d_rope))
    q_up = q_up.view(bs, nh, d_nope + d_rope)     # (bs, nh, d_total)

    q_nope = q_up[..., :d_nope]                   # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                   # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 4️⃣  Split KV into latent (no‑PE) and RoPE parts
    # --------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]            # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]            # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 5️⃣  RoPE – queries (in‑place) and keys
    # --------------------------------------------------------------
    # cached cosine / sine tables
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # query side (single position)
    cos_q = cos_table[query_pos].view(d_rope).contiguous()   # (d_rope,)
    sin_q = sin_table[query_pos].view(d_rope).contiguous()   # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)                # in‑place → q_rope

    # key side – apply RoPE to all cached positions
    #   k_rope = k * cos + rotate_half(k) * sin
    cos_k = cos_table[:kv_len]                               # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                               # (kv_len, d_rope)

    # broadcast cos / sin to (bs, kv_len, d_rope)
    cos_k = cos_k.unsqueeze(0)                               # (1, kv_len, d_rope)
    sin_k = sin_k.unsqueeze(0)                               # (1, kv_len, d_rope)

    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 6️⃣  Project the “no‑PE” query part onto the “latent” dimension
    # --------------------------------------------------------------
    # wK = wUKV_view[:, :d_nope, :]   shape (nh, d_nope, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)          # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                        # (nh, d_nope, dkv)

    # q_nope (bs, nh, d_nope)  ×  wK (nh, d_nope, dkv) → (bs, nh, dkv)
    # use einsum – it is a small per‑head matmul, cheap compared to the
    # later big GEMMs.
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 7️⃣  ATTENTION scores (no‑PE + RoPE)
    # --------------------------------------------------------------
    # ----- No‑PE part ------------------------------------------------
    #   scores_nope = K (bs, kv_len, dkv)  @  Qᵀ (bs, dkv, nh)
    scores_nope = torch.bmm(kv_nope_input, q_nope_latent.transpose(1, 2))   # (bs, kv_len, nh)
    scores_nope = scores_nope.permute(0, 2, 1)                               # (bs, nh, kv_len)

    # ----- RoPE part -------------------------------------------------
    scores_rope = torch.bmm(k_rope, q_rope.transpose(1, 2))                # (bs, kv_len, nh)
    scores_rope = scores_rope.permute(0, 2, 1)                               # (bs, nh, kv_len)

    # ----- Combine & scale -------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                     # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 8️⃣  Softmax (row‑wise) – Triton implementation
    # --------------------------------------------------------------
    scores_flat = scores.reshape(bs * nh, kv_len)                 # (B*H, L)
    attn_flat = _triton_softmax(scores_flat)                     # (B*H, L)  bf16
    attn = attn_flat.view(bs, nh, kv_len)                        # (bs, nh, L)

    # --------------------------------------------------------------
    # 9️⃣  Weighted sum of latent keys  (M = attn @ kv_nope_input)
    # --------------------------------------------------------------
    #   attn : (bs, nh, kv_len)
    #   kv_nope_input : (bs, kv_len, dkv)
    # Use einsum for a single large kernel.
    M = torch.einsum('bhl,bld->bhd', attn, kv_nope_input)       # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 🔟  Project aggregated latent keys to per‑head values
    # --------------------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                               # (nh, dv, dkv)
    # wV_T : (nh, dkv, dv)
    wV_T = wV.permute(0, 2, 1)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)              # (bs, nh, dv)

    # --------------------------------------------------------------
    # 1️⃣1️⃣  Final linear projection
    # --------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)            # (bs, nh*dv)
    y = y.unsqueeze(1)                         # (bs, 1, nh*dv)
    output = F.linear(y, wO)                    # (bs, 1, dim)   bf16

    # --------------------------------------------------------------
    # Return the output and the updated cache tensor
    # --------------------------------------------------------------
    return output, kv_cache.data