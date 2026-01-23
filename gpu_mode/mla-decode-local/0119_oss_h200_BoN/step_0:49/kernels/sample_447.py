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
#  Shared utilities (rotate‑half, rope tables, Triton ROPE kernel)
# ----------------------------------------------------------------------
_rope_cache = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Cache cosine / sine tables for RoPE (shape: max_seq_len x dim)."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len, 1)
        idx = pos * theta[None, :]                # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)       # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                      # [B, T, D] bf16
    cos_ptr, sin_ptr,           # [T, D] or [D] depending on stride
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,            # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,   # processes D/2 elements per iteration
):
    pid = tl.program_id(0)
    bt = pid
    b = bt // T
    t = bt - b * T

    half = D // 2
    off = tl.arange(0, BLOCK_HALF)
    mask = off < half

    # Base pointers
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + off * stride_xd                     # first half
    x1_ptr = x_base + (half + off) * stride_xd            # second half

    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t

    c_ptr = cos_base + off * stride_cos_d
    s_ptr = sin_base + off * stride_sin_d

    # Load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE (rotate‑half)
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # Store back
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """Apply RoPE to query tensor in‑place (shape: [B*H, 1, D])."""
    assert q_rope.is_cuda and q_rope.ndim == 3
    bs, _, d_rope = q_rope.shape
    assert d_rope % 2 == 0
    half = d_rope // 2
    # Choose a power‑of‑2 block size ≥ half, capped at 256
    BLOCK_HALF = 1 << (half - 1).bit_length()
    BLOCK_HALF = min(BLOCK_HALF, 256)

    grid = (bs,)  # one program per (B*H)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=bs, T=1, D=d_rope,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        # broadcast cos/sin across the T dimension
        stride_cos_t=0, stride_cos_d=cos_q.stride(0),
        stride_sin_t=0, stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

# ----------------------------------------------------------------------
#  Custom kernel – highly optimised MLA forward
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # updated KV‑cache tensor
    """
    config, x, kv_cache = data

    # --------------------------------------------------
    # Unpack configuration
    # --------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # always 1
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # --------------------------------------------------
    # Weight tensors (already on device & bf16)
    # --------------------------------------------------
    wDQ   = config.Q_proj_down_weight                # (dq, dim)
    wDKV  = config.KV_proj_down_weight               # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight                  # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight                 # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                         # (dim, nh*dv)

    # --------------------------------------------------
    # 1️⃣ Down‑project
    # --------------------------------------------------
    # x: (bs, 1, dim)
    q_lora = F.linear(x, wDQ)                        # (bs, 1, dq)
    kv_lora_input = F.linear(x, wDKV)                # (bs, 1, dkv + d_rope)

    # --------------------------------------------------
    # 2️⃣ Update KV‑cache
    # --------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)        # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                             # absolute position for query RoPE

    # --------------------------------------------------
    # 3️⃣ Up‑project queries
    # --------------------------------------------------
    # squeeze seq dimension because seq_len==1
    q_up = F.linear(q_lora.squeeze(1), wUQ)           # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)        # (bs, nh, d_nope+d_rope)

    q_nope = q_up[..., :d_nope]                      # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                      # (bs, nh, d_rope)

    # --------------------------------------------------
    # 4️⃣ Split KV into latent (no‑PE) and RoPE parts
    # --------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]               # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]               # (bs, kv_len, d_rope)

    # --------------------------------------------------
    # 5️⃣ RoPE for query (in‑place)
    # --------------------------------------------------
    # reshape to (B*H, 1, D) for the Triton kernel
    q_rope_view = q_rope.contiguous().view(bs * nh, 1, d_rope)

    # fetch cos/sin tables (cached)
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    cos_q = cos_table[query_pos].view(d_rope)        # (d_rope,)
    sin_q = sin_table[query_pos].view(d_rope)        # (d_rope,)

    rope_inplace_query(q_rope_view, cos_q, sin_q)
    # reshape back
    q_rope = q_rope_view.view(bs, nh, d_rope)

    # --------------------------------------------------
    # 6️⃣ RoPE for keys (vectorised, no kernel needed)
    # --------------------------------------------------
    cos_k = cos_table[:kv_len]                       # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                       # (kv_len, d_rope)

    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # --------------------------------------------------
    # 7️⃣ Project the “no‑PE” query part to the latent dimension
    # --------------------------------------------------
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)          # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                        # (nh, d_nope, dkv)

    # q_nope : (bs, nh, d_nope) ; wK : (nh, d_nope, dkv)
    q_nope_lat = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

    # --------------------------------------------------
    # 8️⃣ Concatenate latent + rope parts and compute attention scores
    # --------------------------------------------------
    # (bs, nh, dkv+d_rope)
    q_concat = torch.cat([q_nope_lat, q_rope], dim=-1)

    # (bs, kv_len, dkv+d_rope)
    k_concat = torch.cat([kv_nope_input, k_rope], dim=-1)

    # score matrix (bs, nh, kv_len)
    scale = math.sqrt(d_nope + d_rope)
    scores = torch.matmul(q_concat, k_concat.transpose(-2, -1)) / scale

    # --------------------------------------------------
    # 9️⃣ Softmax (use native torch implementation – fast on H200)
    # --------------------------------------------------
    attn = F.softmax(scores, dim=-1)                     # (bs, nh, kv_len), bf16

    # --------------------------------------------------
    # 🔟 Weighted sum of latent keys (M)
    # --------------------------------------------------
    M = torch.matmul(attn, kv_nope_input)                # (bs, nh, dkv)

    # --------------------------------------------------
    # 1️⃣1️⃣ Project aggregated latent keys to per‑head values
    # --------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                        # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                           # (nh, dkv, dv)

    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)       # (bs, nh, dv)

    # --------------------------------------------------
    # 1️⃣2️⃣ Merge heads & final linear projection
    # --------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                      # (bs, nh*dv)
    y = F.linear(y, wO)                                  # (bs, dim)
    output = y.unsqueeze(1)                              # (bs, 1, dim)

    # --------------------------------------------------
    # Return output and the (now updated) KV‑cache tensor
    # --------------------------------------------------
    return output, kv_cache.data