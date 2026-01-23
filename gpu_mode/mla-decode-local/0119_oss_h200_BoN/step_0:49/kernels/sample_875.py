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
#  Utility helpers (rotate‑half, RoPE cache)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# Global cache for cosine / sine tables (one per (dim, max_seq_len, device) triple)
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    The tables are built exactly like the ``RoPE`` module in the reference code.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta = 10000 ** (-i / half)  (float32 -> bfloat16)
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                     # (max_seq_len,1)
        idx = pos * theta[None, :]                               # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                     # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Triton kernel for in‑place query RoPE (optional – kept for completeness)
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,               # [B, H, D]  bf16
    cos_ptr, sin_ptr,    # [D] or [B, D]  bf16
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,     # must be even
    stride_xb, stride_xh, stride_xd,
    stride_cos_b, stride_cos_d,
    stride_sin_b, stride_sin_d,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # pointers to the two halves of x
    x_base = x_ptr + b * stride_xb + h * stride_xh
    x0_ptr = x_base + offs * stride_xd                     # first half
    x1_ptr = x_base + (half + offs) * stride_xd            # second half

    # strides for cos / sin (broadcast over batch if stride_*_b == 0)
    cos_base = cos_ptr + b * stride_cos_b
    sin_base = sin_ptr + b * stride_sin_b

    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d

    # load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE: out0 = x0*c - x1*s, out1 = x1*c + x0*s
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    Apply RoPE to the query tensor in‑place.
    q_rope: (B, H, D)  bf16
    cos_q / sin_q: (D,) or (B, D)  bf16
    """
    B, H, D = q_rope.shape
    # Choose a power‑of‑2 block size that covers D/2
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    grid = (B * H,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=B, H=H, D=D,
        stride_xb=q_rope.stride(0),
        stride_xh=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        # broadcast cos/sin across B (set stride_*_b = 0)
        stride_cos_b=0, stride_cos_d=cos_q.stride(0),
        stride_sin_b=0, stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

# ----------------------------------------------------------------------
#  Custom kernel – MLA forward pass
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns:
        output       – (batch, seq_len, dim)    bf16
        kv_cache.data – updated cache tensor
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Unpack configuration constants
    # --------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # always 1 for the tested configs
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # --------------------------------------------------------------
    # Extract weight tensors (pre‑transposed for F.linear)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # 1️⃣ Down‑project inputs
    # --------------------------------------------------------------
    # x : (bs, sl, dim)
    q_lora = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)             # (bs, sl, dkv + drope)

    # --------------------------------------------------------------
    # 2️⃣ Update KV‑cache (in‑place) and retrieve the whole cache
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)    # kv_lora: (bs, kv_len, dkv+drope)
    query_pos = kv_len - 1                         # absolute position of the new token

    # --------------------------------------------------------------
    # 3️⃣ Up‑project queries
    # --------------------------------------------------------------
    # squeeze sequence dim (sl == 1)
    q_lora_flat = q_lora.squeeze(1)               # (bs, dq)
    q_up = F.linear(q_lora_flat, wUQ)              # (bs, nh*(d_nope+d_rope))
    q_up = q_up.view(bs, nh, d_nope + d_rope)     # (bs, nh, d_total)

    q_nope = q_up[..., :d_nope]                    # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                    # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 4️⃣ Apply RoPE to queries (single position)
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    # slices for the absolute position of the token we are generating
    cos_q = cos_table[query_pos]   # (d_rope,)
    sin_q = sin_table[query_pos]   # (d_rope,)
    # reshape for broadcasting
    cos_q = cos_q.view(1, 1, d_rope)
    sin_q = sin_q.view(1, 1, d_rope)
    # The rotate‑half + combine step
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q

    # concatenate the “no‑PE” and RoPE parts
    q = torch.cat([q_nope, q_rope], dim=-1)       # (bs, nh, d_total)
    q = q.unsqueeze(2)                           # (bs, nh, 1, d_total)

    # --------------------------------------------------------------
    # 5️⃣ Split cached KV into latent part and RoPE part
    # --------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]            # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]            # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 6️⃣ Up‑project latent KV → (k_nope, v)
    # --------------------------------------------------------------
    # linear projection for all heads at once
    kv_up = F.linear(kv_nope_input, wUKV)        # (bs, kv_len, nh*(d_nope+dv))
    kv_up = kv_up.view(bs, kv_len, nh, d_nope + dv)   # (bs, kv_len, nh, d_nope+dv)

    k_nope = kv_up[..., :d_nope]                 # (bs, kv_len, nh, d_nope)
    v      = kv_up[..., d_nope:]                 # (bs, kv_len, nh, dv)

    # bring heads to the second dimension (B, H, L, D)
    k_nope = k_nope.permute(0, 2, 1, 3)         # (bs, nh, kv_len, d_nope)
    v      = v.permute(0, 2, 1, 3)              # (bs, nh, kv_len, dv)

    # --------------------------------------------------------------
    # 7️⃣ Apply RoPE to keys (all cached positions)
    # --------------------------------------------------------------
    cos_k = cos_table[:kv_len]   # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]   # (kv_len, d_rope)
    # broadcast to batch dimension
    cos_k = cos_k.unsqueeze(0)    # (1, kv_len, d_rope)
    sin_k = sin_k.unsqueeze(0)    # (1, kv_len, d_rope)

    # rotate‑half + combine (batched)
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # repeat for every head
    k_rope = k_rope.unsqueeze(1).expand(-1, nh, -1, -1)   # (bs, nh, kv_len, d_rope)

    # --------------------------------------------------------------
    # 8️⃣ Build full key tensor and run Flash‑attention
    # --------------------------------------------------------------
    k = torch.cat([k_nope, k_rope], dim=-1)          # (bs, nh, kv_len, d_total)

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # Flash attention (scaled‑dot‑product attention)
    # Returns: (bs, nh, 1, dv)
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=False, dropout_p=0.0
    )

    # --------------------------------------------------------------
    # 9️⃣ Merge heads and final linear projection
    # --------------------------------------------------------------
    attn_out = attn_out.squeeze(2)                 # (bs, nh, dv)
    attn_out = attn_out.reshape(bs, nh * dv)      # (bs, nh*dv)

    # final output projection (dim, nh*dv) → (bs, dim)
    output = F.linear(attn_out, wO)                # (bs, dim)
    output = output.unsqueeze(1)                   # (bs, 1, dim)

    # --------------------------------------------------------------
    # Return results
    # --------------------------------------------------------------
    return output, kv_cache.data