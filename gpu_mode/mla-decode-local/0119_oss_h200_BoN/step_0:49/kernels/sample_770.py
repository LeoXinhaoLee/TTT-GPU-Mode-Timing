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
# 0️⃣  RoPE kernel (in‑place for queries)
# ----------------------------------------------------------------------
_rope_cache = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

@triton.jit
def rope_swap_halves_kernel(
    x_ptr,               # [B, T, D]   (bfloat16/fp16/fp32)
    cos_ptr, sin_ptr,    # [T, D] or [D] depending on stride_cos_t/stride_sin_t
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,     # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    bt = pid
    b = bt // T
    t = bt - b * T

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # base pointer for the (b,t) slice
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + offs * stride_xd                    # first half
    x1_ptr = x_base + (half + offs) * stride_xd           # second half

    # cosine / sine pointers
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t
    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d

    # load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr , mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr , mask=mask, other=0.0).to(tl.float32)

    # RoPE = x0*c - x1*s , x1*c + x0*s
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # store back in‑place
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    Apply RoPE in‑place to query tensor of shape (batch, n_heads, dim_rope).
    cos_q / sin_q are 1‑D tensors of length dim_rope.
    """
    assert q_rope.is_cuda
    bs, nh, d = q_rope.shape
    assert d % 2 == 0
    half = d // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()   # next power‑of‑2 ≥ half

    grid = (bs * nh,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=bs, T=nh, D=d,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        stride_cos_t=0,          # broadcast over heads
        stride_cos_d=cos_q.stride(0),
        stride_sin_t=0,
        stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)   # (max_seq_len, 1)
        idx = pos * theta                                            # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                         # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 1️⃣  Optimised MLA forward (uses flash‑attention)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward for Multi‑Head Latent Attention.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # the updated KV‑cache field
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # unpack config
    # --------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len                 # always 1 for the supplied configs
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    dnope = config.qk_nope_head_dim      # may be 0
    drope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # --------------------------------------------------------------
    # weight tensors (be sure they are on the same device)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ   = config.Q_proj_up_weight            # ((dnope+drope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((dnope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # 1️⃣  Down‑project
    # --------------------------------------------------------------
    q_lora   = F.linear(x, wDQ)                         # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)                   # (bs, sl, dkv + drope)

    # --------------------------------------------------------------
    # 2️⃣  Update KV‑cache (in‑place)
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)           # kv_lora: (bs, kv_len, dkv+drope)
    query_pos = kv_len - 1                               # absolute position for RoPE on the query

    # --------------------------------------------------------------
    # 3️⃣  Up‑project Q
    # --------------------------------------------------------------
    # sl == 1 ⇒ squeeze before linear
    q_up = F.linear(q_lora.squeeze(1), wUQ)              # (bs, (dnope+drope)*nh)
    q_up = q_up.view(bs, nh, dnope + drope)             # (bs, nh, d_total)
    q_nope = q_up[..., :dnope]                          # (bs, nh, dnope)   may be empty
    q_rope = q_up[..., dnope:]                          # (bs, nh, drope)

    # --------------------------------------------------------------
    # 4️⃣  Split KV down‑proj, then up‑project
    # --------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]                  # (bs, kv_len, dkv)
    kv_rope_input = kv_lora[..., dkv:]                  # (bs, kv_len, drope)

    kv_up = F.linear(kv_nope_input, wUKV)               # (bs, kv_len, (dnope+dv)*nh)
    kv_up = kv_up.view(bs, kv_len, nh, dnope + dv)      # (bs, kv_len, nh, dnope+dv)

    k_nope = kv_up[..., :dnope]                         # (bs, kv_len, nh, dnope) – may be empty
    v      = kv_up[..., dnope:]                         # (bs, kv_len, nh, dv)

    # permute to (bs, nh, kv_len, …)
    k_nope = k_nope.permute(0, 2, 1, 3)                 # (bs, nh, kv_len, dnope)
    v      = v.permute(0, 2, 1, 3)                      # (bs, nh, kv_len, dv)

    # --------------------------------------------------------------
    # 5️⃣  RoPE – retrieve (cos, sin) tables (cached)
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(drope, msl, x.device)

    # ----- query side ------------------------------------------------
    # cos/sin for the current token position (1‑D)
    cos_q = cos_table[query_pos]                        # (drope,)
    sin_q = sin_table[query_pos]                        # (drope,)
    rope_inplace_query(q_rope, cos_q, sin_q)            # in‑place rotation

    # ----- key side --------------------------------------------------
    # broadcast cosine / sine over batch dimension
    cos_k = cos_table[:kv_len].unsqueeze(0)             # (1, kv_len, drope)
    sin_k = sin_table[:kv_len].unsqueeze(0)             # (1, kv_len, drope)

    # apply RoPE to keys (out‑of‑place, keep cache unchanged)
    k_rotated = kv_rope_input * cos_k + _rotate_half(kv_rope_input) * sin_k   # (bs, kv_len, drope)
    # expand to per‑head layout
    k_rope = k_rotated[:, None, :, :].expand(-1, nh, -1, -1)                # (bs, nh, kv_len, drope)

    # --------------------------------------------------------------
    # 6️⃣  Assemble full Q and K tensors (heads, seq_len, dim)
    # --------------------------------------------------------------
    # Q: (bs, nh, 1, d_total)
    q = torch.cat([q_nope, q_rope], dim=-1)    # (bs, nh, d_total)
    q = q.unsqueeze(2)                         # (bs, nh, 1, d_total)

    # K: (bs, nh, kv_len, d_total)
    k = torch.cat([k_nope, k_rope], dim=-1)   # (bs, nh, kv_len, d_total)

    # --------------------------------------------------------------
    # 7️⃣  Flash‑attention (fused QK‑softmax‑V)
    # --------------------------------------------------------------
    # Scaled‑dot‑product attention automatically uses 1/√(head_dim)
    attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)  # (bs, nh, 1, dv)
    attn_out = attn_out.squeeze(2)                # (bs, nh, dv)

    # --------------------------------------------------------------
    # 8️⃣  Final linear projection
    # --------------------------------------------------------------
    y = attn_out.reshape(bs, nh * dv).unsqueeze(1)   # (bs, 1, nh*dv)
    output = F.linear(y, wO)                         # (bs, 1, dim)

    # --------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # --------------------------------------------------------------
    return output, kv_cache.data