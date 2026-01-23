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
#  RoPE utilities (cached cosine / sine tables)
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
#  Triton kernel for in‑place RoPE (rotate‑half) on half‑precision tensors
# ----------------------------------------------------------------------
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
    BLOCK_HALF: tl.constexpr,   # processes D/2 in blocks
):
    pid = tl.program_id(0)
    bt = pid
    b = bt // T
    t = bt - b * T

    half = D // 2
    off = tl.arange(0, BLOCK_HALF)
    mask = off < half

    # base address for this (b,t) slice
    x_base = x_ptr + b * stride_xb + t * stride_xt

    # pointers to the two halves
    x0_ptr = x_base + off * stride_xd                   # first half
    x1_ptr = x_base + (half + off) * stride_xd           # second half

    # cosine / sine pointers (broadcast over T if stride_*_t == 0)
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t
    c_ptr = cos_base + off * stride_cos_d
    s_ptr = sin_base + off * stride_sin_d

    # load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr , mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr , mask=mask, other=0.0).to(tl.float32)

    # rotate‑half RoPE (out0 = x0*c - x1*s ; out1 = x1*c + x0*s)
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # store back (bf16)
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """In‑place RoPE for queries (shape [bs, nh, d_rope])."""
    assert q_rope.is_cuda
    bs, nh, d_rope = q_rope.shape
    half = d_rope // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()   # next power‑of‑2 >= half
    grid = (bs * nh,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=bs, T=nh, D=d_rope,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        stride_cos_t=0, stride_cos_d=cos_q.stride(0),
        stride_sin_t=0, stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )
    return q_rope

# ----------------------------------------------------------------------
#  Optimised forward kernel for MLA
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_data : torch.Tensor  # the updated KV‑cache tensor (full buffer)
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # always 1 for generation
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    # Extract weight tensors (already on device, bf16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv+rope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project Q and KV
    # ------------------------------------------------------------------
    # Q ↓
    q_lora = F.linear(x, wDQ)                      # (bs, sl, dq)

    # KV ↓  (both latent and rope parts)
    kv_lora_input = F.linear(x, wDKV)              # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  Update KV‑cache
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)      # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                          # absolute position of the current token

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries
    # ------------------------------------------------------------------
    # sl == 1 ⇒ squeeze before linear
    q_up = F.linear(q_lora.squeeze(1), wUQ)         # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)      # (bs, nh, d_total)

    q_nope = q_up[..., :d_nope]                    # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                    # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Up‑project KV (latent part -> per‑head K and V)
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]               # (bs, kv_len, dkv)
    # Linear mapping to (d_nope+dv)*nh per position
    kv_up = F.linear(kv_nope_input, wUKV)            # (bs, kv_len, (d_nope+dv)*nh)
    kv_up = kv_up.view(bs, kv_len, nh, d_nope + dv)  # (bs, kv_len, nh, d_nope+dv)

    k_nope = kv_up[..., :d_nope]                     # (bs, kv_len, nh, d_nope)
    v      = kv_up[..., d_nope:]                     # (bs, kv_len, nh, dv)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE for queries (in‑place) and keys (broadcast)
    # ------------------------------------------------------------------
    # Build (cos, sin) tables (cached)
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # query side (single position)
    cos_q = cos_table[query_pos]          # (d_rope,)
    sin_q = sin_table[query_pos]          # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)   # modifies q_rope in‑place

    # key side (all cached positions)
    cos_k = cos_table[:kv_len]            # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]            # (kv_len, d_rope)
    k_rope_input = kv_lora[..., dkv:]          # (bs, kv_len, d_rope)
    # broadcast cos/sin over batch
    cos_k = cos_k.unsqueeze(0)            # (1, kv_len, d_rope)
    sin_k = sin_k.unsqueeze(0)            # (1, kv_len, d_rope)
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Assemble Q, K, V for scaled‑dot‑product attention
    # ------------------------------------------------------------------
    # Q : (bs, nh, 1, d_total)
    q = torch.cat([q_nope, q_rope], dim=-1).unsqueeze(2)          # (bs, nh, 1, d_total)

    # K : (bs, nh, kv_len, d_total)
    #   - k_nope: (bs, kv_len, nh, d_nope) → (bs, nh, kv_len, d_nope)
    k_nope = k_nope.permute(0, 2, 1, 3)                           # (bs, nh, kv_len, d_nope)
    #   - k_rope: (bs, kv_len, d_rope) → (bs, nh, kv_len, d_rope)
    k_rope = k_rope[:, None, :, :].expand(-1, nh, -1, -1)        # (bs, nh, kv_len, d_rope)
    k = torch.cat([k_nope, k_rope], dim=-1)                     # (bs, nh, kv_len, d_total)

    # V : (bs, nh, kv_len, dv)
    v = v.permute(0, 2, 1, 3)                                   # (bs, nh, kv_len, dv)

    # ------------------------------------------------------------------
    # 7️⃣  Fused attention (FlashAttention / SDPA)
    # ------------------------------------------------------------------
    # Using the built‑in scaled_dot_product_attention which runs on flash‑attention kernels
    attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)  # (bs, nh, 1, dv)

    # ------------------------------------------------------------------
    # 8️⃣  Final projection (WO)
    # ------------------------------------------------------------------
    attn_out = attn_out.squeeze(2)                 # (bs, nh, dv)
    y = attn_out.reshape(bs, nh * dv)              # (bs, nh*dv)
    y = y.unsqueeze(1)                             # (bs, 1, nh*dv)
    output = F.linear(y, wO)                       # (bs, 1, dim)

    return output, kv_cache.data