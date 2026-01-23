### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# 1️⃣  RoPE utilities (cached cosine/sine tables + Triton kernel)
# ----------------------------------------------------------------------
_rope_cache = {}                     # (dim, max_seq_len, device) -> (cos, sin)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta_i = 10000^{-i/half}
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )                      # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
        idx = pos * theta[None, :]                        # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)               # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# --------------------------------------------------------------
# Triton kernel: rotate‑half + apply (cos,sin)  (in‑place)
# --------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                     # [B, H, D]  (bfloat16)
    cos_ptr, sin_ptr,          # [D] or [B, D] (bfloat16)
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,           # must be even
    stride_xb, stride_xh, stride_xd,
    stride_cos_b, stride_cos_d,
    stride_sin_b, stride_sin_d,
    BLOCK_HALF: tl.constexpr,  # processes D/2 elements per thread
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid - b * H

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # base pointer for the vector (b,h,:)
    x_base = x_ptr + b * stride_xb + h * stride_xh
    # split pointer for the two halves
    x0_ptr = x_base + offs * stride_xd                 # first half
    x1_ptr = x_base + (half + offs) * stride_xd        # second half

    # cos / sin pointers (broadcasted over batch/head if stride_*_b == 0)
    cos_base = cos_ptr + b * stride_cos_b
    sin_base = sin_ptr + b * stride_sin_b

    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d

    # loads
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
    Apply RoPE in‑place to a tensor of shape (B, H, D) where D is even.
    cos_q / sin_q are 1‑D tensors of length D (the position‑specific tables).
    """
    assert q_rope.is_cuda and q_rope.dtype == torch.bfloat16
    B, H, D = q_rope.shape
    assert D % 2 == 0

    # use a power‑of‑2 block that covers D/2
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    if BLOCK_HALF > 256:          # 256 is a reasonable cap for TVM‑style kernels
        BLOCK_HALF = 256

    grid = (B * H,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=B, H=H, D=D,
        stride_xb=q_rope.stride(0),
        stride_xh=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        # broadcast cos/sin across B/H by setting stride_*_b = 0
        stride_cos_b=0, stride_cos_d=cos_q.stride(0),
        stride_sin_b=0, stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

# ----------------------------------------------------------------------
# 2️⃣  Optimised forward of the MLA module
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast forward of Multi‑head Latent Attention (MLA).
    Returns
    -------
    output : torch.Tensor   # (batch, seq_len=1, dim)  bf16
    kv_cache.data : torch.Tensor   # (batch, max_seq_len, kv_lora_rank + qk_rope_head_dim)
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration (readability)
    # ------------------------------------------------------------------
    bs   = config.batch_size            # 128
    sl   = config.seq_len               # always 1 in the test harness
    nh   = config.n_heads               # 128
    dq   = config.q_lora_rank           # 1536
    dkv  = config.kv_lora_rank          # 512
    d_nope = config.qk_nope_head_dim    # 64
    d_rope = config.qk_rope_head_dim    # 64
    dv   = config.v_head_dim            # 128
    max_seq_len = config.max_seq_len   # 8192

    # ------------------------------------------------------------------
    # Weight tensors (already on device, bf16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight        # (dq, dim)
    wDKV  = config.KV_proj_down_weight       # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight          # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight         # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                 # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑projection
    # ------------------------------------------------------------------
    # x : (bs, sl, dim)
    q_lora   = F.linear(x, wDQ)                # (bs, sl, dq)
    kv_lora  = F.linear(x, wDKV)               # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  KV‑cache update (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora)        # kv_lora: (bs, kv_len, dkv + d_rope)
    query_pos = kv_len - 1                     # absolute position of the just‑added token

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries (Q)
    # ------------------------------------------------------------------
    # sl == 1 → squeeze before the up‑proj
    q_up = F.linear(q_lora.squeeze(1), wUQ)    # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope) # (bs, nh, d_total)
    q_nope = q_up[..., :d_nope]               # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]               # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV into latent + rope parts
    # ------------------------------------------------------------------
    kv_nope_latent = kv_lora[..., :dkv]        # (bs, kv_len, dkv)
    k_rope_input   = kv_lora[..., dkv:]        # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  KV up‑projection (produces k_nope and v)
    # ------------------------------------------------------------------
    # Linear maps (dkv) → ((d_nope+dv)*nh)
    kv_up = F.linear(kv_nope_latent, wUKV)    # (bs, kv_len, (d_nope+dv)*nh)
    kv_up = kv_up.view(bs, kv_len, nh, d_nope + dv)   # (bs, kv_len, nh, d_nope+dv)
    k_nope = kv_up[..., :d_nope]               # (bs, kv_len, nh, d_nope)
    v      = kv_up[..., d_nope:]               # (bs, kv_len, nh, dv)

    # ------------------------------------------------------------------
    # 6️⃣  RoPE for queries (in‑place Triton kernel)
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, max_seq_len, x.device)
    cos_q = cos_table[query_pos]               # (d_rope,)
    sin_q = sin_table[query_pos]               # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)   # q_rope now contains RoPE‑rotated values

    # ------------------------------------------------------------------
    # 7️⃣  RoPE for keys (vectorised – cheap because it re‑uses the cached tables)
    # ------------------------------------------------------------------
    cos_k = cos_table[:kv_len]                 # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                 # (kv_len, d_rope)

    # broadcast to (bs, kv_len, d_rope) → rotate‑half + apply cos/sin
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 8️⃣  Assemble Q, K, V for flash‑attention
    # ------------------------------------------------------------------
    # q_nope / q_rope : (bs, nh, d_*)
    # -> concatenate & add the (seq_len=1) dimension
    Q = torch.cat([q_nope, q_rope], dim=-1).unsqueeze(2)          # (bs, nh, 1, d_total)

    # k_nope : (bs, kv_len, nh, d_nope) → (bs, nh, kv_len, d_nope)
    k_nope = k_nope.permute(0, 2, 1, 3)                         # (bs, nh, kv_len, d_nope)
    # k_rope : (bs, kv_len, d_rope) → broadcast over heads
    k_rope = k_rope.unsqueeze(1).expand(-1, nh, -1, -1)          # (bs, nh, kv_len, d_rope)
    K = torch.cat([k_nope, k_rope], dim=-1)                     # (bs, nh, kv_len, d_total)

    # V : (bs, kv_len, nh, dv) → (bs, nh, kv_len, dv)
    V = v.permute(0, 2, 1, 3)                                   # (bs, nh, kv_len, dv)

    # ------------------------------------------------------------------
    # 9️⃣  Flash attention (scaled‑dot‑product) – fully fused softmax + matmul
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)    # same scaling as the reference implementation
    attn_out = F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=False,               # cache already guarantees causality
        scale=scale,
    )                                   # (bs, nh, 1, dv)

    attn_out = attn_out.squeeze(2)      # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 🔟  Final linear projection (O‑proj)
    # ------------------------------------------------------------------
    y = attn_out.reshape(bs, nh * dv).unsqueeze(1)   # (bs, 1, nh*dv)
    output = F.linear(y, wO)                         # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the (updated) KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data