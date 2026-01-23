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
#  RoPE utilities (cached cosine / sine tables & inplace Triton kernel)
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

@triton.jit
def _rope_swap_halves_kernel(
    x_ptr,                # [B, T, D]   bf16
    cos_ptr, sin_ptr,    # [T, D] or [D] (broadcast possible)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,     # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # pointers for the two halves of x
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + offs * stride_xd                 # first half
    x1_ptr = x_base + (half + offs) * stride_xd        # second half

    # cosine / sine pointers (broadcast over t if stride_*_t == 0)
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t

    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d

    # load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE (rotate‑half)   out0 = x0*c - x1*s , out1 = x1*c + x0*s
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # store back
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def _rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    Apply RoPE to a tensor of shape (batch, n_heads, d_rope) in‑place.
    `cos_q` / `sin_q` are 1‑D vectors of length d_rope (position‑specific).
    """
    assert q_rope.is_cuda
    assert q_rope.shape[-1] % 2 == 0
    bs, nh, d_rope = q_rope.shape

    half = d_rope // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    if BLOCK_HALF > 256:
        BLOCK_HALF = 256

    grid = (bs * nh,)

    _rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=bs, T=nh, D=d_rope,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        # broadcast across the "time" dimension (head) → stride = 0
        stride_cos_t=0, stride_cos_d=cos_q.stride(0),
        stride_sin_t=0, stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

# ----------------------------------------------------------------------
#  Optimised MLA forward (uses Flash‑Attention via torch.nn.functional.scaled_dot_product_attention)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim)  (bfloat16)
    kv_cache_tensor : torch.Tensor   # the updated KV‑cache tensor
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # unpack config (all values are small python ints → cheap)
    # ------------------------------------------------------------------
    bs   = config.batch_size               # 128
    nh   = config.n_heads                  # 128
    dq   = config.q_lora_rank              # 1536
    dkv  = config.kv_lora_rank             # 512
    d_nope = config.qk_nope_head_dim       # may be 0‑or‑positive (e.g. 0)
    d_rope = config.qk_rope_head_dim       # 64
    dv   = config.v_head_dim               # 128
    msl  = config.max_seq_len              # 8192

    # ------------------------------------------------------------------
    # weights (already on GPU, bfloat16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight        # (dq, dim)
    wDKV  = config.KV_proj_down_weight       # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight          # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight         # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                 # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑projection
    # ------------------------------------------------------------------
    # x : (bs, 1, dim)
    q_lora = F.linear(x, wDQ)                     # (bs, 1, dq)
    kv_lora_input = F.linear(x, wDKV)             # (bs, 1, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  KV‑cache update (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)    # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                         # absolute position for the new token

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries
    # ------------------------------------------------------------------
    q_lora_s = q_lora.squeeze(1)                     # (bs, dq)
    q_up = F.linear(q_lora_s, wUQ)                    # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)        # (bs, nh, d_nope+d_rope)

    if d_nope > 0:
        q_nope = q_up[..., :d_nope]                  # (bs, nh, d_nope)
        q_rope = q_up[..., d_nope:]                  # (bs, nh, d_rope)
    else:
        q_nope = None
        q_rope = q_up                                # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  RoPE on queries (in‑place Triton kernel)
    # ------------------------------------------------------------------
    if d_rope > 0:
        cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
        cos_q = cos_table[query_pos]                  # (d_rope,)
        sin_q = sin_table[query_pos]                  # (d_rope,)
        _rope_inplace_query(q_rope, cos_q, sin_q)     # modifies q_rope in‑place

    # ------------------------------------------------------------------
    # 5️⃣  KV‑up‑projection (latent keys + values)
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]                # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]                # (bs, kv_len, d_rope)

    # Up‑project latent part → (bs, kv_len, (d_nope+dv)*nh)
    kv_up = F.linear(kv_nope_input, wUKV)            # (bs, kv_len, (d_nope+dv)*nh)
    kv_up = kv_up.view(bs, kv_len, nh, d_nope + dv)  # (bs, kv_len, nh, d_nope+dv)

    if d_nope > 0:
        k_nope = kv_up[..., :d_nope]                # (bs, kv_len, nh, d_nope)
    else:
        k_nope = None
    v = kv_up[..., d_nope:]                         # (bs, kv_len, nh, dv)

    # Permute to (bs, nh, kv_len, dim) layout required by Flash‑Attention
    if d_nope > 0:
        k_nope = k_nope.permute(0, 2, 1, 3)        # (bs, nh, kv_len, d_nope)
    v = v.permute(0, 2, 1, 3)                     # (bs, nh, kv_len, dv)

    # ------------------------------------------------------------------
    # 6️⃣  RoPE on keys (vectorised – cheap for the whole cache)
    # ------------------------------------------------------------------
    if d_rope > 0:
        cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
        cos_k = cos_table[:kv_len]                 # (kv_len, d_rope)
        sin_k = sin_table[:kv_len]                 # (kv_len, d_rope)

        # broadcast cos/sin -> (bs, kv_len, d_rope)
        cos_k = cos_k.unsqueeze(0)                 # (1, kv_len, d_rope)
        sin_k = sin_k.unsqueeze(0)                 # (1, kv_len, d_rope)

        # apply RoPE: k' = k * cos + rotate_half(k) * sin
        k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)
        # expand to heads dimension
        k_rope = k_rope.unsqueeze(1).expand(-1, nh, -1, -1)   # (bs, nh, kv_len, d_rope)
    else:
        k_rope = None

    # ------------------------------------------------------------------
    # 7️⃣  Assemble Q, K for Flash‑Attention
    # ------------------------------------------------------------------
    # Q : (bs, nh, 1, d_total)
    if d_nope > 0 and d_rope > 0:
        q_total = torch.cat([q_nope, q_rope], dim=-1)     # (bs, nh, d_total)
    elif d_nope > 0:
        q_total = q_nope
    else:
        q_total = q_rope
    q_total = q_total.unsqueeze(2)                        # (bs, nh, 1, d_total)

    # K : (bs, nh, kv_len, d_total)
    if d_nope > 0 and d_rope > 0:
        k_total = torch.cat([k_nope, k_rope], dim=-1)     # (bs, nh, kv_len, d_total)
    elif d_nope > 0:
        k_total = k_nope
    else:
        k_total = k_rope

    # ------------------------------------------------------------------
    # 8️⃣  Flash‑Attention (scaled dot‑product, fused softmax + weighted sum)
    # ------------------------------------------------------------------
    # `scale=None` lets the kernel use the default 1/√d_k, which is exactly
    # what the MLA formula needs (sqrt(d_nope + d_rope)).
    attn_out = F.scaled_dot_product_attention(
        q_total,                # (bs, nh, 1, d_total)
        k_total,                # (bs, nh, kv_len, d_total)
        v,                      # (bs, nh, kv_len, dv)
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    )                           # (bs, nh, 1, dv)

    # ------------------------------------------------------------------
    # 9️⃣  Final linear projection (per‑head values -> model dimension)
    # ------------------------------------------------------------------
    y_head = attn_out.squeeze(2)                 # (bs, nh, dv)
    y = y_head.reshape(bs, nh * dv)              # (bs, nh*dv)
    y = F.linear(y, wO)                          # (bs, dim)
    y = y.unsqueeze(1)                           # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output tensor and the (now updated) KV‑cache data
    # ------------------------------------------------------------------
    return y, kv_cache.data