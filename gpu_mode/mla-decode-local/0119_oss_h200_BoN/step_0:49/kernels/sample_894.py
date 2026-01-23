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
# 0️⃣  RoPE utilities (cached cosine / sine tables)
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
        idx = pos * theta[None, :]                      # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)             # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 1️⃣  Triton kernel: rotate‑half + apply RoPE (in‑place)
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                      # [B,T,D]  bf16/fp16/fp32
    cos_ptr, sin_ptr,           # [T,D]   or [D]   (broadcast possible)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,            # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,   # processes D/2 in blocks (power‑of‑2)
):
    pid = tl.program_id(0)
    b = pid // T
    t = pid - b * T

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # pointers for the two halves of x
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + offs * stride_xd                     # first half
    x1_ptr = x_base + (half + offs) * stride_xd            # second half

    # pointers for cos / sin (may be broadcast along T)
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t
    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d

    # load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE: out0 = x0*c - x1*s , out1 = x1*c + x0*s
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # store back
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def _rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    q_rope : (B, H, D)  bf16
    cos_q , sin_q : (D,)  bf16    (single position)
    """
    assert q_rope.is_cuda
    B, H, D = q_rope.shape
    assert D % 2 == 0

    # block size for the half‑dimension (next power‑of‑2)
    half = D // 2
    BLOCK_HALF = 1 << ((half - 1).bit_length())
    if BLOCK_HALF > 256:
        BLOCK_HALF = 256

    grid = (B * H,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=B, T=H, D=D,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        stride_cos_t=0,                     # broadcast cos/sin over heads
        stride_cos_d=cos_q.stride(0),
        stride_sin_t=0,
        stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

def _rope_inplace_kv(k_rope: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    k_rope : (B, T, D)  bf16   (T = kv_len)
    cos    : (max_seq_len, D)  bf16
    sin    : (max_seq_len, D)  bf16
    """
    B, T, D = k_rope.shape
    assert D % 2 == 0

    half = D // 2
    BLOCK_HALF = 1 << ((half - 1).bit_length())
    if BLOCK_HALF > 256:
        BLOCK_HALF = 256

    grid = (B * T,)

    rope_swap_halves_kernel[grid](
        k_rope,
        cos, sin,
        B=B, T=T, D=D,
        stride_xb=k_rope.stride(0),
        stride_xt=k_rope.stride(1),
        stride_xd=k_rope.stride(2),
        stride_cos_t=cos.stride(0),
        stride_cos_d=cos.stride(1),
        stride_sin_t=sin.stride(0),
        stride_sin_d=sin.stride(1),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

# ----------------------------------------------------------------------
# 2️⃣  Custom MLA forward – heavily‑optimised
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output   : torch.Tensor   shape (batch, seq_len, dim)  (bf16)
    kv_tensor: torch.Tensor   the updated KV‑cache data field
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Unpack configuration (w/ descriptive names)
    # --------------------------------------------------------------
    bs   = config.batch_size               # batch size
    sl   = config.seq_len                  # token length for this step (usually 1)
    nh   = config.n_heads                  # number of attention heads
    dim  = config.dim                     # model dimension
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    d_total = d_nope + d_rope
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # --------------------------------------------------------------
    # Weight tensors (pre‑loaded on device, bf16)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight      # (dq, dim)
    wDKV  = config.KV_proj_down_weight     # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight        # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight       # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight               # (dim, nh*dv)

    # --------------------------------------------------------------
    # 1️⃣  Queries – down‑project then up‑project (both BF16)
    # --------------------------------------------------------------
    q_lora = F.linear(x, wDQ)                           # (bs, sl, dq)
    q_up   = F.linear(q_lora, wUQ)                       # (bs, sl, d_total*nh)
    q_up   = q_up.view(bs, sl, nh, d_total)             # (bs, sl, nh, d_total)

    # split into NoPE / RoPE parts
    q_nope = q_up[..., :d_nope]                         # (bs, sl, nh, d_nope)
    q_rope = q_up[..., d_nope:]                         # (bs, sl, nh, d_rope)

    # squeeze sequence dim (sl == 1 in the usual decode step)
    q_nope = q_nope.squeeze(1)                          # (bs, nh, d_nope)
    q_rope = q_rope.squeeze(1)                          # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 2️⃣  KV cache – down‑project and store
    # --------------------------------------------------------------
    kv_lora_input = F.linear(x, wDKV)                    # (bs, sl, dkv + d_rope)
    kv_lora, kv_len = kv_cache(kv_lora_input)           # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                               # absolute position for the new query

    # --------------------------------------------------------------
    # 3️⃣  Split KV into latent and RoPE parts
    # --------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]                   # (bs, kv_len, dkv)   latent part
    k_rope_input  = kv_lora[..., dkv:]                   # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 4️⃣  Up‑project the latent KV to (key‑latent + value) space
    # --------------------------------------------------------------
    # flatten the batch‑kv dimension for a single large GEMM
    kv_nope_flat = kv_nope_input.reshape(-1, dkv)        # (bs*kv_len, dkv)
    kv_up_flat   = F.linear(kv_nope_flat, wUKV)          # (bs*kv_len, (d_nope+dv)*nh)

    # reshape back
    kv_up = kv_up_flat.view(bs, kv_len, nh, d_nope + dv) # (bs, kv_len, nh, d_nope+dv)
    k_nope = kv_up[..., :d_nope]                         # (bs, kv_len, nh, d_nope)
    v      = kv_up[..., d_nope:]                         # (bs, kv_len, nh, dv)

    # --------------------------------------------------------------
    # 5️⃣  RoPE tables (cos / sin) – cached per configuration
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ------------ 5a️⃣  Queries (single position) ------------
    cos_q = cos_table[query_pos].view(d_rope).contiguous()   # (d_rope,)
    sin_q = sin_table[query_pos].view(d_rope).contiguous()   # (d_rope,)
    _rope_inplace_query(q_rope, cos_q, sin_q)                # in‑place rotate‑half

    # ------------ 5b️⃣  Keys (all cached positions) ------------
    # apply RoPE to the raw rope‑only part
    k_rope = k_rope_input.clone()
    _rope_inplace_kv(k_rope, cos_table[:kv_len], sin_table[:kv_len])
    # expand the rotated rope part to all heads
    k_rope_exp = k_rope[:, None, :, :].expand(-1, nh, -1, -1)   # (bs, nh, kv_len, d_rope)

    # --------------------------------------------------------------
    # 6️⃣  Assemble full Q and K tensors (shape: B, H, L, D)
    # --------------------------------------------------------------
    # Queries
    q = torch.concat([q_nope, q_rope], dim=-1)               # (bs, nh, d_total)
    q = q.unsqueeze(2)                                       # (bs, nh, 1, d_total)

    # Keys
    k_nope = k_nope.permute(0, 2, 1, 3)                      # (bs, nh, kv_len, d_nope)
    k = torch.concat([k_nope, k_rope_exp], dim=-1)          # (bs, nh, kv_len, d_total)

    # Values (already (bs, kv_len, nh, dv) → (bs, nh, kv_len, dv))
    v = v.permute(0, 2, 1, 3)                               # (bs, nh, kv_len, dv)

    # --------------------------------------------------------------
    # 7️⃣  Scaled‑dot‑product attention – use Flash‑Attention
    # --------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_total)
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q,            # (B, H, L=1, D)
        k,            # (B, H, S, D)
        v,            # (B, H, S, DV)
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )                              # (B, H, 1, DV)

    y_head = attn_out.squeeze(2)    # (bs, nh, dv)

    # --------------------------------------------------------------
    # 8️⃣  Output projection
    # --------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)   # (bs, nh*dv)
    y = y.unsqueeze(1)                # (bs, 1, nh*dv)
    output = F.linear(y, wO)          # (bs, 1, dim)  bf16

    # --------------------------------------------------------------
    # Return the output and the (now updated) KV‑cache tensor
    # --------------------------------------------------------------
    return output, kv_cache.data