### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATMENTS BLOCK ###

# ----------------------------------------------------------------------
#  RoPE utilities (cached cosine / sine tables + in‑place kernel)
# ----------------------------------------------------------------------
_rope_cache = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                         # (max_seq_len,1)
        idx = pos * theta[None, :]                      # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)            # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                     # [B, T, D] bf16
    cos_ptr, sin_ptr,          # [D] or [T, D] (broadcasted if stride_*_t == 0)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,           # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,   # processes D/2 elements per iteration
):
    pid = tl.program_id(0)
    b = pid // T
    t = pid - b * T

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # base address of the row we are working on
    x_base = x_ptr + b * stride_xb + t * stride_xt
    # first half and second half of the vector
    x0_ptr = x_base + offs * stride_xd                  # first half
    x1_ptr = x_base + (half + offs) * stride_xd         # second half

    # cosine / sine (may be broadcasted over T)
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t
    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d

    # load (up‑cast to fp32 for the arithmetic)
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE with rotate‑half:
    # out0 = x0*c - x1*s
    # out1 = x1*c + x0*s
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    In‑place RoPE for the query half‑embeddings.
    q_rope: (bs, nh, d_rope)  bf16
    cos_q/sin_q: (d_rope,)   bf16
    """
    assert q_rope.is_cuda
    assert q_rope.shape[-1] % 2 == 0
    bs, nh, d_rope = q_rope.shape

    half = d_rope // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
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

# ----------------------------------------------------------------------
#  Triton row‑wise softmax (bf16) – same as the reference implementation
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

    if n_cols <= 32:
        BLOCK = 32
    elif n_cols <= 64:
        BLOCK = 64
    elif n_cols <= 128:
        BLOCK = 128
    else:
        BLOCK = 1 << (n_cols - 1).bit_length()
        BLOCK = min(BLOCK, 1024)

    out = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_kernel[grid](
        out,
        x,
        out.stride(0),
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
#  Custom MLA forward – heavily‑optimised with Torch + Triton
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output: torch.Tensor    # shape (batch, seq_len, dim)  bf16
    kv_cache_tensor: torch.Tensor   # the updated KV‑cache buffer (bf16)
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration (readability)
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len                 # always 1 for the forward call
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    # Extract weights (already on device & bf16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣ Down‑projection
    # ------------------------------------------------------------------
    # x : (bs, sl, dim)
    q_lora  = F.linear(x, wDQ)                       # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)                # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣ Update KV‑cache (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)       # kv_lora : (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                           # absolute position of the current token

    # ------------------------------------------------------------------
    # 3️⃣ Up‑project queries (and split into NoPE / RoPE parts)
    # ------------------------------------------------------------------
    q_up = F.linear(q_lora.squeeze(1), wUQ)           # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)       # (bs, nh, d_nope+d_rope)

    q_nope = q_up[..., :d_nope]                     # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                     # (bs, nh, d_rope)

    # apply RoPE on the query half‑embeddings (in‑place)
    cos_q_tbl, sin_q_tbl = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_q_tbl[query_pos].view(d_rope).contiguous()   # (d_rope,)
    sin_q = sin_q_tbl[query_pos].view(d_rope).contiguous()   # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)

    # ------------------------------------------------------------------
    # 4️⃣  Compute the latent part of the query (q_nope -> dkv)
    # ------------------------------------------------------------------
    # view wUKV as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)           # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                         # (nh, d_nope, dkv)

    # q_nope : (bs, nh, d_nope)
    # produce q_nope_latent : (bs, nh, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # ------------------------------------------------------------------
    # 5️⃣  Prepare keys (KV cache) and apply RoPE on the rope part
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]                     # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]                     # (bs, kv_len, d_rope)

    # RoPE tables for keys (all positions up to kv_len)
    cos_k_tbl, sin_k_tbl = _get_rope_tables(d_rope, msl, x.device)
    cos_k = cos_k_tbl[:kv_len]   # (kv_len, d_rope)
    sin_k = sin_k_tbl[:kv_len]   # (kv_len, d_rope)

    # broadcast to batch dimension and apply rotate‑half
    k_rope_rot = _rotate_half(k_rope_input)               # (bs, kv_len, d_rope)
    k_rope = k_rope_input * cos_k + k_rope_rot * sin_k    # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Assemble full Q and K matrices for attention
    # ------------------------------------------------------------------
    Q_full = torch.cat([q_nope_latent, q_rope], dim=-1)   # (bs, nh, dkv + d_rope)
    K_full = torch.cat([kv_nope_input, k_rope], dim=-1)   # (bs, kv_len, dkv + d_rope)

    # ------------------------------------------------------------------
    # 7️⃣  Compute scaled dot‑product scores
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)   # same as sqrt(dkv + d_rope) when d_nope == dkv
    scores = torch.einsum('bhd,bnd->bhn', Q_full, K_full) * scale   # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 8️⃣  Softmax (row‑wise) – Triton implementation
    # ------------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)            # (B*H, N)
    attn_flat = _triton_softmax(scores_flat)              # (B*H, N)  bf16
    attn = attn_flat.view(bs, nh, kv_len)                # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Weighted aggregation of latent keys (M = attn @ kv_nope)
    # ------------------------------------------------------------------
    # attn : (bs, nh, kv_len)   kv_nope_input : (bs, kv_len, dkv)
    M = torch.einsum('bhn,bnd->bhd', attn, kv_nope_input)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent representation to per‑head values
    # ------------------------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                        # (nh, dv, dkv)
    y_head = torch.einsum('bhd,hvd->bhv', M, wV)          # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣ Merge heads and final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                      # (bs, nh*dv)
    y = y.unsqueeze(1)                                   # (bs, 1, nh*dv)
    output = F.linear(y, wO)                             # (bs, 1, dim)   bf16

    # Return the output and the updated KV‑cache tensor.
    return output, kv_cache.data