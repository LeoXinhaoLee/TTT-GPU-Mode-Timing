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
# 0️⃣  RoPE rotate‑half kernel for queries (in‑place)
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                      # [B, H, D]  bf16
    cos_ptr, sin_ptr,           # [D]       bf16 (broadcasted)
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,            # must be even
    stride_xb, stride_xh, stride_xd,
    stride_cos_d,
    stride_sin_d,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid - b * H

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # base pointers of the queried head
    x_base = x_ptr + b * stride_xb + h * stride_xh
    # first / second half pointers
    x0_ptr = x_base + offs * stride_xd
    x1_ptr = x_base + (half + offs) * stride_xd

    # cosine / sine (same for all batches & heads)
    c_ptr = cos_ptr + offs * stride_cos_d
    s_ptr = sin_ptr + offs * stride_sin_d

    # load & cast to fp32
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE with rotate‑half (swap‑halves)
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    q_rope : (B, H, D) bf16   (D must be even)
    cos_q / sin_q : (D,) bf16
    """
    assert q_rope.is_cuda
    B, H, D = q_rope.shape
    assert D % 2 == 0
    half = D // 2
    # pick a power‑of‑2 block size (capped at 256)
    BLOCK_HALF = 1 << (half - 1).bit_length()
    BLOCK_HALF = min(BLOCK_HALF, 256)

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
# 1️⃣  RoPE helpers (cos / sin tables)
# ----------------------------------------------------------------------
_rope_cache = {}
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)
        idx = pos * theta[None, :]            # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)   # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 2️⃣  Triton row‑wise softmax (bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,              # number of columns
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
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
        N=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# 3️⃣  RoPE for keys (out‑of‑place, fused rotate‑half + multiplication)
# ----------------------------------------------------------------------
@triton.jit
def rope_key_kernel(
    in_ptr,               # raw key (B, K, D)  bf16
    out_ptr,              # transformed key (B, K, D)  bf16
    cos_ptr, sin_ptr,     # (max_seq_len, D)  bf16
    B: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_inb, stride_ink, stride_ind,
    stride_outb, stride_outk, stride_outd,
    stride_cos_k, stride_cos_d,
    stride_sin_k, stride_sin_d,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // K          # batch index
    k = pid % K           # position index (absolute)

    # base pointers for this (b, k) row
    in_base  = in_ptr  + b * stride_inb  + k * stride_ink
    out_base = out_ptr + b * stride_outb + k * stride_outk

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # original value
    x = tl.load(in_base + offs * stride_ind, mask=mask, other=0.0).to(tl.float32)

    # cosine / sine for this absolute position
    cos = tl.load(cos_ptr + k * stride_cos_k + offs * stride_cos_d,
                  mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + k * stride_sin_k + offs * stride_sin_d,
                  mask=mask, other=0.0).to(tl.float32)

    half = D // 2
    is_first = offs < half

    # ---- rotate‑half (swap & sign) ----
    # source for the first half: take from second half and negate
    src_first = half + offs                     # always >= half
    rot_first = -tl.load(in_base + src_first * stride_ind,
                         mask=is_first, other=0.0).to(tl.float32)

    # source for the second half: take from first half (no sign)
    src_second = offs - half
    # avoid negative indices when mask is false
    src_second = tl.where(is_first, 0, src_second)
    rot_second = tl.load(in_base + src_second * stride_ind,
                         mask=~is_first, other=0.0).to(tl.float32)

    rot = rot_first + rot_second

    # final RoPE value
    out = x * cos + rot * sin

    tl.store(out_base + offs * stride_outd, out.to(tl.bfloat16), mask=mask)

def apply_rope_to_keys(k_raw: torch.Tensor,
                       cos_table: torch.Tensor,
                       sin_table: torch.Tensor) -> torch.Tensor:
    """
    k_raw : (B, K, D)  bf16   (raw projection, D must be even)
    Returns a new tensor with RoPE applied.  The original k_raw is left untouched.
    """
    assert k_raw.is_cuda
    B, K, D = k_raw.shape
    assert D % 2 == 0
    # pick a power‑of‑2 block size (capped at 256)
    BLOCK_D = 1 << (D - 1).bit_length()
    BLOCK_D = min(BLOCK_D, 256)

    out = torch.empty_like(k_raw)
    grid = (B * K,)

    rope_key_kernel[grid](
        k_raw,
        out,
        cos_table,
        sin_table,
        B=B, K=K, D=D,
        stride_inb=k_raw.stride(0),
        stride_ink=k_raw.stride(1),
        stride_ind=k_raw.stride(2),
        stride_outb=out.stride(0),
        stride_outk=out.stride(1),
        stride_outd=out.stride(2),
        stride_cos_k=cos_table.stride(0),
        stride_cos_d=cos_table.stride(1),
        stride_sin_k=sin_table.stride(0),
        stride_sin_d=sin_table.stride(1),
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# 4️⃣  Optimised MLA forward (the entry point)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output   : torch.Tensor of shape (batch, seq_len, dim)  (bf16)
    kv_cache : the updated KV‑cache tensor (kv_cache.data)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # unpack configuration
    # --------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len                # always 1 for the provided configs
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope + d_rope) * nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # 1️⃣  Down‑project
    # --------------------------------------------------------------
    # x : (bs, sl, dim)   sl == 1
    q_lora      = F.linear(x, wDQ)                # (bs, 1, dq)
    kv_lora_in  = F.linear(x, wDKV)               # (bs, 1, dkv + d_rope)

    # --------------------------------------------------------------
    # 2️⃣  Update KV‑cache (in‑place)
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_in)        # kv_lora: (bs, kv_len, dkv + d_rope)
    query_pos = kv_len - 1                        # absolute position of the current token

    # --------------------------------------------------------------
    # 3️⃣  Up‑project queries
    # --------------------------------------------------------------
    q_up = F.linear(q_lora.squeeze(1), wUQ)        # (bs, (d_nope + d_rope) * nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)     # (bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]                   # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                   # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 4️⃣  Split KV into latent (no‑PE) and RoPE parts
    # --------------------------------------------------------------
    kv_nope_raw = kv_lora[..., :dkv]               # (bs, kv_len, dkv)
    k_rope_raw  = kv_lora[..., dkv:]               # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 5️⃣  RoPE on queries (in‑place) and on keys (out‑of‑place)
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # queries – single position
    rope_inplace_query(q_rope,
                       cos_table[query_pos].view(d_rope),
                       sin_table[query_pos].view(d_rope))

    # keys – whole cache (produces a new tensor)
    k_rope = apply_rope_to_keys(k_rope_raw, cos_table, sin_table)   # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 6️⃣  Prepare per‑head projection matrices for the latent space
    # --------------------------------------------------------------
    # wUKV : ((d_nope + dv) * nh, dkv)  -> view as (nh, d_nope + dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)      # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                    # (nh, d_nope, dkv)
    wV = wUKV_view[:, d_nope:, :]                    # (nh, dv, dkv)
    wK_T = wK.permute(0, 2, 1)                       # (nh, dkv, d_nope)
    wV_T = wV.permute(0, 2, 1)                       # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 7️⃣  Project queries into the latent space (no‑PE part)
    # --------------------------------------------------------------
    # q_nope : (bs, nh, d_nope) , wK_T : (nh, dkv, d_nope)
    q_proj = torch.einsum('bhd,hkd->bhk', q_nope, wK_T)   # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 8️⃣  Compute attention scores (latent + RoPE)
    # --------------------------------------------------------------
    # latent part
    scores_nope = torch.bmm(q_proj, kv_nope_raw.transpose(1, 2))  # (bs, nh, kv_len)

    # RoPE part
    scores_rope = torch.bmm(q_rope, k_rope.transpose(1, 2))       # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                 # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 9️⃣  Softmax (Triton implementation) – row‑wise over kv_len
    # --------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)        # (B*H, kv_len)
    attn_flat = _triton_softmax(scores_flat)         # (B*H, kv_len)
    attn = attn_flat.view(bs, nh, kv_len)            # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 🔟  Weighted sum of latent keys (Z)
    # --------------------------------------------------------------
    Z = torch.bmm(attn, kv_nope_raw)                 # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 1️⃣1️⃣  Project aggregated latent vector to value space
    # --------------------------------------------------------------
    y_head = torch.einsum('bhk,hkd->bhd', Z, wV_T)   # (bs, nh, dv)

    # --------------------------------------------------------------
    # 1️⃣2️⃣  Final linear projection to model dimension
    # --------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)        # (bs, nh*dv)
    output = F.linear(y_head_flat, wO)                # (bs, dim)
    output = output.unsqueeze(1)                      # (bs, 1, dim)

    # --------------------------------------------------------------
    # Return output and the updated KV‑cache tensor
    # --------------------------------------------------------------
    return output, kv_cache.data