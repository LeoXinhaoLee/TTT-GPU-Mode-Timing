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

# -------------------------------------------------
# Helper functions / kernels (RoPE, Softmax, etc.)
# -------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
#  RoPE kernel – in‑place rotation of query vectors (half‑dim blocks)
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                      # [B, H, D] bf16
    cos_ptr, sin_ptr,            # [D] bf16 – broadcasted over B*H
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,            # must be even
    stride_xb, stride_xh, stride_xd,
    stride_cos_d,
    stride_sin_d,
    BLOCK_HALF: tl.constexpr,   # process D/2 elements per block
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # address of the query slice (b, h, :)
    x_base = x_ptr + b * stride_xb + h * stride_xh
    # first half pointer
    x0_ptr = x_base + offs * stride_xd
    # second half pointer
    x1_ptr = x_base + (half + offs) * stride_xd

    # load x halves
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)

    # cos / sin (broadcasted across B*H)
    c_ptr = cos_ptr + offs * stride_cos_d
    s_ptr = sin_ptr + offs * stride_sin_d
    c = tl.load(c_ptr, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr, mask=mask, other=0.0).to(tl.float32)

    # RoPE (rotate‑half)
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # store back
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor,
                       cos_q: torch.Tensor,
                       sin_q: torch.Tensor) -> None:
    """
    In‑place RoPE on query tensor.
    q_rope : (B, H, D)   bf16
    cos_q , sin_q : (D,) bf16
    """
    assert q_rope.is_cuda and q_rope.dtype == torch.bfloat16
    B, H, D = q_rope.shape
    assert D % 2 == 0

    # choose a power‑of‑2 block size covering D/2
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()   # next pow2
    # grid: one program per (B*H)
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
#  Softmax – Triton implementation (row‑wise, BF16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # -------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # -------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # -------- normalize ----------
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape

    # pick a reasonable block size (power‑of‑2, <=1024)
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
        N_COLS=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
#  RoPE cache (cos / sin tables, built once per config)
# ----------------------------------------------------------------------
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                     # (max_seq_len,1)
        idx = pos * theta                                                       # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                     # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# -------------------------------------------------
#  Main kernel – MLA forward
# -------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor   # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # updated KV‑cache tensor
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration
    # ------------------------------------------------------------------
    bs = config.batch_size
    sl = config.seq_len          # always 1 in the provided tests
    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim
    msl = config.max_seq_len

    # ------------------------------------------------------------------
    # Weight tensors (already on the correct device, bf16)
    # ------------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+d_rope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project
    # ------------------------------------------------------------------
    # x : (bs, sl, dim) ; sl == 1
    q_lora = F.linear(x, wDQ)                           # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)                   # (bs, sl, dkv+d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  KV‑cache update (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)          # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = int(kv_len) - 1                         # position of the just‑added token

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries, split NoPE / RoPE parts
    # ------------------------------------------------------------------
    # squeeze because sl == 1
    q_up = F.linear(q_lora.squeeze(1), wUQ)               # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)            # (bs, nh, d_total)
    q_nope = q_up[..., :d_nope]                         # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                         # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV into latent (no‑PE) and RoPE parts
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]                   # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]                   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE tables & apply RoPE
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ---- queries ----
    cos_q = cos_table[query_pos]                         # (d_rope,)
    sin_q = sin_table[query_pos]                         # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)            # in‑place modification

    # ---- keys ----
    cos_k = cos_table[:kv_len]                           # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                           # (kv_len, d_rope)
    # broadcast across batch dimension and apply rotate‑half
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Project query “no‑PE” part into the latent space (dkv)
    # ------------------------------------------------------------------
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)            # (nh, d_total, dkv)
    wK = wUKV_view[:, :d_nope, :]                         # (nh, d_nope, dkv)
    # q_nope : (bs, nh, d_nope)      wK : (nh, d_nope, dkv) → (bs, nh, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 7️⃣  Compute attention scores (latent + RoPE)
    # ------------------------------------------------------------------
    # (bs, nh, dkv) @ (bs, dkv, kv_len) → (bs, nh, kv_len)
    kv_nope_T = kv_nope_input.transpose(1, 2)               # (bs, dkv, kv_len)
    scores_nope = torch.bmm(q_nope_latent, kv_nope_T)       # (bs, nh, kv_len)

    # (bs, nh, d_rope) @ (bs, d_rope, kv_len) → (bs, nh, kv_len)
    scores_rope = torch.bmm(q_rope, k_rope.transpose(1, 2)) # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale            # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 8️⃣  Softmax (Triton) → attention weights
    # ------------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)              # (B*H, K)
    attn_flat = _triton_softmax(scores_flat)                # (B*H, K) bf16, already normalised
    attn = attn_flat.view(bs, nh, kv_len)                  # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Weighted sum of latent keys  (M := Σ a·k_nope)
    # ------------------------------------------------------------------
    # (bs, nh, kv_len) @ (bs, kv_len, dkv) → (bs, nh, dkv)
    M = torch.bmm(attn, kv_nope_input)                     # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent keys to per‑head values
    # ------------------------------------------------------------------
    wV_T = wUKV_view[:, d_nope:, :].transpose(1, 2)        # (nh, dkv, dv)
    # M : (bs, nh, dkv)   wV_T : (nh, dkv, dv) → (bs, nh, dv)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)         # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣ Merge heads & final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                        # (bs, nh*dv)
    y = F.linear(y, wO)                                    # (bs, dim)
    y = y.unsqueeze(1)                                     # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return output and the (now updated) KV‑cache tensor
    # ------------------------------------------------------------------
    return y, kv_cache.data