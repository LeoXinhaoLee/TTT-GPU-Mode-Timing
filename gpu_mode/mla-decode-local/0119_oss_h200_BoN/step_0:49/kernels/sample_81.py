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
#  Utility: rotate‑half (swap the two halves of the last dimension and
#  negate the second half) – used for RoPE.
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
#  Cached RoPE tables (cos / sin) – build once per config
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                     # (max_seq_len,1)
        idx = pos * theta[None, :]                                         # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                 # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Triton soft‑max (row‑wise, BF16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,                     # [N, L] bf16
    stride_out, stride_in,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask,
                      other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask,
                      other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val,
                                                     tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(out_ptr + row_off_out + cur, mask=mask,
                      other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm,
                                                     tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
    # select a power‑of‑2 block size (capped at 1024)
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
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
#  Triton kernel for in‑place RoPE on a 3‑D tensor (B × T × D)
# ----------------------------------------------------------------------
@triton.jit
def _rope_swap_halves_kernel(
    x_ptr,                # [B, T, D]  bf16
    cos_ptr, sin_ptr,    # [T, D]  bf16   (or [1, D] if broadcasted)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,                # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,        # processes D/2 elements per iteration
):
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T

    half = D // 2
    off = tl.arange(0, BLOCK_HALF)               # 0 … BLOCK_HALF‑1
    mask = off < half

    # --------------------------------------------------------------
    #   load x (two halves)
    # --------------------------------------------------------------
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + off * stride_xd                     # first half
    x1_ptr = x_base + (half + off) * stride_xd            # second half

    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)

    # --------------------------------------------------------------
    #   load cos / sin
    # --------------------------------------------------------------
    cos_base = cos_ptr + t * stride_cos_t   # broadcast if stride_cos_t == 0
    sin_base = sin_ptr + t * stride_sin_t

    c_ptr = cos_base + off * stride_cos_d
    s_ptr = sin_base + off * stride_sin_d

    c = tl.load(c_ptr, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr, mask=mask, other=0.0).to(tl.float32)

    # --------------------------------------------------------------
    #   RoPE with rotate‑half (swap‑halves) in‑place
    #   out0 = x0 * c - x1 * s
    #   out1 = x1 * c + x0 * s
    # --------------------------------------------------------------
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def _apply_rope_inplace(x: torch.Tensor,
                        cos: torch.Tensor,
                        sin: torch.Tensor,
                        broadcast_over_t: bool = False):
    """
    In‑place RoPE using the above Triton kernel.
    *x*  : (B, T, D)  bf16
    *cos*, *sin*: (T, D)   if broadcast_over_t=False,
                 (1, D)   if broadcast_over_t=True  (i.e. same pos for all T)
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    B, T, D = x.shape
    assert D % 2 == 0

    half = D // 2
    # pick a block size that is a power‑of‑2 and ≥ half for the kernel
    BLOCK_HALF = 1 << (half - 1).bit_length()
    BLOCK_HALF = min(BLOCK_HALF, 256)   # 256 is a safe upper bound

    stride_xb = x.stride(0)
    stride_xt = x.stride(1)
    stride_xd = x.stride(2)

    # cos / sin strides
    if broadcast_over_t:
        stride_cos_t = 0
        stride_sin_t = 0
    else:
        stride_cos_t = cos.stride(0)
        stride_sin_t = sin.stride(0)
    stride_cos_d = cos.stride(1)
    stride_sin_d = sin.stride(1)

    grid = (B * T,)
    _rope_swap_halves_kernel[grid](
        x,
        cos,
        sin,
        B=B,
        T=T,
        D=D,
        stride_xb=stride_xb,
        stride_xt=stride_xt,
        stride_xd=stride_xd,
        stride_cos_t=stride_cos_t,
        stride_cos_d=stride_cos_d,
        stride_sin_t=stride_sin_t,
        stride_sin_d=stride_sin_d,
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

# ----------------------------------------------------------------------
#  Main kernel – the **optimised** MLA forward pass
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    cache  : torch.Tensor    # updated KV‑cache tensor (same object as kv_cache.data)
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration
    # ------------------------------------------------------------------
    bs   = config.batch_size            # 128
    sl   = config.seq_len               # =1 (always for the forward call)
    nh   = config.n_heads               # 128
    dq   = config.q_lora_rank           # 1536
    dkv  = config.kv_lora_rank          # 512
    d_nope = config.qk_nope_head_dim    # (usually 64 – taken from config)
    d_rope = config.qk_rope_head_dim    # 64
    dv   = config.v_head_dim            # 128
    msl  = config.max_seq_len           # 8192
    total_q_dim = d_nope + d_rope       # per‑head query dimension before split

    # ------------------------------------------------------------------
    # Weight tensors (already on the right device & dtype)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight                # (dq, dim)
    wDKV  = config.KV_proj_down_weight               # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight                  # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight                 # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                         # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project → low‑rank queries / KV
    # ------------------------------------------------------------------
    # x : (bs, sl, dim)
    q_lora   = F.linear(x, wDQ)           # (bs, sl, dq)
    kv_lora0 = F.linear(x, wDKV)          # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  Update KV‑cache (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora0)   # kv_lora : (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                  # absolute position of the **new** token

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries (single token)
    # ------------------------------------------------------------------
    # sl == 1 ⇒ squeeze before the linear
    q_up = F.linear(q_lora.squeeze(1), wUQ)           # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, total_q_dim)            # (bs, nh, d_nope+d_rope)
    q_nope = q_up[..., :d_nope]                      # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                      # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV low‑rank tensor
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]                # (bs, kv_len, dkv)   ← latent keys (no‑PE)
    k_rope_input  = kv_lora[..., dkv:]                # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  Prepare RoPE tables (cached)
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ----- 5.a  RoPE for the **query** (single position) -----
    # cos / sin for the current absolute position
    cos_q = cos_table[query_pos]            # (d_rope,)
    sin_q = sin_table[query_pos]            # (d_rope,)
    # rotate‑half + element‑wise multiplication – in‑place to avoid temporaries
    # (bs, nh, d_rope) → (bs*nh, 1, d_rope) is fine; we simply use torch ops:
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # bf16 broadcasting

    # ----- 5.b  RoPE for the **keys** (all cached positions) -----
    # cos / sin for every position up to kv_len (shape: kv_len × d_rope)
    cos_k = cos_table[:kv_len]            # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]            # (kv_len, d_rope)
    # broadcast over batch dimension
    cos_k = cos_k.unsqueeze(0)            # (1, kv_len, d_rope)
    sin_k = sin_k.unsqueeze(0)            # (1, kv_len, d_rope)
    # in‑place RoPE on k_rope_input
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Project the **no‑PE** part of the queries to the latent space
    # ------------------------------------------------------------------
    # wUKV contains both wK (for queries) and wV (for values)
    #   wUKV shape : ((d_nope+dv)*nh, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)            # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                         # (nh, d_nope, dkv)

    # einsum == batched matmul per‑head : (bs, nh, d_nope) @ (nh, d_nope, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 7️⃣  Compute attention scores (latent + RoPE)
    # ------------------------------------------------------------------
    # latent part
    kv_nope_T = kv_nope_input.transpose(1, 2)               # (bs, dkv, kv_len)
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)    # (bs, nh, kv_len)

    # RoPE part (query‑RoPE vs key‑RoPE). Both have shape (bs, nh, d_rope) and (bs, kv_len, d_rope)
    # Use einsum to avoid an explicit unsqueeze/expand
    scores_rope = torch.einsum('bhd,bkd->bhk', q_rope, k_rope)   # (bs, nh, kv_len)

    # combine & scale
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale          # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 8️⃣  Softmax (row‑wise) → attention weights
    # ------------------------------------------------------------------
    # flatten heads into the batch dimension for the Triton softmax
    scores_flat = scores.reshape(bs * nh, kv_len)          # (B*H, kv_len)
    attn_flat = _triton_softmax(scores_flat)               # (B*H, kv_len)  bf16
    attn = attn_flat.view(bs, nh, kv_len)                  # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Weighted sum of latent keys (M)
    # ------------------------------------------------------------------
    # M = Σ_i attn_i * kv_nope_i   →   (bs, nh, dkv)
    M = torch.matmul(attn, kv_nope_input)                 # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent M to values (wV)
    # ------------------------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                         # (nh, dv, dkv)
    # wV_T : (nh, dkv, dv)
    wV_T = wV.permute(0, 2, 1)                            # (nh, dkv, dv)

    # per‑head projection: (bs, nh, dkv) @ (nh, dkv, dv) → (bs, nh, dv)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)        # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣  Merge heads and final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                       # (bs, nh*dv)
    y = y.unsqueeze(1)                                    # (bs, 1, nh*dv)
    output = F.linear(y, wO)                              # (bs, 1, dim)  bf16

    # ------------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor (as required by the
    #   external interface)
    # ------------------------------------------------------------------
    return output, kv_cache.data