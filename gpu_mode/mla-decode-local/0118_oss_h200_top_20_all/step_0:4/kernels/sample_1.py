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
# 0️⃣  Helper: rotate‑half (used for RoPE)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Swap the two halves of the last dimension and negate the second half.
    x : (..., dim)  where dim is even
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


# ----------------------------------------------------------------------
# 1️⃣  Helper: RoPE tables (cached)
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    dim must be even ( = rope_head_dim ).
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta = 10000 ** (-i/half)
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                     # (max_seq_len, 1)
        idx = pos * theta[None, :]                                            # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                   # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# 2️⃣  Triton row‑wise softmax (unchanged – already optimal)
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
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # -------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # -------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # -------- normalize ----------
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

    # pick a power‑of‑2 block size (capped at 1024)
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
# 3️⃣  Optimised MLA forward (the required custom_kernel)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor       # shape (batch, 1, dim), bf16
    kv_cache.data : torch.Tensor   # the up‑to‑date cache
    """
    # ------------------------------------------------------------------
    # Unpack arguments
    # ------------------------------------------------------------------
    config, x, kv_cache = data

    bs   = config.batch_size          # 128
    sl   = config.seq_len             # always 1 for the provided tests
    nh   = config.n_heads             # 128
    dq   = config.q_lora_rank         # 1536
    dkv  = config.kv_lora_rank        # 512
    d_nope = config.qk_nope_head_dim  # may be 0 in tests
    d_rope = config.qk_rope_head_dim # 64
    dv   = config.v_head_dim          # 128
    msl  = config.max_seq_len         # e.g. 8192

    # ------------------------------------------------------------------
    # Extract weight tensors (already on CUDA, bfloat16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight            # (dq, dim)
    wDKV  = config.KV_proj_down_weight           # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight              # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight             # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                     # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 0️⃣  Down‑project
    # ------------------------------------------------------------------
    # x : (bs, 1, dim)
    q_lora = F.linear(x, wDQ)                     # (bs, 1, dq)
    kv_lora_in = F.linear(x, wDKV)                # (bs, 1, dkv + d_rope)

    # ------------------------------------------------------------------
    # 1️⃣  KV‑cache update (in‑place)
    # ------------------------------------------------------------------
    # kv_lora : (bs, kv_len, dkv + d_rope)
    kv_lora, kv_len = kv_cache(kv_lora_in)
    query_pos = kv_len - 1                         # absolute position of the current query

    # ------------------------------------------------------------------
    # 2️⃣  Up‑project queries  (no‑PE + RoPE split)
    # ------------------------------------------------------------------
    # squeeze singleton seq dim before up‑project
    q_up = F.linear(q_lora.squeeze(1), wUQ)        # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)     # (bs, nh, d_nope+d_rope)

    q_nope = q_up[..., :d_nope]                    # (bs, nh, d_nope) – may be empty
    q_rope = q_up[..., d_nope:]                    # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 3️⃣  Split KV into latent part and RoPE part
    # ------------------------------------------------------------------
    kv_nope_raw = kv_lora[..., :dkv]               # (bs, kv_len, dkv)
    k_rope_raw  = kv_lora[..., dkv:]               # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  RoPE rotation (pure torch – tiny tensors)
    # ------------------------------------------------------------------
    # pre‑compute cosine / sine tables (cached)
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # queries
    if d_rope > 0:
        cos_q = cos_table[query_pos].view(1, 1, d_rope)   # (1,1,d_rope) – broadcastable
        sin_q = sin_table[query_pos].view(1, 1, d_rope)
        q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q
    else:
        q_rope = torch.empty_like(q_rope)  # shape (bs, nh, 0)

    # keys
    if d_rope > 0:
        # (kv_len, d_rope) → broadcast over batch
        cos_k = cos_table[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_table[:kv_len].unsqueeze(0)
        k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k
    else:
        k_rope = torch.empty_like(k_rope_raw)

    # ------------------------------------------------------------------
    # 5️⃣  Project queries into latent space (head‑wise weight wK)
    # ------------------------------------------------------------------
    # wUKV : ((d_nope+dv)*nh, dkv) → view → split into wK & wV
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)      # (nh, d_nope+dv, dkv)

    wK = wUKV_view[:, :d_nope, :]                   # (nh, d_nope, dkv)
    wV = wUKV_view[:, d_nope:, :]                   # (nh, dv, dkv)

    # wK_T : (nh, dkv, d_nope)   – ready for einsum with q_nope
    wK_T = wK.permute(0, 2, 1)                      # (nh, dkv, d_nope)

    # latent query part (B, H, dkv)
    if d_nope > 0:
        q_latent = torch.einsum('bhd,hkd->bhk', q_nope, wK_T)   # (bs, nh, dkv)
    else:
        q_latent = torch.zeros(bs, nh, dkv, dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------------
    # 6️⃣  Build scores (latent + RoPE) using broadcasted matmul
    # ------------------------------------------------------------------
    # latent part: (bs, nh, dkv) @ (bs, dkv, kv_len) → (bs, nh, kv_len)
    scores_nope = torch.matmul(q_latent, kv_nope_raw.transpose(-2, -1))   # (bs, nh, kv_len)

    # RoPE part: (bs, nh, d_rope) @ (bs, d_rope, kv_len)
    if d_rope > 0:
        scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))     # (bs, nh, kv_len)
        scores = scores_nope + scores_rope
    else:
        scores = scores_nope

    # ------------------------------------------------------------------
    # 7️⃣  Softmax (row‑wise, Triton implementation)
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)           # original scaling
    scores = scores * scale
    # flatten to (B*H, kv_len) for the Triton kernel
    scores_flat = scores.reshape(bs * nh, kv_len)
    attn_flat = _triton_softmax(scores_flat)           # (B*H, kv_len)
    attn = attn_flat.view(bs, nh, kv_len)              # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 8️⃣  Weighted sum of latent keys  (attn @ kv_nope_raw)
    # ------------------------------------------------------------------
    # attn : (bs, nh, kv_len)   kv_nope_raw : (bs, kv_len, dkv)
    Z = torch.einsum('bht,btk->bhk', attn, kv_nope_raw)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 9️⃣  Project aggregated latent vector to value space (wV)
    # ------------------------------------------------------------------
    # wV_T : (nh, dkv, dv)  – note we have wV of shape (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                         # (nh, dkv, dv)
    y_head = torch.einsum('bhd,hdv->bhv', Z, wV_T)    # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 🔟  Final linear projection back to model dimension
    # ------------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)          # (bs, nh*dv)
    output = F.linear(y_head_flat, wO)                 # (bs, dim)
    output = output.unsqueeze(1)                       # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the (now updated) KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data