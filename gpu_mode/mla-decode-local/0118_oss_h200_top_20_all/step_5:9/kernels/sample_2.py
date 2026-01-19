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
# Helper – rotate‑half (used for RoPE)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the last dimension (the operation used by RoPE)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


# ----------------------------------------------------------------------
# RoPE tables (cos/sin) – cached lazily
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    `dim` must be even.
    The tables are cached per (dim, max_seq_len, device) pair.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta[i] = 10000^{-i/half}
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)   # (max_seq_len, 1)
        idx = pos * theta[None, :]               # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)      # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# Triton soft‑max (row‑wise) – kept from the reference implementation
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
# Optimised MLA forward – custom_kernel
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward pass.
    Returns:
        output   : torch.Tensor of shape (batch, seq_len, dim) (bf16)
        kv_data  : the updated cache tensor (raw kv‑lora)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Local aliases (avoid repeated attribute look‑ups)
    # --------------------------------------------------------------
    bs   = config.batch_size          # e.g. 128
    sl   = config.seq_len             # always 1 for the benchmark
    msl  = config.max_seq_len         # 8192 in the common cfg
    nh   = config.n_heads             # 128
    d    = config.dim                 # 7168
    dq   = config.q_lora_rank         # 1536
    dkv  = config.kv_lora_rank        # 512
    d_nope = config.qk_nope_head_dim  # may be 0
    d_rope = config.qk_rope_head_dim # 64
    dv   = config.v_head_dim          # 128

    # --------------------------------------------------------------
    # Extract weight tensors (already on CUDA, bfloat16)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight            # (dq, dim)
    wDKV  = config.KV_proj_down_weight           # (dkv+d_rope, dim)
    wUQ   = config.Q_proj_up_weight              # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight             # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                     # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 0️⃣ Down‑projections (Linear without bias)
    # ------------------------------------------------------------------
    # x : (bs, sl, d) → (bs, sl, dq)  and (bs, sl, dkv+d_rope)
    q_lora   = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora0 = F.linear(x, wDKV)                    # (bs, sl, dkv+d_rope)

    # ------------------------------------------------------------------
    # 1️⃣ KV‑cache in‑place update (avoid the costly .to copy in the reference)
    # ------------------------------------------------------------------
    # kv_lora0: (bs, 1, dkv+d_rope)
    cur_len = kv_cache.seq_len                     # length already stored in the cache
    assert cur_len + kv_lora0.size(1) <= kv_cache.data.size(1), "KV Cache Exceeded"
    # in‑place write – no dtype conversion required because everything is bf16
    kv_cache.data[:, cur_len:cur_len + kv_lora0.size(1), :] = kv_lora0
    kv_cache.seq_len = cur_len + kv_lora0.size(1)   # new length
    kv_len = kv_cache.seq_len

    # ------------------------------------------------------------------
    # 2️⃣ Gather the whole cache (latents + rope) and split it
    # ------------------------------------------------------------------
    kv_lora = kv_cache.data[:, :kv_len, :]          # (bs, kv_len, dkv+d_rope)
    kv_latent_raw = kv_lora[..., :dkv]              # (bs, kv_len, dkv)   → will become V
    kv_rope_raw   = kv_lora[..., dkv:]              # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 3️⃣ Up‑project queries
    # ------------------------------------------------------------------
    # q_lora : (bs, sl, dq) → (bs, (d_nope+d_rope)*nh)
    q_lora_flat = q_lora.squeeze(1)                # (bs, dq)
    q_up = F.linear(q_lora_flat, wUQ)              # (bs, (d_nope+d_rope)*nh)

    # Split NoPE / RoPE parts of the query (support d_nope==0)
    if d_nope > 0:
        q_up = q_up.view(bs, nh, d_nope + d_rope)   # (bs, nh, d_nope+d_rope)
        q_nope = q_up[..., :d_nope]                # (bs, nh, d_nope)
        q_rope_raw = q_up[..., d_nope:]            # (bs, nh, d_rope)
    else:
        q_up = q_up.view(bs, nh, d_rope)           # (bs, nh, d_rope)
        q_nope = None
        q_rope_raw = q_up

    # ------------------------------------------------------------------
    # 4️⃣ RoPE tables (cached) – compute cos/sin once per call
    # ------------------------------------------------------------------
    cos_tbl, sin_tbl = _get_rope_tables(d_rope, msl, x.device)

    # ------------------------------------------------------------------
    # 5️⃣ Apply RoPE to queries (single token → position = kv_len-1)
    # ------------------------------------------------------------------
    query_pos = kv_len - 1
    c_q = cos_tbl[query_pos].view(1, 1, d_rope)    # (1,1,d_rope) – broadcast over batch & heads
    s_q = sin_tbl[query_pos].view(1, 1, d_rope)
    q_rope = q_rope_raw * c_q + _rotate_half(q_rope_raw) * s_q   # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣ Apply RoPE to all keys in the cache
    # ------------------------------------------------------------------
    # (kv_len, d_rope) → (1, kv_len, d_rope) → broadcast over batch
    c_k = cos_tbl[:kv_len].unsqueeze(0)            # (1, kv_len, d_rope)
    s_k = sin_tbl[:kv_len].unsqueeze(0)
    k_rope = kv_rope_raw * c_k + _rotate_half(kv_rope_raw) * s_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 7️⃣ Prepare tensors for attention
    # ------------------------------------------------------------------
    # Q : (bs, 1, nh, d_rope)
    q = q_rope.unsqueeze(1)                         # (bs, 1, nh, d_rope)
    # K : (bs, kv_len, nh, d_rope) – broadcast heads
    k = k_rope.unsqueeze(2).expand(-1, -1, nh, -1)
    # V : (bs, kv_len, nh, dkv) – broadcast heads
    v = kv_latent_raw.unsqueeze(2).expand(-1, -1, nh, -1)

    # ------------------------------------------------------------------
    # 8️⃣ Attention
    # ------------------------------------------------------------------
    # Fast path when there is no NoPE part (d_nope == 0)
    if d_nope == 0:
        # Flash‑attention (scaled dot‑product) – operates directly on the
        # raw latent values. The value projection (dkv → dv) is performed
        # *after* the attention via wV_T.
        attn_out = F.scaled_dot_product_attention(q, k, v,
                                                  dropout_p=0.0,
                                                  is_causal=False)          # (bs, 1, nh, dkv)
        Z = attn_out.squeeze(1)                     # (bs, nh, dkv)
    else:
        # General path (both rope and latent scores)
        # ---- latent query representation ----
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)  # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :]                # (nh, d_nope, dkv)
        # q_nope : (bs, nh, d_nope)   wK : (nh, d_nope, dkv) → (bs, nh, dkv)
        q_latent = torch.einsum('bhd, hdk -> bhk', q_nope, wK)

        # ---- scores ----
        scores_rope = torch.matmul(q_rope, k_rope.transpose(-1, -2))          # (bs, nh, kv_len)
        scores_lat  = torch.matmul(q_latent, kv_latent_raw.transpose(-1, -2))# (bs, nh, kv_len)
        scale = 1.0 / math.sqrt(d_nope + d_rope)
        scores = (scores_rope + scores_lat) * scale

        # ---- soft‑max (row‑wise) ----
        B, H, S = scores.shape
        attn = _triton_softmax(scores.view(B * H, S)).view(B, H, S)

        # ---- aggregated latent (Z) ----
        Z = torch.matmul(attn, kv_latent_raw)               # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 9️⃣ Project aggregated latent Z → value space (dv)
    # ------------------------------------------------------------------
    # wV_T : (nh, dkv, dv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)           # (nh, d_nope+dv, dkv)
    wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)     # (nh, dkv, dv)

    # y_head : (bs, nh, dv)
    y_head = torch.einsum('bhd, hdf -> bhf', Z, wV_T)

    # ------------------------------------------------------------------
    # 🔟 Output projection
    # ------------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)            # (bs, nh*dv)
    output = F.linear(y_head_flat, wO)                    # (bs, dim)
    output = output.unsqueeze(1)                          # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the (now‑updated) KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data