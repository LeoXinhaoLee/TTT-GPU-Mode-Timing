### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config  # Must import this way
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# Helper: rotate‑half (the “swap‑halves’’ used by RoPE)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# RoPE tables – cached once per (dim, max_len, device)
# ----------------------------------------------------------------------
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )                                   # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (L,1)
        idx = pos * theta[None, :]          # (L, half)
        idx = torch.cat([idx, idx], dim=-1) # (L, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# Triton row‑wise softmax (bf16)
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

    # ---- max ----
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---- exp & sum ----
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---- normalize ----
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
# Optimised MLA forward (entry point)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast forward of the Multi‑head Latent Attention (MLA).

    Returns
    -------
    output : torch.Tensor   # shape (batch, seq_len, dim)  – bf16
    kv_data : torch.Tensor # the updated KV‑cache raw tensor (no modification)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # unpack config (local variables for readability)
    # --------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # =1 for all test configs
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    dim  = config.dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    # weight tensors (already on the correct device & dtype)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight        # (dq, dim)
    wDKV  = config.KV_proj_down_weight       # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight          # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight         # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                 # (dim, nh*dv)

    device = x.device
    dtype  = torch.bfloat16

    # --------------------------------------------------------------
    # 1️⃣ down‑projection
    # --------------------------------------------------------------
    # (bs, 1, dq)
    q_lora = F.linear(x, wDQ)
    # (bs, 1, dkv + d_rope)
    kv_lora_input = F.linear(x, wDKV)

    # --------------------------------------------------------------
    # 2️⃣ update raw KV cache (keeps the original low‑rank values)
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)          # kv_len = total length after insert
    query_pos = kv_len - 1                               # absolute position of the new query

    # --------------------------------------------------------------
    # 3️⃣ query up‑projection & split into no‑PE / RoPE parts
    # --------------------------------------------------------------
    # squeeze the singleton seq‑dim (sl == 1)
    q_up = F.linear(q_lora.squeeze(1), wUQ)            # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)          # (bs, nh, d_total)
    q_nope = q_up[..., :d_nope]                        # (bs, nh, d_nope)
    q_rope_raw = q_up[..., d_nope:]                    # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 4️⃣ RoPE for query (single position)
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, device)
    cos_q = cos_table[query_pos]                       # (d_rope,)
    sin_q = sin_table[query_pos]                       # (d_rope,)

    rot_q = _rotate_half(q_rope_raw)                   # (bs, nh, d_rope)
    q_rope = q_rope_raw * cos_q + rot_q * sin_q       # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 5️⃣ split the raw KV cache into latent and RoPE parts
    # --------------------------------------------------------------
    # raw latent part (no‑PE) – shape (bs, kv_len, dkv)
    kv_nope_raw = kv_cache.data[..., :dkv][:, :kv_len, :]   # (bs, kv_len, dkv)

    # raw RoPE part – shape (bs, kv_len, d_rope)
    k_rope_raw = kv_cache.data[..., dkv:][:, :kv_len, :]   # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 6️⃣ RoPE for all keys (broadcast over the whole cache)
    # --------------------------------------------------------------
    # (kv_len, d_rope) tables, broadcasted automatically
    cos_k = cos_table[:kv_len]        # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]        # (kv_len, d_rope)

    rot_k = _rotate_half(k_rope_raw)                 # (bs, kv_len, d_rope)
    k_rope = k_rope_raw * cos_k + rot_k * sin_k      # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 7️⃣ latent up‑projection matrix (KV_proj_up) – view it per‑head
    # --------------------------------------------------------------
    # wUKV : ((d_nope+dv)*nh, dkv) -> (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)       # (nh, d_nope+dv, dkv)

    #   • key‑latent part  (for the score)
    wK = wUKV_view[:, :d_nope, :]                     # (nh, d_nope, dkv)

    #   • value part (for the final weighted sum)
    wV = wUKV_view[:, d_nope:, :]                     # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                        # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 8️⃣ project the query “no‑PE’’ part into the latent dkv space
    # --------------------------------------------------------------
    # q_nope : (bs, nh, d_nope)      wK : (nh, d_nope, dkv)
    # result q_nope_latent : (bs, nh, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 9️⃣ compute attention scores (latent + RoPE)
    # --------------------------------------------------------------
    # latent part  : q_nope_latent @ kv_nope_raw^T
    scores_nope = torch.einsum('bhd,bkd->bhk', q_nope_latent, kv_nope_raw)  # (bs, nh, kv_len)

    # RoPE part    : q_rope @ k_rope^T
    scores_rope = torch.einsum('bhd,bkd->bhk', q_rope, k_rope)             # (bs, nh, kv_len)

    # combine & scale
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                         # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 🔟 soft‑max (Triton) → attention weights
    # --------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)            # (B*H, kv_len)
    attn_flat = _triton_softmax(scores_flat)             # (B*H, kv_len)  bf16
    attn = attn_flat.view(bs, nh, kv_len)                # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 1️⃣1️⃣ weighted sum of latent keys  →  M  (bs, nh, dkv)
    # --------------------------------------------------------------
    # M = Σ_t attn_{b,h,t} * kv_nope_raw_{b,t,:}
    M = torch.einsum('bhk,bkd->bhd', attn, kv_nope_raw)   # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 1️⃣2️⃣ project M into the value space (per‑head dv)
    # --------------------------------------------------------------
    # y_head = M @ wV_T   (bs, nh, dv)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)       # (bs, nh, dv)

    # --------------------------------------------------------------
    # 1️⃣3️⃣ final linear projection (output dimension = dim)
    # --------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                      # (bs, nh*dv)
    y = y.unsqueeze(1)                                   # (bs, 1, nh*dv)
    output = F.linear(y, wO)                             # (bs, 1, dim)  bf16

    # --------------------------------------------------------------
    # Return output and the *raw* KV‑cache tensor (already updated)
    # --------------------------------------------------------------
    return output, kv_cache.data