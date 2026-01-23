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
# Utility helpers (rotate‑half, rope tables, softmax)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# Cached cosine / sine tables for RoPE
# ----------------------------------------------------------------------
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) – bf16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                          dtype=torch.float32,
                                          device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                     # (max_seq_len,1)
        idx = pos * theta.unsqueeze(0)                                       # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                 # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# Triton row‑wise softmax (bf16) – unchanged from the reference
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
    # pick a good block size (power‑of‑2, capped at 1024)
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
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# Custom fused MLA kernel
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast implementation of the Multi‑head Latent Attention forward pass.
    Returns
    -------
    output : torch.Tensor          # (batch, seq_len=1, dim)   bf16
    cache  : torch.Tensor          # the updated KV‑cache tensor (raw down‑projected values)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # unpack config (readability)
    # --------------------------------------------------------------
    bs   = config.batch_size                     # 128
    sl   = config.seq_len                        # always 1
    nh   = config.n_heads                        # 128
    d    = config.dim                            # 7168
    dq   = config.q_lora_rank                    # 1536
    dkv  = config.kv_lora_rank                   # 512
    d_nope = config.qk_nope_head_dim             # could be 0
    d_rope = config.qk_rope_head_dim            # 64
    dv   = config.v_head_dim                     # 128
    msl  = config.max_seq_len                    # 8192

    # --------------------------------------------------------------
    # weight tensors (already on device, bf16)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight       # (dq, dim)
    wDKV  = config.KV_proj_down_weight      # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight         # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight        # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                # (dim, nh*dv)

    ######################################################################
    # 1️⃣  Down‑project
    ######################################################################
    # x has shape (bs, 1, d); squeeze the length dimension for the linear APIs.
    x_flat = x.squeeze(1)                     # (bs, dim)

    q_lora   = F.linear(x_flat, wDQ)           # (bs, dq) – bf16
    kv_lora  = F.linear(x_flat, wDKV)          # (bs, dkv + d_rope)

    ######################################################################
    # 2️⃣  KV‑cache update (raw down‑projected values)
    ######################################################################
    # KVCache expects a 3‑D tensor (bs, seq_len, dim).  seq_len is 1 here.
    kv_lora_3d, kv_len = kv_cache(kv_lora.unsqueeze(1))   # kv_len is Python int after the call
    kv_len_int = int(kv_len)                               # ensure Python scalar
    # kv_lora_3d has shape (bs, max_seq_len, dkv + d_rope)
    # Slice the portions we need for the rest of the computation
    kv_nope_input = kv_lora_3d[:, :kv_len_int, :dkv]                  # (bs, kv_len, dkv)
    k_rope_input  = kv_lora_3d[:, :kv_len_int, dkv:]                  # (bs, kv_len, d_rope)

    ######################################################################
    # 3️⃣  Up‑project queries (low‑rank)
    ######################################################################
    q_up = F.linear(q_lora, wUQ)                       # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)          # (bs, nh, d_nope+d_rope)

    q_nope = q_up[..., :d_nope]                        # (bs, nh, d_nope) – may be empty
    q_rope = q_up[..., d_nope:]                        # (bs, nh, d_rope)

    ######################################################################
    # 4️⃣  RoPE on queries (single position) & on keys (all positions)
    ######################################################################
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ----- query side (single position) ---------------------------------
    query_pos = kv_len_int - 1                                 # absolute position of the new token
    cos_q = cos_table[query_pos]                               # (d_rope,)
    sin_q = sin_table[query_pos]                               # (d_rope,)

    # rotate‑half + apply cos / sin  (bf16 arithmetic)
    q_rope_rot = _rotate_half(q_rope)                          # (bs, nh, d_rope)
    q_rope = q_rope * cos_q + q_rope_rot * sin_q               # (bs, nh, d_rope)

    # ----- key side (all cached positions) -------------------------------
    # cos/sin tables for the whole prefix
    cos_k = cos_table[:kv_len_int]                             # (kv_len, d_rope)
    sin_k = sin_table[:kv_len_int]                             # (kv_len, d_rope)

    # broadcast over batch dimension
    cos_k = cos_k.unsqueeze(0)                                 # (1, kv_len, d_rope)
    sin_k = sin_k.unsqueeze(0)                                 # (1, kv_len, d_rope)

    k_rope_rot = _rotate_half(k_rope_input)                    # (bs, kv_len, d_rope)
    k_rope = k_rope_input * cos_k + k_rope_rot * sin_k        # (bs, kv_len, d_rope)

    ######################################################################
    # 5️⃣  Split KV‑up weights into K‑ and V‑parts (per head)
    ######################################################################
    # wUKV : ((d_nope+dv)*nh, dkv)  →  (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)

    if d_nope > 0:
        wK = wUKV_view[:, :d_nope, :]                         # (nh, d_nope, dkv)
    else:
        wK = None

    wV = wUKV_view[:, d_nope:, :]                             # (nh, dv, dkv)

    ######################################################################
    # 6️⃣  Compute attention scores (factorised)
    ######################################################################
    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # ---- rope part (always present) -----------------------------------
    # scores_rope : (bs, nh, kv_len) = q_rope @ k_rope^T   (dot over d_rope)
    scores_rope = torch.einsum('bhd,btd->bht', q_rope, k_rope)   # bf16

    # ---- No‑PE part (may be empty) ------------------------------------
    if d_nope > 0:
        # project queries into the "dkv" space:  q_nope_proj = q_nope @ wK^T
        #   q_nope : (bs, nh, d_nope)
        #   wK    : (nh, d_nope, dkv)
        q_nope_proj = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

        # scores_nope = q_nope_proj @ kv_nope_input^T   (dot over dkv)
        scores_nope = torch.einsum('bhk,btk->bht', q_nope_proj, kv_nope_input)   # (bs, nh, kv_len)

        scores = (scores_nope + scores_rope) * scale
    else:
        scores = scores_rope * scale

    ######################################################################
    # 7️⃣  Softmax (row‑wise) – Triton implementation
    ######################################################################
    # reshape to (B*H, kv_len) for the kernel
    scores_flat = scores.view(bs * nh, kv_len_int)               # (B*H, kv_len)
    attn_flat = _triton_softmax(scores_flat)                     # (B*H, kv_len)   bf16
    attn = attn_flat.view(bs, nh, kv_len_int)                    # (bs, nh, kv_len)

    ######################################################################
    # 8️⃣  Weighted sum of the latent keys  →  M (bs, nh, dkv)
    ######################################################################
    # M = attn @ kv_nope_input      (dot over kv_len)
    M = torch.einsum('bht,btk->bhk', attn, kv_nope_input)       # (bs, nh, dkv)

    ######################################################################
    # 9️⃣  Final projection to values (per head) → y_head (bs, nh, dv)
    ######################################################################
    # y_head = M @ wV^T     (dot over dkv)
    y_head = torch.einsum('bhk,hdk->bhd', M, wV)                 # (bs, nh, dv)

    ######################################################################
    # 🔟  Merge heads & output linear projection
    ######################################################################
    y_head_flat = y_head.reshape(bs, nh * dv)                    # (bs, nh*dv)
    output = F.linear(y_head_flat, wO)                           # (bs, dim)   bf16
    output = output.unsqueeze(1)                                 # (bs, 1, dim)

    # --------------------------------------------------------------
    # Return the output and the (updated) raw KV‑cache tensor.
    # --------------------------------------------------------------
    return output, kv_cache.data