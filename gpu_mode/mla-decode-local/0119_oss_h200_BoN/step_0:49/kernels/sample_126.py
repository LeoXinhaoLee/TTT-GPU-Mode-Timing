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
# 1️⃣  RoPE helper (rotate‑half) – pure torch (very cheap)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# 2️⃣  Cached sinusoid tables (one per config)
# ----------------------------------------------------------------------
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
        idx = pos * theta  # (max_seq_len, half)  broadcasting
        idx = torch.cat([idx, idx], dim=-1)  # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 3️⃣  Triton row‑wise softmax (bf16)
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
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# 4️⃣  The custom kernel – fully‑fused MLA forward
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # the updated KV‑cache tensor
    """
    # ------------------------------------------------------------------
    # Unpack
    # ------------------------------------------------------------------
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Alias for clarity (all tensors already on GPU & bf16)
    # --------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len          # =1 for generation
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    # Weights (already bf16 on device)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight            # (dq, dim)
    wDKV  = config.KV_proj_down_weight           # (dkv+d_rope, dim)
    wUQ   = config.Q_proj_up_weight              # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight             # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                     # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project & cache KV
    # ------------------------------------------------------------------
    with torch.no_grad():
        q_lora = F.linear(x, wDQ)                              # (bs, sl, dq)
        kv_lora_input = F.linear(x, wDKV)                     # (bs, sl, dkv+d_rope)

        kv_lora, kv_len = kv_cache(kv_lora_input)             # (bs, kv_len, dkv+d_rope)
        query_pos = kv_len - 1                                 # int, absolute position for the new query

        # ----------------------------------------------------------------------
        # 2️⃣  Up‑project queries (only one token, so squeeze seq dim)
        # ----------------------------------------------------------------------
        q_up = F.linear(q_lora.squeeze(1), wUQ)                # (bs, (d_nope+d_rope)*nh)
        q_up = q_up.view(bs, nh, d_nope + d_rope)              # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)  # (bs,nh,d_nope) & (bs,nh,d_rope)

        # ----------------------------------------------------------------------
        # 3️⃣  Split KV into latent + RoPE parts
        # ----------------------------------------------------------------------
        kv_nope_input = kv_lora[..., :dkv]                     # (bs, kv_len, dkv)
        k_rope_input  = kv_lora[..., dkv:]                     # (bs, kv_len, d_rope)

        # ----------------------------------------------------------------------
        # 4️⃣  RoPE tables (cached) – get cos / sin for positions we need
        # ----------------------------------------------------------------------
        cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

        # query side: single position
        cos_q = cos_table[query_pos]                           # (d_rope,)
        sin_q = sin_table[query_pos]                           # (d_rope,)
        q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q  # (bs, nh, d_rope)

        # key side: all cached positions (broadcast over batch)
        cos_k = cos_table[:kv_len]                             # (kv_len, d_rope)
        sin_k = sin_table[:kv_len]                             # (kv_len, d_rope)
        k_rope = k_rope_input * cos_k[None, :, :] + _rotate_half(k_rope_input) * sin_k[None, :, :]  # (bs, kv_len, d_rope)

        # ----------------------------------------------------------------------
        # 5️⃣  Decompose KV‑up‑projection weight into latent‑key and latent‑value parts
        # ----------------------------------------------------------------------
        # wUKV shape ((d_nope+dv)*nh, dkv) → (nh, d_nope+dv, dkv)
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)            # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :]                         # (nh, d_nope, dkv)
        wV = wUKV_view[:, d_nope:, :]                         # (nh, dv, dkv)
        wV_T = wV.permute(0, 2, 1)                            # (nh, dkv, dv)

        # ----------------------------------------------------------------------
        # 6️⃣  Project queries to latent space (no‑PE part)
        # ----------------------------------------------------------------------
        # q_nope: (bs, nh, d_nope), wK: (nh, d_nope, dkv)
        q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

        # ----------------------------------------------------------------------
        # 7️⃣  Compute attention scores (latent + RoPE)
        # ----------------------------------------------------------------------
        # latent part
        scores_nope = torch.einsum('bhd,bld->bhl', q_nope_latent, kv_nope_input)   # (bs, nh, kv_len)
        # RoPE part
        scores_rope = torch.einsum('bhd,bld->bhl', q_rope, k_rope)                 # (bs, nh, kv_len)

        scale = 1.0 / math.sqrt(d_nope + d_rope)
        scores = (scores_nope + scores_rope) * scale                               # (bs, nh, kv_len)

        # ----------------------------------------------------------------------
        # 8️⃣  Softmax (Triton) → attention weights
        # ----------------------------------------------------------------------
        bs, nh, kv_len = scores.shape
        scores_flat = scores.view(bs * nh, kv_len)                                 # (B*H, L)
        attn_flat = _triton_softmax(scores_flat)                                   # (B*H, L)
        attn = attn_flat.view(bs, nh, kv_len)                                      # (bs, nh, kv_len)

        # ----------------------------------------------------------------------
        # 9️⃣  Weighted sum of latent keys  M = attn @ kv_nope_input
        # ----------------------------------------------------------------------
        M = torch.einsum('bhl,bld->bhd', attn, kv_nope_input)                     # (bs, nh, dkv)

        # ----------------------------------------------------------------------
        # 🔟  Project aggregated latent keys to per‑head values
        # ----------------------------------------------------------------------
        y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)                            # (bs, nh, dv)

        # ----------------------------------------------------------------------
        # 1️⃣1️⃣  Merge heads & final linear projection
        # ----------------------------------------------------------------------
        y_head_flat = y_head.reshape(bs, nh * dv)                                  # (bs, nh*dv)
        y = F.linear(y_head_flat, wO)                                              # (bs, dim)
        output = y.unsqueeze(1)                                                    # (bs, 1, dim)

    # ----------------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # ----------------------------------------------------------------------
    return output, kv_cache.data