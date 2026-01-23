### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# Helper utilities (RoPE tables, rotate‑half, etc.)
# ----------------------------------------------------------------------
_rope_cache = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Build (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    Cached globally to avoid recomputation across calls.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta = 10000 ** (-i/half)  (float32 for accuracy, then cast)
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half))
        theta = theta.to(torch.bfloat16)          # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
        idx = pos * theta                         # (max_seq_len, half)  (bf16 * bf16 => bf16)
        idx = torch.cat([idx, idx], dim=-1)       # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# Triton‑accelerated softmax (fallback to torch softmax if needed)
# ----------------------------------------------------------------------
import triton.language as tl

@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # max reduction
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # exp & sum
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # normalize
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Row‑wise softmax for a 2‑D bfloat16 tensor using Triton.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
    # pick a power‑of‑2 block size (capped at 1024)
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
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# Optimised MLA forward – single entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor            # shape (batch, seq_len, dim), dtype bfloat16
    kv_cache_tensor : torch.Tensor  # the updated KV‑cache data field
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration – keep terse for readability
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len               # always 1 for the provided configs
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    max_seq_len = config.max_seq_len

    # ------------------------------------------------------------------
    # Extract weight tensors (already on device, dtype bfloat16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope + d_rope) * nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑project input
    # ------------------------------------------------------------------
    q_lora = F.linear(x, wDQ)                 # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)         # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  Update the KV‑cache (in‑place) and fetch the absolute query position
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)   # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                       # integer, position of the just‑added token

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project the queries (Q) – split into no‑PE and RoPE parts
    # ------------------------------------------------------------------
    # sl == 1 → squeeze the temporal dimension for the linear projection
    q_up = F.linear(q_lora.squeeze(1), wUQ)      # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)   # (bs, nh, d_nope+d_rope)
    q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)   # (bs, nh, d_nope) & (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Prepare RoPE tables (cos / sin) – cached & reused across calls
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, max_seq_len, x.device)

    # ------------------------------------------------------------------
    # Query RoPE (single position)
    # ------------------------------------------------------------------
    cos_q = cos_table[query_pos]                # (d_rope,)
    sin_q = sin_table[query_pos]                # (d_rope,)
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  Project the query “no‑PE” part into the latent space (dkv)
    # ------------------------------------------------------------------
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)   # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                # (nh, d_nope, dkv)
    # einsum: batch b, head h, dim_nope d -> latent dkv p
    q_nope_latent = torch.einsum('bhd,hdp->bhp', q_nope, wK)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 6️⃣  Build the full query tensor (latent + RoPE) for attention
    # ------------------------------------------------------------------
    Q_comb = torch.cat([q_nope_latent, q_rope], dim=-1)   # (bs, nh, dkv+d_rope)

    # ------------------------------------------------------------------
    # 7️⃣  Split KV cache into latent (no‑PE) and RoPE parts
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]              # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]              # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 8️⃣  Apply RoPE to the key side (all cached positions)
    # ------------------------------------------------------------------
    # cos / sin for each cached position
    cos_k = cos_table[:kv_len]                      # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                      # (kv_len, d_rope)
    # broadcast to batch dimension
    cos_k = cos_k.unsqueeze(0)                      # (1, kv_len, d_rope)
    sin_k = sin_k.unsqueeze(0)                      # (1, kv_len, d_rope)
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 9️⃣  Build the full key tensor (latent + RoPE)
    # ------------------------------------------------------------------
    K_comb = torch.cat([kv_nope_input, k_rope], dim=-1)   # (bs, kv_len, dkv+d_rope)

    # ------------------------------------------------------------------
    # 10️⃣  Scaled‑dot‑product attention (fused softmax + weighted sum)
    # ------------------------------------------------------------------
    # Shapes required by torch.nn.functional.scaled_dot_product_attention:
    #   Q : (batch, heads, 1, D)
    #   K : (batch, heads, S, D)   – broadcasted across heads (no copy)
    #   V : (batch, heads, S, dkv) – broadcasted across heads
    D = dkv + d_rope
    scale = 1.0 / math.sqrt(d_nope + d_rope)   # scalar scaling factor

    Q = Q_comb.unsqueeze(2)                               # (bs, nh, 1, D)
    K = K_comb.unsqueeze(1).expand(-1, nh, -1, -1)        # (bs, nh, kv_len, D) – broadcast, no memory copy
    V = kv_nope_input.unsqueeze(1).expand(-1, nh, -1, -1)  # (bs, nh, kv_len, dkv)

    # The fused attention kernel (CUDA) works with bfloat16 directly.
    M = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )                                     # (bs, nh, 1, dkv)
    M = M.squeeze(2)                        # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 11️⃣  Project the aggregated latent keys to per‑head values
    # ------------------------------------------------------------------
    wV = wUKV_view[:, d_nope:, :]                # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)                   # (nh, dkv, dv)
    y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)   # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 12️⃣  Merge heads & final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)               # (bs, nh*dv)
    y = y.unsqueeze(1)                            # (bs, 1, nh*dv)
    output = F.linear(y, wO)                      # (bs, 1, dim)  (BF16)

    # ------------------------------------------------------------------
    # Return the output tensor and the updated KV‑cache data field
    # ------------------------------------------------------------------
    return output, kv_cache.data