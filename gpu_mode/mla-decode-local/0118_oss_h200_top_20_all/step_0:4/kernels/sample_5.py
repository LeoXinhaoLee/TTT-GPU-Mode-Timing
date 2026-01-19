### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config   # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# Helper utilities : RoPE cache, rotate‑half and a tiny Triton Softmax
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


_rope_cache = {}
def _get_rope_tables(d_rope: int, max_seq_len: int, device: torch.device):
    """
    Build (cos, sin) tables for a given RoPE dimension and max sequence length.
    The tables are cached for the life‑time of the Python process.
    """
    key = (d_rope, max_seq_len, device)
    if key not in _rope_cache:
        half = d_rope // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)        # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                              # (max_seq_len, 1)
        idx = pos * theta[None, :]                                                 # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                        # (max_seq_len, d_rope)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# Triton row‑wise soft‑max (bf16 → fp32 accumulation → bf16 output)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N_COLS,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    off_in  = row * stride_in
    off_out = row * stride_out

    # ----- max (fp32) --------------------------------------------------
    max_val = tl.full([BLOCK_SIZE], -float('inf'), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ----- exp & sum (fp32) --------------------------------------------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + off_in + cur, mask=mask, other=-float('inf'))
        e = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + off_out + cur, tl.cast(e, tl.bfloat16), mask=mask)
        sum_val += e
    row_sum = tl.sum(sum_val)

    # ----- normalize ----------------------------------------------------
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(out_ptr + off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)


def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using the Triton kernel above."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
    # pick a block size (next power‑of‑2, capped at 1024)
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
# Custom kernel – a fused, Triton‑friendly forward for MLA
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched, Triton‑accelerated forward pass for the Multi‑head Latent
    Attention module.
    Returns
    -------
    output : torch.Tensor          # shape (batch, seq_len, dim) – bf16
    kv_cache.data : torch.Tensor   # the (now‑updated) KV‑cache tensor
    """
    # --------------------------------------------------------------
    # Unpack configuration & tensors
    # --------------------------------------------------------------
    config, x, kv_cache = data

    bs   = config.batch_size          # B
    sl   = config.seq_len             # = 1 (generation)
    msl  = config.max_seq_len
    nh   = config.n_heads
    dim  = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim

    # ------------------------------------------------------------------
    # Weight tensors – already on the correct device / dtype
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight                # (dq, dim)
    wDKV  = config.KV_proj_down_weight               # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight                  # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight                 # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                         # (dim, nh*dv)

    # --------------------------------------------------------------
    # 1️⃣  Fused down‑projection + KV‑cache update
    # --------------------------------------------------------------
    # x : (B, 1, dim) → (B, dim)
    x = x.squeeze(1)                                                # (B, dim)

    # One GEMM for BOTH Q‑down and KV‑down
    w_down = torch.cat([wDQ, wDKV], dim=0)                         # (dq+dkv+d_rope, dim)
    proj   = F.linear(x, w_down)                                   # (B, dq+dkv+d_rope)

    q_lora    = proj[:, :dq]                                       # (B, dq)
    kv_lora0  = proj[:, dq:]                                       # (B, dkv+d_rope)

    # Insert the fresh token into the cache (kv_lora0 is  (B,1,dkv+d_rope) )
    kv_lora0  = kv_lora0.unsqueeze(1)                              # (B,1,dkv+d_rope)
    kv_lora, kv_len = kv_cache(kv_lora0)                           # kv_lora : (B, L, dkv+d_rope)
    query_pos = kv_len - 1                                          # integer position of the new token

    # --------------------------------------------------------------
    # 2️⃣  Up‑project queries (Q) and split into No‑PE / RoPE parts
    # --------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                                    # (B, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)                      # (B, nh, d_nope+d_rope)

    q_nope, q_rope = torch.split(q_up,
                                 [d_nope, d_rope],
                                 dim=-1)                         # each (B, nh, …)

    # --------------------------------------------------------------
    # 3️⃣  Separate the cached KV tensor
    # --------------------------------------------------------------
    kv_latent = kv_lora[..., :dkv]                                  # (B, L, dkv)
    k_rope_raw = kv_lora[..., dkv:]                                 # (B, L, d_rope)

    # --------------------------------------------------------------
    # 4️⃣  RoPE tables (cached) – one table per RoPE dimension
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ----- queries -----
    cos_q = cos_table[query_pos]                                    # (d_rope,)
    sin_q = sin_table[query_pos]                                    # (d_rope,)
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q          # (B, nh, d_rope)

    # ----- keys (entire cache) -----
    cos_k = cos_table[:kv_len].unsqueeze(0)                         # (1, L, d_rope)
    sin_k = sin_table[:kv_len].unsqueeze(0)                         # (1, L, d_rope)
    k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k # (B, L, d_rope)

    # --------------------------------------------------------------
    # 5️⃣  Split KV‑up‑projection weight into K‑ and V‑parts
    # --------------------------------------------------------------
    # wUKV : ((d_nope+dv)*nh, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)                     # (nh, d_nope+dv, dkv)
    wK   = wUKV_view[:, :d_nope, :]                                 # (nh, d_nope, dkv)
    wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)               # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 6️⃣  Project the query‑No‑PE part into the latent space
    # --------------------------------------------------------------
    # q_nope : (B, nh, d_nope)   wK : (nh, d_nope, dkv)
    # Result: q_latent : (B, nh, dkv)
    q_latent = torch.einsum('bhd,hdc->bhc', q_nope, wK)              # (B, nh, dkv)

    # --------------------------------------------------------------
    # 7️⃣  Compute attention scores (latent + RoPE) and softmax
    # --------------------------------------------------------------
    # latent part
    # scores_nope : (B, nh, L)
    scores_nope = torch.matmul(q_latent, kv_latent.transpose(-2, -1)) # (B, nh, L)

    # RoPE part (broadcasted over heads automatically)
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))     # (B, nh, L)

    # Combine + scaling
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                     # (B, nh, L)

    # Row‑wise softmax (Triton implementation)
    # Collapse batch & head → 2‑D view, softmax, then reshape back.
    scores_flat = scores.view(bs * nh, kv_len)                       # (B*nh, L)
    attn_flat   = _triton_softmax(scores_flat)                       # (B*nh, L)
    attn = attn_flat.view(bs, nh, kv_len)                           # (B, nh, L)

    # --------------------------------------------------------------
    # 8️⃣  Weighted sum over the latent KV (produces (B, nh, dkv))
    # --------------------------------------------------------------
    M = torch.matmul(attn, kv_latent)                               # (B, nh, dkv)

    # --------------------------------------------------------------
    # 9️⃣  Up‑project the aggregated latent vector into the value space
    # --------------------------------------------------------------
    # M : (B, nh, dkv)   wV_T : (nh, dkv, dv)
    y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)                  # (B, nh, dv)

    # --------------------------------------------------------------
    # 🔟  Final linear projection back to model dimension
    # --------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, -1)                             # (B, nh*dv)
    y = F.linear(y_head_flat, wO)                                    # (B, dim)
    output = y.unsqueeze(1)                                          # (B, 1, dim)

    # --------------------------------------------------------------
    # Return the output and the (now‑updated) KV cache tensor
    # --------------------------------------------------------------
    return output, kv_cache.data