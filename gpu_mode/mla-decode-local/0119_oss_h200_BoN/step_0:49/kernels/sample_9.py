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
#  Utility helpers (RoPE tables, rotate‑half)
# ----------------------------------------------------------------------
_rope_cache: dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Swaps the two halves of the last dimension and negates the second half.
    Same as the `rotate_half` method of the reference `RoPE` module.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Cached cosine / sine tables used by RoPE.
    Returned shape: (max_seq_len, dim) , (max_seq_len, dim)   (both bf16)
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta = 10000^{ -i/half }
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                     # (max_seq_len,1)
        idx = pos * theta[None, :]                     # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)           # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Triton soft‑max (row‑wise, bf16) – kept because it is often faster
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)                         # one program per row
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ------------------------------------------------------------------
    # 1️⃣ max (for numerical stability)
    # ------------------------------------------------------------------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ------------------------------------------------------------------
    # 2️⃣ exp & sum
    # ------------------------------------------------------------------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ------------------------------------------------------------------
    # 3️⃣ normalize
    # ------------------------------------------------------------------
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Row‑wise softmax for a 2‑D bf16 tensor using Triton.
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
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
#  Main kernel – fully‑torch (the heavy work is already in cuBLAS/FlashAttn)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns a tuple (output, updated_kv_cache_tensor)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # 1️⃣ unpack configuration & weights (all on CUDA, bf16)
    # --------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len                 # always 1 in the provided benchmarks
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # 2️⃣ Down‑project (fast PyTorch linear – cuBLAS under the hood)
    # --------------------------------------------------------------
    # x : (bs, sl, dim)  ; sl == 1
    q_lora = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)             # (bs, sl, dkv + d_rope)

    # --------------------------------------------------------------
    # 3️⃣ KV‑cache update (in‑place, Python‑side bookkeeping)
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)    # kv_lora : (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                         # absolute position of the current token

    # --------------------------------------------------------------
    # 4️⃣ Up‑project queries (no‑PE + RoPE split)
    # --------------------------------------------------------------
    # squeeze the (bs,1,dq) → (bs,dq)
    q_up = F.linear(q_lora.squeeze(1), wUQ)       # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)    # (bs, nh, d_total)
    q_nope = q_up[..., :d_nope]                  # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                  # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 5️⃣ RoPE on queries (pure torch – no extra kernel launches)
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]                 # (d_rope,)
    sin_q = sin_table[query_pos]                 # (d_rope,)
    # rotate‑half + apply cos/sin
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 6️⃣ Split KV‑cache into latent part & RoPE part
    # --------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]            # (bs, kv_len, dkv) – keys & values used for attention
    k_rope_input = kv_lora[..., dkv:]            # (bs, kv_len, d_rope)

    # RoPE for keys (vectorised, uses broadcasting)
    cos_k = cos_table[:kv_len].unsqueeze(0)      # (1, kv_len, d_rope)
    sin_k = sin_table[:kv_len].unsqueeze(0)      # (1, kv_len, d_rope)
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k   # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 7️⃣ Extract per‑head projection matrices from the big KV‑up weight
    # --------------------------------------------------------------
    # (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                # (nh, d_nope, dkv)
    wV_T = wUKV_view[:, d_nope:, :].transpose(1, 2)   # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 8️⃣ Latent query projections (q_nope → latent space)
    # --------------------------------------------------------------
    # Einstein‑summation – effectively a batched matrix multiply per head
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 9️⃣ Score computation (latent + RoPE) + scaling
    # --------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # (bs, nh, kv_len)
    scores_nope = torch.bmm(q_nope_latent, kv_nope_input.transpose(1, 2))
    scores_rope = torch.bmm(q_rope, k_rope.transpose(1, 2))
    scores = (scores_nope + scores_rope) * scale

    # --------------------------------------------------------------
    # 🔟 Soft‑max over the KV‑dimension (use Triton for a fast row‑wise version)
    # --------------------------------------------------------------
    # Collapse batch+head → 2‑D for the kernel
    scores_flat = scores.view(bs * nh, kv_len)          # (B*H, L)
    attn_flat = _triton_softmax(scores_flat)           # (B*H, L)  bf16
    attn = attn_flat.view(bs, nh, kv_len)               # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 1️⃣1️⃣ Compute the (latent) weighted sum of keys → M
    # --------------------------------------------------------------
    M = torch.bmm(attn, kv_nope_input)                 # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 1️⃣2️⃣ Project M to per‑head values via wV_T
    # --------------------------------------------------------------
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)     # (bs, nh, dv)

    # --------------------------------------------------------------
    # 1️⃣3️⃣ Final linear projection back to model space
    # --------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                    # (bs, nh*dv)
    y = y.unsqueeze(1)                                 # (bs, 1, nh*dv)
    output = F.linear(y, wO)                           # (bs, 1, dim)

    # --------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # --------------------------------------------------------------
    return output, kv_cache.data