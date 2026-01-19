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
#  Helper utilities (RoPE tables, rotate‑half, Triton softmax)
# ----------------------------------------------------------------------
_rotate_half_cache: dict[int, torch.Tensor] = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    # Simple but fast for the small rope dimension
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

_rope_cache: dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        assert dim % 2 == 0, "RoPE dimension must be even"
        half = dim // 2
        i = torch.arange(half, dtype=torch.float32, device=device)                     # (half,)
        theta = (10000.0 ** (-i / half)).to(torch.bfloat16)                           # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device)             # (max_seq_len,)
        idx = pos[:, None] * theta[None, :]                                           # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                           # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Triton soft‑max (row‑wise, bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ------- normalize ----------
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax using the Triton kernel above."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
    if n_cols <= 32:
        BLOCK = 32
    elif n_cols <= 64:
        BLOCK = 64
    elif n_cols <= 128:
        BLOCK = 128
    else:
        BLOCK = 1 << (n_cols-1).bit_length()
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
#  Optimised MLA forward – everything fused with fast matmuls
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns (output, kv_cache_tensor) where
      - output : torch.Tensor of shape (batch, seq_len, dim)   (bf16)
      - kv_cache_tensor : the updated KV‑cache data tensor
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Unpack config (all scalars are Python ints)
    # --------------------------------------------------------------
    bs   = config.batch_size          # batch size
    sl   = config.seq_len             # always 1 in the current runtime
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # --------------------------------------------------------------
    # Weights (already on the right device and in bfloat16)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, d)
    wDKV  = config.KV_proj_down_weight         # (dkv+d_rope, d)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (d, nh*dv)

    # --------------------------------------------------------------
    # 1️⃣ Down‑projection + KV‑cache update
    # --------------------------------------------------------------
    # x : (bs, 1, d)
    q_lora   = F.linear(x, wDQ)               # (bs, 1, dq)
    kv_lora_in = F.linear(x, wDKV)            # (bs, 1, dkv + d_rope)

    # Store new KV tokens → updated cache
    kv_lora, kv_len = kv_cache(kv_lora_in)    # kv_lora : (bs, kv_len, dkv + d_rope)
    kv_len_int = int(kv_len)
    query_pos = kv_len_int - 1                 # position of *new* token

    # --------------------------------------------------------------
    # 2️⃣ Up‑project Q and split into NoPE / RoPE parts
    # --------------------------------------------------------------
    # remove singleton time dim before up‑proj
    q_up = F.linear(q_lora.squeeze(1), wUQ)    # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)   # (bs, nh, d_nope+d_rope)

    q_nope      = q_up[..., :d_nope]          # (bs, nh, d_nope)
    q_rope_raw  = q_up[..., d_nope:]          # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 3️⃣ Split KV into latent + RoPE components
    # --------------------------------------------------------------
    kv_latent = kv_lora[..., :dkv]            # (bs, kv_len, dkv)
    k_rope_raw = kv_lora[..., dkv:]           # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 4️⃣ Pre‑compute RoPE tables (cached)
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)   # (max_seq_len, d_rope)

    # ---- Q‑RoPE ---------------------------------------------------
    cos_q = cos_table[query_pos]               # (d_rope,)
    sin_q = sin_table[query_pos]               # (d_rope,)
    # broadcast to (bs, nh, d_rope)
    cos_q = cos_q.view(1, 1, d_rope)
    sin_q = sin_q.view(1, 1, d_rope)

    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, d_rope)

    # ---- K‑RoPE ---------------------------------------------------
    # slice the tables to the current cache length
    cos_k = cos_table[:kv_len_int]            # (kv_len, d_rope)
    sin_k = sin_table[:kv_len_int]            # (kv_len, d_rope)
    # broadcast over batch dimension
    cos_k = cos_k.unsqueeze(0)                # (1, kv_len, d_rope)
    sin_k = sin_k.unsqueeze(0)                # (1, kv_len, d_rope)

    k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k   # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 5️⃣ Project Q‑NoPE into the latent space (dkv)
    # --------------------------------------------------------------
    # wUKV holds ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)        # (nh, d_nope+dv, dkv)

    # only the first d_nope rows are needed for the query projection
    wK = wUKV_view[:, :d_nope, :]                      # (nh, d_nope, dkv)

    # q_nope : (bs, nh, d_nope) → q_nope_latent : (bs, nh, dkv)
    # small dense matmul; einsum is fine here
    q_nope_latent = torch.einsum('bhd,hdm->bhm', q_nope, wK)   # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 6️⃣ Compute raw attention scores (latent + RoPE)
    # --------------------------------------------------------------
    # latent part  : torch.matmul((bs,nh,dkv), (bs,dkv,kv_len)) → (bs,nh,kv_len)
    scores_nope = torch.matmul(q_nope_latent, kv_latent.transpose(-2, -1))   # (bs, nh, kv_len)

    # rope part    : torch.matmul((bs,nh,d_rope), (bs,d_rope,kv_len)) → (bs,nh,kv_len)
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))             # (bs, nh, kv_len)

    # combine + scaling
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale               # (bs, nh, kv_len)
    scores = scores.to(torch.bfloat16)                         # Triton softmax expects bf16

    # --------------------------------------------------------------
    # 7️⃣ Row‑wise softmax (Triton)
    # --------------------------------------------------------------
    B = bs * nh
    attn = _triton_softmax(scores.view(B, kv_len_int)).view(bs, nh, kv_len_int)   # (bs, nh, kv_len)

    # --------------------------------------------------------------
    # 8️⃣ Weighted sum of latent keys (M = Σ attn * KV_latent)
    # --------------------------------------------------------------
    # attn (bs, nh, kv_len) @ kv_latent (bs, kv_len, dkv) → (bs, nh, dkv)
    M = torch.matmul(attn, kv_latent)                         # (bs, nh, dkv)

    # --------------------------------------------------------------
    # 9️⃣ Project M → per‑head values (dv)
    # --------------------------------------------------------------
    # wV_T : (nh, dkv, dv)  =  (nh, dkv, dv)
    wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)         # (nh, dkv, dv)

    # M (bs, nh, dkv)  ×  wV_T (nh, dkv, dv) → (bs, nh, dv)
    y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)           # (bs, nh, dv)

    # --------------------------------------------------------------
    # 🔟 Merge heads & final linear projection
    # --------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                           # (bs, nh*dv)
    output = F.linear(y, wO)                                  # (bs, d)
    output = output.unsqueeze(1)                               # (bs, 1, d)

    # --------------------------------------------------------------
    # Return output and the *updated* KV‑cache tensor
    # --------------------------------------------------------------
    return output, kv_cache.data