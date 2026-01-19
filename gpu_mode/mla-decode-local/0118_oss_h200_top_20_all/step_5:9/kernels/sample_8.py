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
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


# ----------------------------------------------------------------------
# 1️⃣  RoPE cache (cos / sin tables) – stored lazily
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Returns (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    `dim` must be even.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta = 10000 ** (-i/half)
        theta = (10000.0 ** (-torch.arange(half,
                                            dtype=torch.float32,
                                            device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                     # (max_seq_len, 1)
        idx = pos * theta[None, :]                                          # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                 # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# 2️⃣  Triton row‑wise softmax (unchanged – already optimal)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,                # number of columns
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

    # choose a power‑of‑2 block size (capped at 1024)
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
# 3️⃣  Optimised MLA forward – custom_kernel
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward pass.
    Returns
    -------
    output : torch.Tensor       # shape (batch, seq_len, dim) – bf16
    kv_cache.data : torch.Tensor   # up‑to‑date cache (raw kv‑lora)
    """
    # ------------------------------------------------------------------
    # Unpack arguments
    # ------------------------------------------------------------------
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Local aliases (avoid repeated attribute look‑ups)
    # ------------------------------------------------------------------
    bs   = config.batch_size          # e.g. 128
    sl   = config.seq_len             # always 1 in the benchmark
    nh   = config.n_heads             # 128
    d    = config.dim                 # 7168
    dq   = config.q_lora_rank         # 1536
    dkv  = config.kv_lora_rank        # 512
    d_nope = config.qk_nope_head_dim # may be 0
    d_rope = config.qk_rope_head_dim # 64
    dv   = config.v_head_dim          # 128
    msl  = config.max_seq_len         # 8192

    # ------------------------------------------------------------------
    # Extract weight tensors (already on CUDA, bfloat16)
    # ------------------------------------------------------------------
    # Down‑proj weights (out_features, in_features)
    wDQ   = config.Q_proj_down_weight            # (dq, dim)
    wDKV  = config.KV_proj_down_weight           # (dkv+d_rope, dim)

    # Up‑proj weight for Q (out_features, in_features)
    wUQ   = config.Q_proj_up_weight              # ((d_nope+d_rope)*nh, dq)

    # KV up‑proj weight – contains both key‑ and value‑projections
    wUKV  = config.KV_proj_up_weight             # ((d_nope+dv)*nh, dkv)

    # Output projection
    wO    = config.wo_weight                     # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 0️⃣  Down‑projections
    # ------------------------------------------------------------------
    # x : (bs, sl, d) → (bs, sl, dq) and (bs, sl, dkv+d_rope)
    q_lora   = F.linear(x, wDQ)                     # (bs, sl, dq)
    kv_lora0 = F.linear(x, wDKV)                    # (bs, sl, dkv+d_rope)

    # ------------------------------------------------------------------
    # 1️⃣  KV‑cache update (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora0)            # kv_lora : (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                           # absolute position of the current query token

    # ------------------------------------------------------------------
    # 2️⃣  Up‑project queries (split into NoPE & RoPE)
    # ------------------------------------------------------------------
    # Q up‑projection (remove the sequence dimension – it is always 1)
    q_up = F.linear(q_lora.squeeze(1), wUQ)          # (bs, nh*(d_nope+d_rope))
    q_up = q_up.view(bs, nh, d_nope + d_rope)       # (bs, nh, d_nope+d_rope)

    q_nope = q_up[..., :d_nope]                     # (bs, nh, d_nope) – may be empty
    q_rope_raw = q_up[..., d_nope:]                 # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 3️⃣  RoPE for queries
    # ------------------------------------------------------------------
    if d_rope > 0:
        cos_tbl, sin_tbl = _get_rope_tables(d_rope, msl, x.device)
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)   # (1,1,d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q
    else:
        q_rope = torch.zeros_like(q_rope_raw)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV cache into latent part and rope part
    # ------------------------------------------------------------------
    kv_latent_raw = kv_lora[..., :dkv]                # (bs, kv_len, dkv)

    if d_rope > 0:
        kv_rope_raw = kv_lora[..., dkv:]                 # (bs, kv_len, d_rope)
        cos_tbl, sin_tbl = _get_rope_tables(d_rope, msl, x.device)
        cos_k = cos_tbl[:kv_len]                         # (kv_len, d_rope)
        sin_k = sin_tbl[:kv_len]                         # (kv_len, d_rope)
        cos_k = cos_k.unsqueeze(0)                       # (1, kv_len, d_rope)
        sin_k = sin_k.unsqueeze(0)                       # (1, kv_len, d_rope)
        k_rope = kv_rope_raw * cos_k + _rotate_half(kv_rope_raw) * sin_k  # (bs, kv_len, d_rope)
    else:
        # dummy tensor – never used when d_rope == 0
        k_rope = torch.empty((bs, kv_len, 0), dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------------
    # 5️⃣  Split KV‑up‑projection weights into key‑ and value‑projections
    # ------------------------------------------------------------------
    # wUKV shape : ((d_nope+dv)*nh, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)      # (nh, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                    # (nh, d_nope, dkv)
    wV = wUKV_view[:, d_nope:, :]                    # (nh, dv, dkv)

    # ------------------------------------------------------------------
    # 6️⃣  Project the query NoPE part into latent space (batched GEMM)
    # ------------------------------------------------------------------
    if d_nope > 0:
        # (nh, bs, d_nope) @ (nh, d_nope, dkv) -> (nh, bs, dkv)
        q_nope_perm = q_nope.permute(1, 0, 2)                     # (nh, bs, d_nope)
        q_latent_perm = torch.bmm(q_nope_perm, wK)                # (nh, bs, dkv)
        q_latent = q_latent_perm.permute(1, 0, 2)                 # (bs, nh, dkv)
    else:
        q_latent = torch.zeros(bs, nh, dkv, dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------------
    # 7️⃣  Compute latent‑space scores  (batch GEMM)
    # ------------------------------------------------------------------
    if d_nope > 0:
        # (bs, nh, dkv) @ (bs, dkv, kv_len) -> (bs, nh, kv_len)
        scores_latent = torch.matmul(q_latent, kv_latent_raw.transpose(1, 2))
    else:
        scores_latent = torch.zeros(bs, nh, kv_len, dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------------
    # 8️⃣  Compute RoPE scores  (small‑dim GEMM via einsum)
    # ------------------------------------------------------------------
    if d_rope > 0:
        # q_rope : (bs, nh, d_rope)
        # k_rope : (bs, kv_len, d_rope)
        scores_rope = torch.einsum('bhd,bkd->bhk', q_rope, k_rope)   # (bs, nh, kv_len)
    else:
        scores_rope = torch.zeros(bs, nh, kv_len, dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------------
    # 9️⃣  Combine scores & scale
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_latent + scores_rope) * scale

    # ------------------------------------------------------------------
    # 🔟  Row‑wise softmax (Triton implementation)
    # ------------------------------------------------------------------
    scores_flat = scores.reshape(bs * nh, kv_len)          # (bs*nh, kv_len)
    attn_flat = _triton_softmax(scores_flat)              # same shape, bf16
    attn = attn_flat.view(bs, nh, kv_len)                 # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣  Weighted sum of the latent vectors  (batch GEMM)
    # ------------------------------------------------------------------
    # Z : (bs, nh, dkv)
    Z = torch.matmul(attn, kv_latent_raw)                 # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 1️⃣2️⃣  Project aggregated latents to value space
    # ------------------------------------------------------------------
    # wV : (nh, dv, dkv) → transpose for GEMM
    wV_T = wV.permute(0, 2, 1)                            # (nh, dkv, dv)
    Z_perm = Z.permute(1, 0, 2)                            # (nh, bs, dkv)
    y_head_perm = torch.bmm(Z_perm, wV_T)                 # (nh, bs, dv)
    y_head = y_head_perm.permute(1, 0, 2)                 # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣3️⃣  Final linear projection back to model dimension
    # ------------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)             # (bs, nh*dv)
    output = F.linear(y_head_flat, wO)                    # (bs, dim)
    output = output.unsqueeze(1)                          # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the updated KV‑cache buffer
    # ------------------------------------------------------------------
    return output, kv_cache.data