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
# 1️⃣  Helper: cached RoPE tables (cos / sin) – immutable after first use
# ----------------------------------------------------------------------
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Returns (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    The last dimension must be even.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                          dtype=torch.float32,
                                          device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                     # (max_seq_len, 1)
        idx = pos * theta[None, :]                                            # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                   # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos(), idx.sin())
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 2️⃣  Triton softmax (row‑wise) – faster than PyTorch for bf16
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,                # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Row‑wise softmax for a 2‑D bf16 tensor using Triton.
    x must be contiguous.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
    # pick block size (power‑of‑2) based on column dim
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
        N=n_cols,
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# 3️⃣  Main kernel – highly‑optimised MLA forward
# ----------------------------------------------------------------------
def _mlA_forward(
    x,                     # (B, 1, D)         – bf16
    kv_data,               # (B, M, Dkv+Drope) – bf16, cache (already updated)
    kv_len,                # int (current length after insert)
    wDQ, wDKV, wUQ, wUKV, wO,
    dnope, drope, dkv, dv,
    nh,
    rope_cos, rope_sin,
    rope_head_dim,
    max_seq_len,
):
    """
    All operations are plain torch kernels (no explicit Python loops).  The
    function is intended to be passed to ``torch.compile`` – the heavy
    linear / matmul / einsum calls are then executed as fused CUDA kernels.
    """
    B = x.shape[0]                # batch size
    # ------------------------------------------------------------------
    # 0️⃣  Down‑projection
    # ------------------------------------------------------------------
    q_lora = F.linear(x, wDQ)                               # (B,1,Dq)
    kv_lora_in = F.linear(x, wDKV)                          # (B,1,Dkv+Dro)

    # ------------------------------------------------------------------
    # 1️⃣  Insert into KV‑cache (in‑place – no copy)
    # ------------------------------------------------------------------
    start = kv_data.shape[1] - max_seq_len + kv_len - 1   # start index of the new token in the cache
    # NOTE: the cache tensor is already sized to (B, max_seq_len, Dkv+Dro)
    #       we simply write the new token at position ``kv_len-1``.
    kv_data[:, kv_len-1:kv_len, :] = kv_lora_in
    # ------------------------------------------------------------------
    # 2️⃣  Up‑project queries (split NoPE / RoPE)
    # ------------------------------------------------------------------
    # squeeze seq‑dim (always 1) before the up‑projection
    q_up = F.linear(q_lora.squeeze(1), wUQ)               # (B, (dnope+drope)*nh)
    q_up = q_up.view(B, nh, dnope + drope)               # (B, nh, dnope+drope)

    q_nope = q_up[..., :dnope]        # (B,nh,dnope) – may be empty
    q_rope = q_up[..., dnope:]       # (B,nh,drope)

    # ------------------------------------------------------------------
    # 3️⃣  Split KV cache into latent part + RoPE part
    # ------------------------------------------------------------------
    kv_slice = kv_data[:, :kv_len, :]                         # (B, kv_len, dkv+drope)
    kv_nope_raw = kv_slice[..., :dkv]                         # (B, kv_len, dkv)
    k_rope_raw  = kv_slice[..., dkv:]                         # (B, kv_len, drope)

    # ------------------------------------------------------------------
    # 4️⃣  Prepare the up‑projection matrices for the latent space
    # ------------------------------------------------------------------
    # wUKV : ((dnope+dv)*nh, dkv) → (nh, dnope+dv, dkv)
    wUKV_view = wUKV.view(nh, dnope + dv, dkv)
    wK = wUKV_view[:, :dnope, :]               # (nh, dnope, dkv)
    wV = wUKV_view[:, dnope:, :]               # (nh, dv, dkv)

    # ------------------------------------------------------------------
    # 5️⃣  Compute latent scores (NoPE part) – if dnope > 0
    # ------------------------------------------------------------------
    if dnope > 0:
        # wK_T : (nh, dkv, dnope)
        wK_T = wK.permute(0, 2, 1)
        # q_latent : (B, nh, dkv) = q_nope (B,nh,dnope) @ wK_T (nh, dkv, dnope)
        q_latent = torch.einsum('bhd, hkd->bhk', q_nope, wK_T)          # (B, nh, dkv)
        # scores_nope : (B, nh, kv_len) = q_latent @ kv_nope_raw^T
        scores_nope = torch.matmul(q_latent,
                                   kv_nope_raw.permute(0, 2, 1))      # (B, nh, kv_len)
    else:
        scores_nope = torch.zeros(B, nh, kv_len,
                                 dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------------
    # 6️⃣  RoPE for queries
    # ------------------------------------------------------------------
    if drope > 0:
        # position of the current query (absolute)
        q_pos = kv_len - 1
        cos_q = rope_cos[q_pos].view(1, 1, drope)      # (1,1,drope)
        sin_q = rope_sin[q_pos].view(1, 1, drope)      # (1,1,drope)
        q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q
        # --------------------------------------------------------------
        # RoPE for keys (broadcast over heads)
        # --------------------------------------------------------------
        cos_k = rope_cos[:kv_len].unsqueeze(0)         # (1, kv_len, drope)
        sin_k = rope_sin[:kv_len].unsqueeze(0)         # (1, kv_len, drope)
        k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k
        # scores_rope : (B, nh, kv_len) via einsum (head‑wise dot‑product)
        scores_rope = torch.einsum('bhd,btd->bht', q_rope, k_rope)
    else:
        scores_rope = torch.zeros_like(scores_nope)

    # ------------------------------------------------------------------
    # 7️⃣  Combine the two score components
    # ------------------------------------------------------------------
    scores = scores_nope + scores_rope                       # (B, nh, kv_len)

    # ------------------------------------------------------------------
    # 8️⃣  Scale and soft‑max (row‑wise Triton implementation)
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(dnope + drope)
    scores = scores * scale
    attn = _triton_softmax(scores.view(B * nh, kv_len)).view(B, nh, kv_len)   # (B, nh, kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Aggregate the latent key vectors (Z = Σ attn * kv_nope_raw)
    # ------------------------------------------------------------------
    # kv_T : (B, dkv, kv_len)
    kv_T = kv_nope_raw.permute(0, 2, 1)
    # attn_T : (B, kv_len, nh)
    attn_T = attn.permute(0, 2, 1)
    # Z_T : (B, dkv, nh)  = kv_T @ attn_T
    Z_T = torch.bmm(kv_T, attn_T)                 # (B, dkv, nh)
    # Z : (B, nh, dkv)
    Z = Z_T.permute(0, 2, 1)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent vectors to the value space (wV)
    # ------------------------------------------------------------------
    # wV_T : (nh, dkv, dv)
    wV_T = wV.permute(0, 2, 1)
    y_head = torch.einsum('bhd,hdv->bhv', Z, wV_T)   # (B, nh, dv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣  Final linear projection back to model dimension
    # ------------------------------------------------------------------
    y_head_flat = y_head.reshape(B, nh * dv)          # (B, nh*dv)
    output = F.linear(y_head_flat, wO)                # (B, D)
    output = output.unsqueeze(1)                      # (B, 1, D)
    return output

# ----------------------------------------------------------------------
# 2️⃣  Wrapper that updates the cache and calls the compiled forward
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor        # shape (batch, 1, dim), bf16
    kv_cache.data : torch.Tensor   # the updated cache tensor
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack static configuration values
    # ------------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len                # always 1 in this model
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len
    # ------------------------------------------------------------------
    # Weights (already on CUDA, bfloat16)
    # ------------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight               # (dq, dim)
    wDKV = config.KV_proj_down_weight              # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight                 # ((dnope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight                # ((dnope+dv)*nh, dkv)
    wO   = config.wo_weight                       # (dim, nh*dv)

    # ------------------------------------------------------------------
    # Retrieve / create RoPE tables (cached globally)
    # ------------------------------------------------------------------
    rope_cos, rope_sin = _get_rope_tables(drope, msl, x.device)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Compile the heavy forward function only once (first call)
    # ------------------------------------------------------------------
    global _compiled_mla_forward
    if '_compiled_mla_forward' not in globals():
        # torch.compile will fuse the large matmuls/einsums together.
        _compiled_mla_forward = torch.compile(
            _mlA_forward,
            mode='max-autotune',
            fullgraph=True,
        )

    # ------------------------------------------------------------------
    #  KV‑cache handling – update in‑place and get new length
    # ------------------------------------------------------------------
    # The cache tensor has shape (B, max_seq_len, dkv+drope) and stores the
    # low‑rank KV vectors.  ``seq_len`` is always 1, therefore we simply write
    # the new token at position ``kv_cache.seq_len``.
    cur_len = kv_cache.seq_len
    new_len = cur_len + sl
    kv_cache.data[:, cur_len:new_len, :] = F.linear(x, wDKV)   # (B,1,dkv+drope)
    kv_cache.seq_len = new_len
    kv_len = new_len                                           # absolute length after insert

    # ------------------------------------------------------------------
    #  Call the compiled forward pass
    # ------------------------------------------------------------------
    out = _compiled_mla_forward(
        x, kv_cache.data, kv_len,
        wDQ, wDKV, wUQ, wUKV, wO,
        dnope, drope, dkv, dv,
        nh,
        rope_cos, rope_sin,
        drope,
        msl,
    )
    # ``out`` already has shape (B,1,D) and dtype bfloat16
    return out, kv_cache.data