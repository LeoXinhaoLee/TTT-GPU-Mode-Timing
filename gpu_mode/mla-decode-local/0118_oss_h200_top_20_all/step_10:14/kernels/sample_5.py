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

# ---------------------------------------------------------
#  Helper utilities (same as the reference code)
# ---------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

# ---------------------------------------------------------
#  RoPE cache – lazily constructed once per (dim, max_seq_len, device)
# ---------------------------------------------------------
_rope_cache: dict = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Returns (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    The tables are cached globally – the first call does the work,
    subsequent calls are O(1).
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta has shape (half,) – compute in float32 then cast
        theta = (10000.0 ** (-torch.arange(half,
                                          dtype=torch.float32,
                                          device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)  # (max_seq_len, 1)
        idx = pos * theta[None, :]                       # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)              # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ---------------------------------------------------------
#  Triton row‑wise soft‑max (unchanged – already highly tuned)
# ---------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,              # number of columns
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
        val = tl.load(in_ptr + row_off_in + cur,
                      mask=mask,
                      other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # -------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur,
                      mask=mask,
                      other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur,
                 tl.cast(exp_val, tl.bfloat16),
                 mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # -------- normalize ----------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur,
                 tl.cast(norm, tl.bfloat16),
                 mask=mask)

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

# ---------------------------------------------------------
#  Compiled forward – we compile once and reuse
# ---------------------------------------------------------
_compiled_forward = None          # will hold the torch.compile version
_cached_cos = None                # cached RoPE cosine table (global)
_cached_sin = None                # cached RoPE sine table   (global)

def _build_compiled_forward():
    """
    Build (and cache) a torch‑compiled version of the whole forward pass.
    The function is written exactly like the reference implementation,
    but written in such a way that `torch.compile` can fuse the kernels.
    """
    def _inner(x: torch.Tensor,
               kv_data: torch.Tensor,
               cur_len: int,
               cos_tbl: torch.Tensor,
               sin_tbl: torch.Tensor,
               # model hyper‑parameters – all of them are plain python ints / tensors
               wDQ: torch.Tensor,
               wDKV: torch.Tensor,
               wUQ: torch.Tensor,
               wUKV: torch.Tensor,
               wO: torch.Tensor,
               nh: int,
               d_nope: int,
               d_rope: int,
               dkv: int,
               dv: int):
        """
        Returns
        -------
        out : torch.Tensor            # (bs, 1, dim) – bf16
        kv_data_out : torch.Tensor   # same storage as input, but with the new
                                     # token appended
        new_len : int                # updated sequence length inside the cache
        """
        # -----------------------------------------------------------------
        # 1) Down‑projection
        # -----------------------------------------------------------------
        # x : (bs, 1, dim)   – always seq_len == 1 here
        q_lora   = F.linear(x, wDQ)          # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)         # (bs, 1, dkv + d_rope)

        # -----------------------------------------------------------------
        # 2) KV‑cache write (in‑place on the supplied buffer)
        # -----------------------------------------------------------------
        new_len = cur_len + kv_lora0.shape[1]          # adds exactly one token
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]              # (bs, kv_len, dkv+d_rope)
        kv_len  = new_len
        query_pos = kv_len - 1

        # -----------------------------------------------------------------
        # 3) Up‑project Queries (fast‑path when d_nope == 0)
        # -----------------------------------------------------------------
        if d_nope == 0:
            # ---- q_up (only the RoPE part) ---------------------------------
            # q_lora.shape -> (bs, 1, dq) → squeeze → (bs, dq)
            q_up = F.linear(q_lora.squeeze(1), wUQ)               # (bs, nh * d_rope)
            q_up = q_up.view(x.shape[0], nh, d_rope)             # (bs, nh, d_rope)

            # ---- split KV --------------------------------------------------
            kv_latent = kv_lora[..., :dkv]                         # (bs, kv_len, dkv)
            k_rope    = kv_lora[..., dkv:]                         # (bs, kv_len, d_rope)

            # ---- RoPE on queries -------------------------------------------
            cos_q = cos_tbl[query_pos].view(1, 1, d_rope)        # (1,1,d_rope)
            sin_q = sin_tbl[query_pos].view(1, 1, d_rope)

            q_rot = q_up * cos_q + _rotate_half(q_up) * sin_q      # (bs, nh, d_rope)

            # ---- RoPE on keys (broadcast over heads) -----------------------
            cos_k = cos_tbl[:kv_len].view(1, kv_len, d_rope)      # (1,kv_len,d_rope)
            sin_k = sin_tbl[:kv_len].view(1, kv_len, d_rope)

            k_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)
            k_rot = k_rot[:, None, :, :].expand(-1, nh, -1, -1)     # (bs, nh, kv_len, d_rope)

            # ---- Values (latent vectors) ------------------------------------
            v = kv_latent[:, None, :, :].expand(-1, nh, -1, -1)    # (bs, nh, kv_len, dkv)

            # ---- Flash‑Attention (scaled‑dot‑product) ----------------------
            scale = 1.0 / math.sqrt(d_rope)
            latent_agg = F.scaled_dot_product_attention(
                q_rot, k_rot, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=scale
            )                                                       # (bs, nh, 1, dkv)
            latent_agg = latent_agg.squeeze(2)                     # (bs, nh, dkv)

            # ---- Project latent aggregation to value space ------------------
            # wUKV shape: ((d_nope + dv)*nh, dkv)  → here d_nope == 0
            wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1)        # (nh, dkv, dv)
            y_head = torch.einsum('bhd, hdf -> bhf', latent_agg, wV_T)  # (bs, nh, dv)

            # ---- Output projection -----------------------------------------
            y_head_flat = y_head.reshape(x.shape[0], nh * dv)    # (bs, nh*dv)
            out = F.linear(y_head_flat, wO)                       # (bs, dim)
            out = out.unsqueeze(1)                                 # (bs, 1, dim)

        else:
            # -----------------------------------------------------------------
            # General path – identical to the reference implementation
            # -----------------------------------------------------------------
            # ---- Up‑project queries (NoPE + RoPE) -------------------------
            q_up = F.linear(q_lora.squeeze(1), wUQ)                     # (bs, nh*(d_nope+d_rope))
            q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)          # (bs, nh, d_nope+d_rope)
            q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

            # ---- Split KV ------------------------------------------------
            kv_latent = kv_lora[..., :dkv]          # (bs, kv_len, dkv)
            k_rope    = kv_lora[..., dkv:]          # (bs, kv_len, d_rope)

            # ---- Prepare up‑projection weight for KV ---------------------
            wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
            wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
            wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

            # ---- Project NoPE part of queries into latent space ----------
            if d_nope > 0:
                q_nope_latent = torch.einsum('bhd, hdk -> bhk',
                                             q_nope, wK)               # (bs, nh, dkv)
            else:
                q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                             dtype=torch.bfloat16,
                                             device=x.device)

            # ---- RoPE for queries ----------------------------------------
            cos_q = cos_tbl[query_pos].view(1, 1, d_rope)
            sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
            q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

            # ---- RoPE for keys (shared across heads) --------------------
            cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
            sin_k = sin_tbl[:kv_len].unsqueeze(0)
            k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)

            # ---- Scores (rope part + nope part) -------------------------
            scores_rope = torch.matmul(q_rope_rot,
                                       k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
            scores_nope = torch.matmul(q_nope_latent,
                                       kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
            scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

            # ---- Softmax (row‑wise, Triton) ------------------------------
            scores_flat = scores.reshape(x.shape[0] * nh, kv_len)
            attn_flat = _triton_softmax(scores_flat)
            attn = attn_flat.view(x.shape[0], nh, kv_len)          # (bs, nh, kv_len)

            # ---- Weighted sum over latent vectors -------------------------
            latent_agg = torch.matmul(attn, kv_latent)               # (bs, nh, dkv)

            # ---- Project to value space ----------------------------------
            y_head = torch.einsum('bhd, hdf -> bhf',
                                 latent_agg, wV_T)                  # (bs, nh, dv)

            # ---- Output projection ---------------------------------------
            y_head_flat = y_head.reshape(x.shape[0], nh * dv)       # (bs, nh*dv)
            out = F.linear(y_head_flat, wO)                         # (bs, dim)
            out = out.unsqueeze(1)                                   # (bs, 1, dim)

        # -----------------------------------------------------------------
        # Return the output together with the updated KV cache tensor
        # -----------------------------------------------------------------
        return out, kv_data, new_len

    # The torch.compile wrapper – we ask TorchInductor to use its Triton
    # backend (the default) and to cache the compiled version.
    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False   # all shapes are known at compile‑time except `cur_len`
    )

# ---------------------------------------------------------
#  Main entry point – called from the evaluation harness
# ---------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward pass – heavily fused via torch.compile.
    Returns
    -------
    output: torch.Tensor       # (batch, seq_len=1, dim) – bfloat16
    kv_cache.data: torch.Tensor   # the cache after inserting the new token
    """
    config, x, kv_cache = data

    # -----------------------------------------------------------------
    # Local aliases (only Python integers – cheap)
    # -----------------------------------------------------------------
    bs   = config.batch_size
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim          # may be 0
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # -----------------------------------------------------------------
    # Weights – already on the correct device & dtype
    # -----------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv+d_rope, dim)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # -----------------------------------------------------------------
    # RoPE tables – cached globally; the first call builds them.
    # They are tensors, not Python objects, so they can be captured
    # by the compiled function without extra overhead.
    # -----------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape != (msl, d_rope):
        _cached_cos, _cached_sin = _get_rope_tables(d_rope, msl, x.device)

    # -----------------------------------------------------------------
    # Compile the inner function lazily (only once per Python process)
    # -----------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    # -----------------------------------------------------------------
    # Call the compiled forward – all heavy work (projections, RoPE,
    # attention, final projection) happens inside the fused graph.
    # -----------------------------------------------------------------
    out, new_kv_data, new_len = _compiled_forward(
        x,                               # (bs, 1, dim)
        kv_cache.data,                   # (bs, max_seq_len, dkv+d_rope)
        kv_cache.seq_len,                # Python int – current length
        _cached_cos,                     # (max_seq_len, d_rope)
        _cached_sin,                     # (max_seq_len, d_rope)
        wDQ, wDKV, wUQ, wUKV, wO,
        nh, d_nope, d_rope, dkv, dv
    )

    # -----------------------------------------------------------------
    # Update the KVCache instance (seq_len and data buffer)
    # -----------------------------------------------------------------
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)   # ensure a Python int

    # -----------------------------------------------------------------
    # Return output (shape: [bs, 1, dim]) and the *tensor* holding the KV cache.
    # -----------------------------------------------------------------
    return out, kv_cache.data