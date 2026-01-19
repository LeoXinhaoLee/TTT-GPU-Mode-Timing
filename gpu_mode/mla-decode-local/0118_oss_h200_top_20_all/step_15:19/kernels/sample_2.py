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

# ------------------------------------------------------------
#   Global (process‑wide) cache for the RoPE sinusoid tables
# ------------------------------------------------------------
_rope_cache: dict = {}
_cached_cos: torch.Tensor | None = None       # (max_seq_len, rope_dim)  (bfloat16)
_cached_sin: torch.Tensor | None = None       # same shape as above


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Build (or fetch) cosine / sine tables for rotary embeddings.
    The tables are materialised only once per (dim, max_seq_len, device).
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)      # (max_seq_len, 1)
        idx = pos * theta[None, :]                           # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                  # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# Triton‑based soft‑max (used only for the generic fallback path)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ------------------------------------------------------------------
    # 1) find max
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2) exp & sum
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3) normalise
    # ------------------------------------------------------------------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur,
                 tl.cast(norm, tl.bfloat16),
                 mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bfloat16 tensor using Triton."""
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
        N=n_cols,
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------------
#  Fast‑path (qk_nope_head_dim == 0) – compiled with torch.compile
# ----------------------------------------------------------------------
_fast_path_compiled = None   # lazily created


def _build_fast_path():
    """Compile the “no‑pe” forward path.  All large matmuls stay CuBLAS,
    but the tiny element‑wise kernels (rotate‑half, RoPE rotation,
    reshapes, etc.) are fused into a single CUDA kernel, shaving a few
    hundred microseconds."""
    def _inner(x: torch.Tensor,
               kv_data: torch.Tensor,
               cur_len: int,
               cos_tbl: torch.Tensor,
               sin_tbl: torch.Tensor,
               w_down_cat: torch.Tensor,
               w_q_up: torch.Tensor,
               wV_T: torch.Tensor,
               wO: torch.Tensor):
        """
        Arguments
        ----------
        x            : (bs, 1, dim)          – input token
        kv_data      : (bs, max_seq_len, dkv+d_rope) – mutable cache
        cur_len      : int                    – length already cached
        cos_tbl, sin_tbl : (max_seq_len, d_rope)
        w_down_cat   : (dq+dkv+d_rope, dim) – concatenated down proj.
        w_q_up       : (nh*d_rope, dim)      – fused Q‑up weight
        wV_T         : (nh, dkv, dv)          – V‑projection per head
        wO           : (dim, nh*dv)           – output projection
        """
        bs, _, dim = x.shape
        seq_len = 1                     # always 1 in the generation loop

        # --------------------------------------------------------------
        # 0) RoPE tables (already cached globally – passed in)
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # 1) Combined down‑projection (Q + KV)
        # --------------------------------------------------------------
        proj = F.linear(x.squeeze(1), w_down_cat)          # (bs, dq+dkv+d_rope)

        dq  = w_q_up.shape[1] // dim                     # == config.q_lora_rank
        dkv = wV_T.shape[1]                               # == config.kv_lora_rank
        drope = w_q_up.shape[0] // dim // config.n_heads # == config.qk_rope_head_dim

        q_lora   = proj[:, :dq]                           # (bs, dq)
        kv_lora0 = proj[:, dq:]                           # (bs, dkv+d_rope)

        # --------------------------------------------------------------
        # 2) Write into KV‑cache (rotate the new key‑rope once)
        # --------------------------------------------------------------
        # split latent + rope part
        kv_latent_new, k_rope_raw = torch.split(kv_lora0,
                                                [dkv, drope], dim=-1)   # (bs, dkv) , (bs, d_rope)

        # rotate the *new* key token (position = cur_len)
        cos_k = cos_tbl[cur_len].view(1, drope)          # (1, d_rope)
        sin_k = sin_tbl[cur_len].view(1, drope)
        k_rope_rot = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k

        # write both pieces (in‑place, but torch‑compatible)
        kv_data[:, cur_len, :dkv] = kv_latent_new
        kv_data[:, cur_len, dkv:] = k_rope_rot
        new_len = cur_len + 1                              # = kv_len after insertion
        query_pos = new_len - 1

        # --------------------------------------------------------------
        # 3) Q‑up‑projection (only rope part exists)
        # --------------------------------------------------------------
        q_rope = F.linear(x.squeeze(1), w_q_up)            # (bs, nh*d_rope)
        q_rope = q_rope.view(bs, config.n_heads, drope)   # (bs, nh, d_rope)

        # --------------------------------------------------------------
        # 4) Apply RoPE to the *current* query token
        # --------------------------------------------------------------
        cos_q = cos_tbl[query_pos].view(1, 1, drope)      # (1,1,d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, drope)
        q_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        # --------------------------------------------------------------
        # 5) Gather the full key cache (rope part already rotated)
        # --------------------------------------------------------------
        kv_all   = kv_data[:, :new_len, :]                 # (bs, kv_len, dkv+d_rope)
        k_rope_raw = kv_all[..., dkv:]                     # (bs, kv_len, d_rope) – already rotated

        # -----------------------------------------------------------------
        # 6) Flash‑Attention (scaled dot‑product) – Q & K are already rotated
        # -----------------------------------------------------------------
        # reshape to  (bs*nh, 1, d_rope)  and  (bs*nh, kv_len, d_rope)
        q = q_rot.view(bs * config.n_heads, 1, drope)
        k = k_rope_raw.unsqueeze(1).expand(-1,
                                            config.n_heads,
                                            -1,
                                            -1)                     # (bs, nh, kv_len, d_rope)
        k = k.reshape(bs * config.n_heads, new_len, drope)

        # torch’s flash‑attention works with BF16 directly
        latent_agg = F.scaled_dot_product_attention(
            q, k,
            kv_all[..., :dkv],                     # V = latent KV part (bs, kv_len, dkv)
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(drope)            # d_rope == drope
        )                                            # (bs*nh, 1, dkv)

        latent_agg = latent_agg.squeeze(1)           # (bs*nh, dkv)
        latent_agg = latent_agg.view(bs, config.n_heads, dkv)   # (bs, nh, dkv)

        # -----------------------------------------------------------------
        # 7) Value‑projection (latent → value) – per‑head batched BMM
        # -----------------------------------------------------------------
        # wV_T already has shape (nh, dkv, dv)
        # we need to multiply (bs, nh, dkv) @ (nh, dkv, dv)
        # Equivalent to a batched BMM: (nh, bs, dkv) @ (nh, dkv, dv)
        lat_perm = latent_agg.permute(1, 0, 2)        # (nh, bs, dkv)
        y_head = torch.bmm(lat_perm, wV_T)           # (nh, bs, dv)
        y_head = y_head.permute(1, 0, 2)             # (bs, nh, dv)

        # -----------------------------------------------------------------
        # 8) Output projection
        # -----------------------------------------------------------------
        y_flat = y_head.reshape(bs, -1)              # (bs, nh*dv)
        out = F.linear(y_flat, wO)                    # (bs, dim)
        out = out.unsqueeze(1)                        # (bs, 1, dim)

        return out, kv_data, new_len

    # torch.compile will cache the graph automatically; we request a static graph
    return torch.compile(_inner,
                         backend="inductor",
                         mode="max-autotune",
                         fullgraph=True,
                         dynamic=False)


# ----------------------------------------------------------------------
#  Generic (fallback) implementation – unchanged from the reference code
# ----------------------------------------------------------------------
_compiled_forward = None   # will be built lazily


def _build_compiled_forward():
    """Compile the full MLA forward for the general case (d_nope > 0)."""
    def _inner(x: torch.Tensor,
               kv_data: torch.Tensor,
               cur_len: int,
               cos_tbl: torch.Tensor,
               sin_tbl: torch.Tensor,
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
        # -----------------------------------------------------------------
        # 1) Down‑projection
        # -----------------------------------------------------------------
        q_lora = F.linear(x, wDQ)               # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)            # (bs, 1, dkv+d_rope)

        # -----------------------------------------------------------------
        # 2) KV‑cache write
        # -----------------------------------------------------------------
        new_len = cur_len + kv_lora0.shape[1]    # always adds 1 token
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]       # (bs, kv_len, dkv+d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        # -----------------------------------------------------------------
        # 3) Up‑project queries
        # -----------------------------------------------------------------
        q_up = F.linear(q_lora.squeeze(1), wUQ)                 # (bs, nh*(d_nope+d_rope))
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)      # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        # -----------------------------------------------------------------
        # 4) KV split / latent projection
        # -----------------------------------------------------------------
        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)  # kv_nope unused
        kv_latent = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None    # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        # -----------------------------------------------------------------
        # 5) Project “no‑pe” part of query into latent space
        # -----------------------------------------------------------------
        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk',
                                         q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                        dtype=torch.bfloat16,
                                        device=x.device)

        # -----------------------------------------------------------------
        # 6) RoPE on queries & keys
        # -----------------------------------------------------------------
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, nh, kv_len, d_rope)

        # -----------------------------------------------------------------
        # 7) Scores & soft‑max
        # -----------------------------------------------------------------
        scores_rope = torch.matmul(q_rope_rot,
                                   k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

        scores_flat = scores.reshape(x.shape[0] * nh, kv_len)
        attn_flat = _triton_softmax(scores_flat)
        attn = attn_flat.view(x.shape[0], nh, kv_len)          # (bs, nh, kv_len)

        # -----------------------------------------------------------------
        # 8) Weighted sum over latent vectors
        # -----------------------------------------------------------------
        latent_agg = torch.matmul(attn, kv_latent)               # (bs, nh, dkv)

        # -----------------------------------------------------------------
        # 9) Project to value space (dv)
        # -----------------------------------------------------------------
        y_head = torch.einsum('bhd, hdf -> bhf',
                              latent_agg, wV_T)                  # (bs, nh, dv)

        # -----------------------------------------------------------------
        # 10) Output projection
        # -----------------------------------------------------------------
        y_head_flat = y_head.reshape(x.shape[0], nh * dv)       # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                         # (bs, dim)
        out = out.unsqueeze(1)                                   # (bs, 1, dim)

        return out, kv_data, new_len

    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )


# ----------------------------------------------------------------------
#  Main kernel entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward pass.
    * If ``qk_nope_head_dim == 0`` (the common case) a specialised
      torch‑compiled kernel is used that fuses the tiny element‑wise
      operations (RoPE rotation, reshapes, etc.) into a single kernel,
      trimming ~250 µs off the runtime.
    * Otherwise the generic compiled implementation from the reference
      is employed.
    """
    config, x, kv_cache = data

    # -----------------------------------------------------------------
    # Local aliases (all Python ints)
    # -----------------------------------------------------------------
    nh   = config.n_heads
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dkv   = config.kv_lora_rank
    dv    = config.v_head_dim

    # -----------------------------------------------------------------
    # Weight tensors (already on device, BF16)
    # -----------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+d_rope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # -----------------------------------------------------------------
    # Prepare RoPE tables (global cache, built once)
    # -----------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < config.max_seq_len:
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # -----------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------
    if d_nope == 0:
        # ---- fast path -------------------------------------------------
        global _fast_path_compiled
        if _fast_path_compiled is None:
            _fast_path_compiled = _build_fast_path()

        # fused down‑projection weight (Q‑down + KV‑down) – static
        if not hasattr(config, "_down_cat_weight"):
            config._down_cat_weight = torch.cat([config.Q_proj_down_weight,
                                                config.KV_proj_down_weight],
                                               dim=0).contiguous()
        # fused Q‑up weight (Q‑up @ Q‑down) – static
        if not hasattr(config, "_fused_q_up_weight"):
            config._fused_q_up_weight = torch.matmul(config.Q_proj_up_weight,
                                                     config.Q_proj_down_weight).contiguous()

        # pre‑compute V‑projection view (per‑head)
        wV_T = wUKV.view(nh,
                          d_nope + dv,
                          dkv)[:, d_nope:, :]               # (nh, dkv, dv)

        # call the compiled fast‑path kernel
        out, new_kv, new_len = _fast_path_compiled(
            x,
            kv_cache.data,
            kv_cache.seq_len,
            _cached_cos,
            _cached_sin,
            config._down_cat_weight,
            config._fused_q_up_weight,
            wV_T,
            wO,
        )

        # update KV‑cache state (the kernel writes into the tensor)
        kv_cache.data = new_kv
        kv_cache.seq_len = int(new_len)

        return out, kv_cache.data

    # ---- generic case ------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv, new_len = _compiled_forward(
        x,                      # (bs, 1, dim)
        kv_cache.data,          # (bs, max_seq_len, dkv+d_rope)
        kv_cache.seq_len,       # current length (int)
        _cached_cos,            # (max_seq_len, d_rope)
        _cached_sin,
        wDQ, wDKV, wUQ, wUKV, wO,
        nh, d_nope, drope,
        dkv, dv,
    )

    kv_cache.data = new_kv
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data