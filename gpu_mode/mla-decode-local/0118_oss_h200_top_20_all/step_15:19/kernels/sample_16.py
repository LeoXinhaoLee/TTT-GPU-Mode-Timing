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
#  Small helper utilities
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ----------------------------------------------------------------------
#  RoPE caching utilities
# ----------------------------------------------------------------------
_rope_cache: dict = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cosine / sine tables for rotary positional embeddings."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                     # (max_seq_len,1)
        idx = pos * theta[None, :]                                        # (max_seq_len,half)
        idx = torch.cat([idx, idx], dim=-1)                               # (max_seq_len,dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
#  Down‑projection weight cache (combined Q‑down + KV‑down)
# ----------------------------------------------------------------------
_combined_down_cache = {}

def _get_combined_down_weight(wDQ: torch.Tensor, wDKV: torch.Tensor) -> torch.Tensor:
    """Cache the concatenated down‑projection matrix."""
    key = (id(wDQ), id(wDKV))
    if key not in _combined_down_cache:
        _combined_down_cache[key] = torch.cat([wDQ, wDKV], dim=0)
    return _combined_down_cache[key]


# ----------------------------------------------------------------------
#  Softmax kernel (used by the general‑case compiled forward)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,
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
        val = tl.load(out_ptr + row_off_out + cur,
                      mask=mask,
                      other=0.0)
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


# ----------------------------------------------------------------------
#  Compiled forward for the *general* case (qk_nope_head_dim > 0)
# ----------------------------------------------------------------------
_compiled_forward = None          # will hold the torch.compile version
_cached_cos = None                # cached RoPE cosine table (global)
_cached_sin = None                # cached RoPE sine table   (global)

def _build_compiled_forward():
    """Build a torch‑compiled version of the full forward (used when d_nope>0)."""
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
        # -------------------------------------------------------------
        # 1) Down‑projection
        # -------------------------------------------------------------
        q_lora   = F.linear(x, wDQ)          # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)         # (bs, 1, dkv+d_rope)

        # -------------------------------------------------------------
        # 2) KV‑cache write
        # -------------------------------------------------------------
        new_len = cur_len + kv_lora0.shape[1]          # always adds exactly one token
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]              # (bs, kv_len, dkv+d_rope)
        kv_len  = new_len
        query_pos = kv_len - 1

        # -------------------------------------------------------------
        # 3) Up‑project queries (general case)
        # -------------------------------------------------------------
        q_up = F.linear(q_lora.squeeze(1), wUQ)                     # (bs, nh*(d_nope+d_rope))
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)          # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        # -------------------------------------------------------------
        #    KV split / up‑project
        # -------------------------------------------------------------
        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)   # kv_nope unused later
        kv_latent = kv_lora[..., :dkv]                                     # (bs, kv_len, dkv)

        # -------------------------------------------------------------
        #    Prepare weight slices for the latent → value projection
        # -------------------------------------------------------------
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        # -------------------------------------------------------------
        #    Project query‑nope into latent space
        # -------------------------------------------------------------
        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk',
                                         q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                         dtype=torch.bfloat16,
                                         device=x.device)

        # -------------------------------------------------------------
        #    RoPE on queries
        # -------------------------------------------------------------
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)   # (1,1,d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        # -------------------------------------------------------------
        #    RoPE on keys (shared across heads)
        # -------------------------------------------------------------
        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)

        # -------------------------------------------------------------
        #    Scores (rope part + nope part)
        # -------------------------------------------------------------
        scores_rope = torch.matmul(q_rope_rot,
                                   k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

        # -------------------------------------------------------------
        #    Softmax (row‑wise, Triton)
        # -------------------------------------------------------------
        scores_flat = scores.reshape(x.shape[0] * nh, kv_len)
        attn_flat = _triton_softmax(scores_flat)
        attn = attn_flat.view(x.shape[0], nh, kv_len)          # (bs, nh, kv_len)

        # -------------------------------------------------------------
        #    Weighted sum over latent vectors
        # -------------------------------------------------------------
        latent_agg = torch.matmul(attn, kv_latent)               # (bs, nh, dkv)

        # -------------------------------------------------------------
        #    Project to value space
        # -------------------------------------------------------------
        y_head = torch.einsum('bhd, hdf -> bhf',
                             latent_agg, wV_T)                  # (bs, nh, dv)

        # -------------------------------------------------------------
        #    Output projection
        # -------------------------------------------------------------
        y_head_flat = y_head.reshape(x.shape[0], nh * dv)       # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                         # (bs, dim)
        out = out.unsqueeze(1)                                   # (bs, 1, dim)

        return out, kv_data, new_len
    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False   # only cur_len varies at runtime
    )


# ----------------------------------------------------------------------
#  Compiled fast‑path for the “no‑PE” case (qk_nope_head_dim == 0)
# ----------------------------------------------------------------------
_compiled_no_pe = None

def _build_compiled_no_pe():
    """Returns a torch‑compiled function for the d_nope==0 fast‑path."""
    def _inner(x: torch.Tensor,
               kv_data: torch.Tensor,
               cur_len: int,
               cos_tbl: torch.Tensor,
               sin_tbl: torch.Tensor,
               w_comb: torch.Tensor,
               wUQ: torch.Tensor,
               wUKV: torch.Tensor,
               wO: torch.Tensor,
               nh: int,
               d_rope: int,
               dkv: int,
               dv: int):
        # -------------------------------------------------------------
        # 1) Combined down‑projection (Q + KV)
        # -------------------------------------------------------------
        combined = F.linear(x.squeeze(1), w_comb)   # (bs, dq + dkv + d_rope)

        # split the combined tensor
        dq = wUQ.shape[1]                           # q‑lora rank
        q_lora   = combined[:, :dq]                 # (bs, dq)
        kv_lora  = combined[:, dq:]                 # (bs, dkv + d_rope)

        # -------------------------------------------------------------
        # KV split
        # -------------------------------------------------------------
        kv_latent_raw = kv_lora[:, :dkv]            # (bs, dkv)
        k_rope_raw    = kv_lora[:, dkv:]            # (bs, d_rope)

        # -------------------------------------------------------------
        # 2) Up‑project queries (only rope part exists)
        # -------------------------------------------------------------
        q_up = F.linear(q_lora, wUQ)                # (bs, nh * d_rope)
        q_up = q_up.view(x.shape[0], nh, d_rope)   # (bs, nh, d_rope)

        # -------------------------------------------------------------
        # 3) Write the new token into the KV‑cache (apply RoPE once)
        # -------------------------------------------------------------
        cos_k = cos_tbl[cur_len].view(1, d_rope)   # (1, d_rope)
        sin_k = sin_tbl[cur_len].view(1, d_rope)   # (1, d_rope)
        k_rot = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k   # (bs, d_rope)

        kv_data[:, cur_len, :dkv] = kv_latent_raw
        kv_data[:, cur_len, dkv:] = k_rot
        new_len = cur_len + 1

        # -------------------------------------------------------------
        # 4) Gather keys / values for attention
        # -------------------------------------------------------------
        kv_latent = kv_data[:, :new_len, :dkv]      # (bs, new_len, dkv)
        k_rot_all = kv_data[:, :new_len, dkv:]      # (bs, new_len, d_rope)

        # broadcast to heads
        k_exp = k_rot_all[:, None, :, :].expand(-1, nh, -1, -1)   # (bs, nh, new_len, d_rope)
        v_exp = kv_latent[:, None, :, :].expand(-1, nh, -1, -1)   # (bs, nh, new_len, dkv)

        # -------------------------------------------------------------
        # 5) RoPE for the current query token
        # -------------------------------------------------------------
        query_pos = new_len - 1
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)   # (1,1,d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rot = q_up * cos_q + _rotate_half(q_up) * sin_q   # (bs, nh, d_rope)
        q_rot = q_rot.unsqueeze(2)                         # (bs, nh, 1, d_rope)

        # -------------------------------------------------------------
        # 6) Attention (fused QK^T + softmax + weighted sum over V)
        # -------------------------------------------------------------
        scale = 1.0 / math.sqrt(d_rope)
        attn_out = F.scaled_dot_product_attention(
            q_rot, k_exp, v_exp,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )                                            # (bs, nh, 1, dkv)
        attn_out = attn_out.squeeze(2)                # (bs, nh, dkv)

        # -------------------------------------------------------------
        # 7) Project latent aggregation to value space (per‑head linear)
        # -------------------------------------------------------------
        # wUKV: (nh*dv, dkv) -> (nh, dkv, dv)
        wV = wUKV.view(nh, dv, dkv).permute(0, 2, 1)   # (nh, dkv, dv)
        y_head = torch.einsum('bhd, hdf -> bhf', attn_out, wV)   # (bs, nh, dv)

        # -------------------------------------------------------------
        # 8) Final output projection
        # -------------------------------------------------------------
        y_head_flat = y_head.reshape(x.shape[0], nh * dv)   # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                     # (bs, dim)
        out = out.unsqueeze(1)                              # (bs, 1, dim)

        return out, kv_data, new_len
    return torch.compile(_inner,
                         backend="inductor",
                         mode="max-autotune",
                         fullgraph=True,
                         dynamic=False)


# ----------------------------------------------------------------------
#  Main entry point – highly‑optimised forward
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised MLA forward pass.

    * If ``qk_nope_head_dim == 0`` we take a fast‑path that:
      - concatenates the Q‑down and KV‑down weights and performs a single
        matrix‑multiply;
      - writes the rotated key directly into the KV‑cache;
      - uses torch's fused ``scaled_dot_product_attention`` (which already
        performs the softmax);
      - projects the aggregated latent vector to the value space with a
        per‑head linear layer;
      - finally applies the output projection.

    * For the general case (``qk_nope_head_dim > 0``) we fall back to the
      existing compiled implementation.
    """
    config, x, kv_cache = data

    # ------------------- aliases -------------------
    bs = config.batch_size
    nh = config.n_heads
    d  = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # ------------------- weights (already on the proper device) -------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + d_rope) * nh, dq) -> here d_nope==0
    wUKV = config.KV_proj_up_weight           # ((d_nope + dv) * nh, dkv) -> here d_nope==0
    wO   = config.wo_weight                   # (dim, nh * dv)

    # ------------------- RoPE tables (cached) -------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape != (config.max_seq_len, d_rope):
        _cached_cos, _cached_sin = _get_rope_tables(d_rope,
                                                   config.max_seq_len,
                                                   x.device)

    # ------------------- fast‑path for d_nope == 0 -------------------
    if d_nope == 0:
        # fused down‑projection weight (Q‑down + KV‑down)
        w_comb = _get_combined_down_weight(wDQ, wDKV)   # (dq + dkv + d_rope, dim)

        global _compiled_no_pe
        if _compiled_no_pe is None:
            _compiled_no_pe = _build_compiled_no_pe()

        out, new_kv_data, new_len = _compiled_no_pe(
            x,                           # (bs, 1, dim)
            kv_cache.data,               # (bs, max_seq_len, dkv + d_rope)
            kv_cache.seq_len,            # current cached length (int)
            _cached_cos,                 # (max_seq_len, d_rope)
            _cached_sin,                 # (max_seq_len, d_rope)
            w_comb,                      # fused down‑proj weight
            wUQ,                         # Q up‑proj weight
            wUKV,                        # KV up‑proj weight (value projection)
            wO,                          # output projection
            nh, d_rope, dkv, dv
        )
        # update the KV‑cache metadata
        kv_cache.seq_len = int(new_len)
        # kv_cache.data already points to the updated tensor (in‑place write)
        return out, kv_cache.data

    # ------------------- general case (fallback) -------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,
        kv_cache.data,
        kv_cache.seq_len,
        _cached_cos,
        _cached_sin,
        wDQ, wDKV, wUQ, wUKV, wO,
        nh, d_nope, d_rope, dkv, dv
    )
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data