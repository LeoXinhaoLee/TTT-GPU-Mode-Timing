### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config   # noqa: F401
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# Helper utilities (RoPE cache, rotation)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the last dim (used by RoPE)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_cache(drope: int,
                max_len: int,
                dtype: torch.dtype,
                device: torch.device):
    """
    Build (and cache) cosine / sine tables for Rotary Positional Embedding.
    Returns (cos, sin) of shape (max_len, drope) in BF16.
    """
    key = (drope, max_len, dtype, device)
    if not hasattr(_rope_cache, "_CACHE"):
        _rope_cache._CACHE = {}
    cache = _rope_cache._CACHE
    if key in cache:
        return cache[key]

    half = drope // 2
    theta = 10000.0 ** (-torch.arange(0, half, dtype=dtype, device=device) /
                         float(half))
    pos = torch.arange(max_len, dtype=dtype, device=device)
    theta_pos = torch.einsum("p,d->pd", pos, theta)          # (max_len, half)
    theta_pos = torch.cat([theta_pos, theta_pos], dim=-1)    # (max_len, drope)

    cos = torch.cos(theta_pos).to(torch.bfloat16)           # (max_len, drope)
    sin = torch.sin(theta_pos).to(torch.bfloat16)           # (max_len, drope)

    cache[key] = (cos, sin)
    return cos, sin


# ----------------------------------------------------------------------
# Triton‑fused row‑wise softmax (used for the attention scores)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    row_stride, col_stride,
    rows, cols,
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    col = tl.arange(0, BLOCK)
    mask = col < cols

    # Very negative sentinel for masked elements (BF16)
    other = tl.full([], -5e4, tl.bfloat16)

    val = tl.load(in_ptr + row * row_stride + col * col_stride,
                  mask=mask, other=other)
    # Cast to FP32 for stable softmax
    val_f32 = tl.cast(val, tl.float32)

    # stable softmax
    row_max = tl.max(val_f32, axis=0)
    val_f32 = val_f32 - row_max
    exp_val = tl.exp(val_f32)
    sum_exp = tl.sum(exp_val, axis=0)

    out_f32 = exp_val / sum_exp
    out = tl.cast(out_f32, tl.bfloat16)

    tl.store(out_ptr + row * row_stride + col * col_stride, out, mask=mask)


def _rowwise_softmax(x: torch.Tensor) -> torch.Tensor:
    """Fused row‑wise softmax for a 2‑D BF16 tensor."""
    rows, cols = x.shape
    BLOCK = triton.next_power_of_2(cols)

    out = torch.empty_like(x)
    _softmax_kernel[(rows,)](
        out_ptr=out,
        in_ptr=x,
        row_stride=x.stride(0),
        col_stride=x.stride(1),
        rows=rows,
        cols=cols,
        BLOCK=BLOCK,
        NUM_STAGES=4,
        num_warps=8,
    )
    return out


# ----------------------------------------------------------------------
# Core forward – torch‑compile + Triton softmax
# ----------------------------------------------------------------------
def _core_forward(
    x_flat: torch.Tensor,                     # (bs, dim)
    wDQ: torch.Tensor,                        # (dq, dim)
    wDKV: torch.Tensor,                       # (dkv+drope, dim)
    wUQ: torch.Tensor,                        # (nh*(dnope+drope), dq)
    wUKV: torch.Tensor,                       # (nh*(dnope+dv), dkv)
    wO: torch.Tensor,                         # (dim, nh*dv)
    rope_cos: torch.Tensor,                   # (max_seq_len, drope)
    rope_sin: torch.Tensor,                   # (max_seq_len, drope)
    kv_cache_data: torch.Tensor,              # (bs, max_seq_len, dkv+drope)
    kv_cache_seq_len: int,
    config: Config,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Forward for a single token (seq_len == 1).
    Returns (output, updated_kv_cache, new_seq_len).
    """
    # -------------------------------------------------
    # Config shortcuts
    # -------------------------------------------------
    bs    = config.batch_size
    nh    = config.n_heads
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv    = config.v_head_dim
    dkv   = config.kv_lora_rank
    dq    = config.q_lora_rank

    # -------------------------------------------------
    # 1️⃣ Down‑project Q and KV (low‑rank)
    # -------------------------------------------------
    q_lora  = F.linear(x_flat, wDQ)          # (bs, dq)
    kv_lora = F.linear(x_flat, wDKV)         # (bs, dkv+drope)

    # -------------------------------------------------
    # 2️⃣ Update KV‑cache (store low‑rank + RoPE part)
    # -------------------------------------------------
    seq = kv_cache_seq_len
    kv_cache_data[:, seq:seq + 1, :] = kv_lora.unsqueeze(1)
    seq += 1                      # new length
    kv_len = seq                  # = old_len + 1
    query_pos = kv_len - 1        # position of the just‑added token

    # -------------------------------------------------
    # 3️⃣ Up‑project Q and split into NoPE / RoPE
    # -------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                     # (bs, nh*(dnope+drope))
    q_up = q_up.view(bs, nh, dnope + drope)          # (bs, nh, dnope+drope)
    q_nope_raw, q_rope_raw = torch.split(
        q_up, [dnope, drope], dim=-1)                # (bs, nh, dnope), (bs, nh, drope)

    # -------------------------------------------------
    # 4️⃣ KV up‑projection, split into NoPE / RoPE, and build low‑rank KV
    # -------------------------------------------------
    kv_nope_raw, k_rope_raw = torch.split(
        kv_lora, [dkv, drope], dim=-1)                # (bs, dkv), (bs, drope)
    # Upscale the low‑rank NoPE part
    kv_nope = F.linear(kv_nope_raw, wUKV)                # (bs, nh*(dnope+dv))
    kv_nope = kv_nope.view(bs, nh, dnope + dv)            # (bs, nh, dnope+dv)
    k_nope, v = torch.split(kv_nope, [dnope, dv], dim=-1)   # (bs, nh, dnope), (bs, nh, dv)

    # -------------------------------------------------
    # 5️⃣ Project query NoPE into low‑rank space (per‑head)
    # -------------------------------------------------
    w_UKV = wUKV.view(nh, dnope + dv, dkv)               # (nh, dnope+dv, dkv)
    w_UKV_k = w_UKV[:, :dnope, :]                        # (nh, dnope, dkv)

    if dnope > 0:
        # (bs, nh, dkv) = bmm over heads
        q_nope = torch.einsum('bhd,hdc->bhc', q_nope_raw, w_UKV_k)
    else:
        q_nope = torch.zeros((bs, nh, dkv),
                             dtype=x_flat.dtype,
                             device=x_flat.device)

    # -------------------------------------------------
    # 6️⃣ RoPE for queries (single token) – fused into one op
    # -------------------------------------------------
    cos_q = rope_cos[query_pos].view(1, 1, drope)   # (1,1,drope)
    sin_q = rope_sin[query_pos].view(1, 1, drope)
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, drope)

    # -------------------------------------------------
    # 7️⃣ Prepare keys for RoPE (full cache)
    # -------------------------------------------------
    # low‑rank key part (shared across heads)
    kv_nope_seq = kv_cache_data[:, :kv_len, :dkv]                # (bs, kv_len, dkv)
    # RoPE part (shared across heads)
    k_rope = kv_cache_data[:, :kv_len, dkv:]                # (bs, kv_len, drope)

    cos_k = rope_cos[:kv_len].unsqueeze(0)      # (1, kv_len, drope)
    sin_k = rope_sin[:kv_len].unsqueeze(0)      # (1, kv_len, drope)

    k_rope = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, drope)

    # -------------------------------------------------
    # 8️⃣ Compute attention scores (low‑rank + RoPE)
    # -------------------------------------------------
    # low‑rank dot‑product   (bs, nh, kv_len)
    scores_nope = torch.bmm(q_nope, kv_nope_seq.permute(0, 2, 1))
    # rope dot‑product        (bs, nh, kv_len)
    scores_rope = torch.bmm(q_rope, k_rope.permute(0, 2, 1))

    # scaling
    scale = 1.0 / math.sqrt(dnope + drope)
    scores = (scores_nope + scores_rope) * scale

    # -------------------------------------------------
    # 9️⃣ Row‑wise softmax (fused Triton kernel)
    # -------------------------------------------------
    scores_flat = scores.reshape(bs * nh, kv_len)                # (B*H, kv_len)
    attn_flat = _rowwise_softmax(scores_flat)                    # (B*H, kv_len)
    attn = attn_flat.view(bs, nh, kv_len)                        # (bs, nh, kv_len)

    # -------------------------------------------------
    # 🔟 Weighted sum in low‑rank space (Z = Σ a_i * v_i)
    # -------------------------------------------------
    Z = torch.bmm(attn, kv_nope_seq)                            # (bs, nh, dkv)

    # -------------------------------------------------
    # 1️⃣1️⃣ Project Z → value space and final projection
    # -------------------------------------------------
    # w_UKV_val_T : (nh, dkv, dv)
    w_UKV_val_T = w_UKV[:, dnope:, :].permute(0, 2, 1)          # (nh, dkv, dv)
    y_head = torch.einsum('bhd,hdc->bhc', Z, w_UKV_val_T)       # (bs, nh, dv)

    # Output projection back to model dimension
    y_head_flat = y_head.reshape(bs, nh * dv)                  # (bs, nh*dv)
    out = F.linear(y_head_flat, wO)                            # (bs, dim)
    out = out.unsqueeze(1)                                     # (bs, 1, dim)

    return out, kv_cache_data, seq


# ----------------------------------------------------------------------
# Public entry point – the required kernel interface
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton‑accelerated MLA forward (single‑token generation step).

    Parameters
    ----------
    data : Tuple[Config, torch.Tensor, KVCache]
        - config  : Model hyper‑parameters and pre‑loaded weights.
        - x       : Input tensor of shape [batch_size, 1, dim] (seq_len == 1).
        - kv_cache: KVCache instance that holds the past keys/values.

    Returns
    -------
    output : torch.Tensor
        Shape [batch_size, 1, dim] (BF16).
    kv_cache : torch.Tensor
        Updated cache tensor of shape [batch_size, max_seq_len, kv_lora_rank + qk_rope_head_dim].
    """
    config, x, kv_cache = data

    # -----------------------------------------------------------------
    # Convenience shortcuts
    # -----------------------------------------------------------------
    bs   = config.batch_size
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    max_seq_len = config.max_seq_len

    # -----------------------------------------------------------------
    # Weight tensors (already on the correct device & dtype)
    # -----------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv+drope, dim)
    wUQ   = config.Q_proj_up_weight            # (nh*(dnope+drope), dq)
    wUKV  = config.KV_proj_up_weight           # (nh*(dnope+dv), dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # -----------------------------------------------------------------
    # RoPE cosine / sine tables (cached on the Config instance)
    # -----------------------------------------------------------------
    if not hasattr(config, "_rope_cos"):
        cos_tab, sin_tab = _rope_cache(drope, max_seq_len, x.dtype, x.device)
        config._rope_cos, config._rope_sin = cos_tab, sin_tab
    rope_cos = config._rope_cos
    rope_sin = config._rope_sin

    # -----------------------------------------------------------------
    # Flatten the input (seq_len == 1 for generation)
    # -----------------------------------------------------------------
    x_flat = x.squeeze(1)                     # (bs, dim)

    # -----------------------------------------------------------------
    # Compile the heavy sub‑graph once (torch.compile ↔ fusion)
    # -----------------------------------------------------------------
    if not hasattr(custom_kernel, "_compiled"):
        custom_kernel._compiled = torch.compile(
            _core_forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

    # -----------------------------------------------------------------
    # Execute the compiled forward
    # -----------------------------------------------------------------
    out, new_kv_data, new_kv_len = custom_kernel._compiled(
        x_flat,
        wDQ,
        wDKV,
        wUQ,
        wUKV,
        wO,
        rope_cos,
        rope_sin,
        kv_cache.data,
        kv_cache.seq_len,
        config,
    )

    # -----------------------------------------------------------------
    # Update KVCache in‑place so the caller sees the new state
    # -----------------------------------------------------------------
    kv_cache.data = new_kv_data
    kv_cache.seq_len = new_kv_len

    # -----------------------------------------------------------------
    # Return output and the updated cache tensor
    # -----------------------------------------------------------------
    return out, kv_cache.data