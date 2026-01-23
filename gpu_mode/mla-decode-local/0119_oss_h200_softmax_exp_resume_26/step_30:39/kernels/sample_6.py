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
# Helper utilities
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the last dimension (used by RoPE)."""
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((-b, a), dim=-1)


def _rope_cache(drope: int, max_len: int, dtype: torch.dtype, device: torch.device):
    """
    Build (cos, sin) tables for Rotary Positional Embedding (BF16).
    Cached across calls to avoid recomputation.
    """
    key = (drope, max_len, dtype, device)
    if not hasattr(_rope_cache, "_CACHE"):
        _rope_cache._CACHE = {}
    cache = _rope_cache._CACHE
    if key in cache:
        return cache[key]

    half = drope // 2
    theta = 10000.0 ** (-torch.arange(0, half, dtype=dtype, device=device) / float(half))
    pos = torch.arange(max_len, dtype=dtype, device=device)
    theta_pos = torch.einsum("p,d->pd", pos, theta)            # (max_len, half)
    theta_pos = torch.cat([theta_pos, theta_pos], dim=-1)      # (max_len, drope)

    cos = torch.cos(theta_pos).to(torch.bfloat16)
    sin = torch.sin(theta_pos).to(torch.bfloat16)

    cache[key] = (cos, sin)
    return cos, sin


# ----------------------------------------------------------------------
# Triton fused row‑wise softmax (bfloat16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_fused_kernel(
    out_ptr, in_ptr,                 # pointers
    row_stride, col_stride,          # strides (in elements)
    rows, cols,                      # matrix dimensions
    BLOCK: tl.constexpr,             # power‑of‑2 ≥ cols
    NUM_STAGES: tl.constexpr,        # software‑pipelining depth
):
    """Fused row‑wise softmax for BF16 tensors."""
    row = tl.program_id(0)
    if row >= rows:
        return

    col = tl.arange(0, BLOCK)
    mask = col < cols

    # Load (padded) row
    ptr = in_ptr + row * row_stride + col * col_stride
    val_bf16 = tl.load(ptr, mask=mask, other=-float("inf"))
    val_f32 = tl.cast(val_bf16, tl.float32)

    # Stable softmax
    row_max = tl.max(val_f32, axis=0)
    val_f32 = val_f32 - row_max
    exp_val = tl.exp(val_f32)
    sum_exp = tl.sum(exp_val, axis=0)
    out_f32 = exp_val / sum_exp
    out_bf16 = tl.cast(out_f32, tl.bfloat16)

    # Write back
    out_ptrs = out_ptr + row * row_stride + col * col_stride
    tl.store(out_ptrs, out_bf16, mask=mask)


def _rowwise_softmax(x: torch.Tensor) -> torch.Tensor:
    """Apply the fused softmax to a 2‑D BF16 tensor."""
    rows, cols = x.shape
    BLOCK = triton.next_power_of_2(cols)

    out = torch.empty_like(x)
    _softmax_fused_kernel[(rows,)](
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
# Core forward (single‑token generation)
# ----------------------------------------------------------------------
def _core_forward(
    x_flat: torch.Tensor,                 # (bs, dim)
    wDQ: torch.Tensor,                    # (dq, dim)
    wDKV: torch.Tensor,                   # (dkv+drope, dim)
    wUQ: torch.Tensor,                    # (nh*(dnope+drope), dq)
    wUKV: torch.Tensor,                   # (nh*(dnope+dv), dkv)
    wO: torch.Tensor,                     # (dim, nh*dv)
    rope_cos: torch.Tensor,               # (max_seq_len, drope)
    rope_sin: torch.Tensor,               # (max_seq_len, drope)
    kv_cache_data: torch.Tensor,          # (bs, max_seq_len, dkv+drope)
    kv_cache_len: int,                    # number of tokens already in cache
    config: Config,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute a single generation step of the MLA module.
    Returns:
        out                – (bs, 1, dim)
        new_kv_cache_data  – (bs, max_seq_len, dkv+drope)
        new_kv_len         – int
    """
    bs = config.batch_size
    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # -------------------------------------------------
    # 1️⃣ Down‑projections
    # -------------------------------------------------
    q_lora = F.linear(x_flat, wDQ)          # (bs, dq)
    kv_lora = F.linear(x_flat, wDKV)        # (bs, dkv + drope)

    # -------------------------------------------------
    # 2️⃣ KV‑cache update (write new token)
    # -------------------------------------------------
    seq_pos = kv_cache_len                     # position where the new token will be written
    kv_cache_data[:, seq_pos:seq_pos + 1, :] = kv_lora.unsqueeze(1)
    new_kv_len = seq_pos + 1

    # -------------------------------------------------
    # 3️⃣ Up‑project Q and split into NoPE / RoPE parts
    # -------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                        # (bs, nh*(dnope+drope))
    q_up = q_up.view(bs, nh, dnope + drope)            # (bs, nh, dnope+drope)
    q_nope, q_rope_raw = torch.split(q_up, [dnope, drope], dim=-1)  # each (bs, nh, *)

    # -------------------------------------------------
    # 4️⃣ Extract low‑rank KV from cache
    # -------------------------------------------------
    kv_nope_seq = kv_cache_data[:, :new_kv_len, :dkv]          # (bs, kv_len, dkv)
    k_rope_raw = kv_cache_data[:, :new_kv_len, dkv:]          # (bs, kv_len, drope)

    # -------------------------------------------------
    # 5️⃣ Project Q‑NoPE into low‑rank space (heads)
    # -------------------------------------------------
    # w_UKV is (nh, dnope+dv, dkv)
    w_UKV = wUKV.view(nh, dnope + dv, dkv)   # (nh, dnope+dv, dkv)

    # split into NoPE and V parts
    w_UKV_k = w_UKV[:, :dnope, :]            # (nh, dnope, dkv)
    w_UKV_val_T = w_UKV[:, dnope:, :].permute(0, 2, 1)  # (nh, dkv, dv)

    # (bs, nh, dnope) @ (nh, dnope, dkv) -> (bs, nh, dkv)
    q_nope_proj = torch.einsum('bhd,hdc->bhc', q_nope, w_UKV_k)

    # -------------------------------------------------
    # 6️⃣ RoPE for queries (single position)
    # -------------------------------------------------
    qpos = new_kv_len - 1
    cos_q = rope_cos[qpos].view(1, 1, drope)   # (1,1,drope)
    sin_q = rope_sin[qpos].view(1, 1, drope)
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, drope)

    # -------------------------------------------------
    # 7️⃣ RoPE for keys (broadcast over heads)
    # -------------------------------------------------
    # We need cos/sin for each position up to new_kv_len
    cos_k = rope_cos[:new_kv_len].unsqueeze(0)   # (1, kv_len, drope)
    sin_k = rope_sin[:new_kv_len].unsqueeze(0)   # (1, kv_len, drope)
    k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k   # (bs, kv_len, drope)

    # -------------------------------------------------
    # 8️⃣ Compute attention scores (NoPE + RoPE)
    # -------------------------------------------------
    # (bs, nh, dkv) @ (bs, kv_len, dkv).transpose(-2,-1) -> (bs, nh, kv_len)
    scores_nope = torch.matmul(q_nope_proj, kv_nope_seq.transpose(-2, -1))
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))
    scale = 1.0 / math.sqrt(dnope + drope)
    scores = (scores_nope + scores_rope) * scale

    # -------------------------------------------------
    # 9️⃣ Row‑wise softmax (fused Triton kernel)
    # -------------------------------------------------
    scores_flat = scores.reshape(bs * nh, new_kv_len)          # (B*H, kv_len)
    attn_flat = _rowwise_softmax(scores_flat)                 # (B*H, kv_len)
    attn = attn_flat.view(bs, nh, new_kv_len)                # (bs, nh, kv_len)

    # -------------------------------------------------
    # 🔟 Weighted sum of low‑rank V (using raw KV)
    # -------------------------------------------------
    Z = torch.matmul(attn, kv_nope_seq)                       # (bs, nh, dkv)

    # -------------------------------------------------
    # 1️⃣1️⃣ Project Z to value dimension (dv)
    # -------------------------------------------------
    y_head = torch.einsum('bhd,hdc->bhc', Z, w_UKV_val_T)    # (bs, nh, dv)

    # -------------------------------------------------
    # 1️⃣2️⃣ Output projection back to model dim
    # -------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)                # (bs, nh*dv)
    out = F.linear(y_head_flat, wO)                          # (bs, dim)
    out = out.unsqueeze(1)                                   # (bs, 1, dim)

    return out, kv_cache_data, new_kv_len


# ----------------------------------------------------------------------
# Public entry point – Triton‑accelerated MLA forward
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton‑accelerated MLA forward (single‑token generation step).

    Returns
    -------
    output : torch.Tensor
        Shape [batch_size, seq_len, dim]   (seq_len == 1)
    kv_cache : torch.Tensor
        Updated cache tensor (shape [bs, max_seq_len, dkv+drope])
    """
    config, x, kv_cache = data

    # -------------------------------------------------
    # Resolve shortcuts
    # -------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim
    max_seq_len = config.max_seq_len

    # -------------------------------------------------
    # Model weights (BF16, already on device)
    # -------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight            # (nh*(dnope+drope), dq)
    wUKV = config.KV_proj_up_weight           # (nh*(dnope+dv), dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # -------------------------------------------------
    # RoPE tables – cache once per config
    # -------------------------------------------------
    if not hasattr(config, "_rope_cos"):
        cos_tab, sin_tab = _rope_cache(drope, max_seq_len, x.dtype, x.device)
        config._rope_cos, config._rope_sin = cos_tab, sin_tab
    rope_cos = config._rope_cos
    rope_sin = config._rope_sin

    # -------------------------------------------------
    # Flatten the (bs, 1, dim) token into (bs, dim)
    # -------------------------------------------------
    x_flat = x.squeeze(1)   # (bs, dim)

    # -------------------------------------------------
    # Compile core forward once (torch.compile)
    # -------------------------------------------------
    if not hasattr(custom_kernel, "_compiled"):
        custom_kernel._compiled = torch.compile(
            _core_forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

    # -------------------------------------------------
    # Execute compiled core forward
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Update KVCache instance in‑place so the caller sees it
    # -------------------------------------------------
    kv_cache.data = new_kv_data
    kv_cache.seq_len = new_kv_len

    # -------------------------------------------------
    # Return output (shape [bs, 1, dim]) and the updated cache tensor
    # -------------------------------------------------
    return out, kv_cache.data