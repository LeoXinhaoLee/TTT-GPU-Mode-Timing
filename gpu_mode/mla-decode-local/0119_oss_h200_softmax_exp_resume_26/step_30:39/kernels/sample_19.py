### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###

# --------------------------------------------------------------
# Additional imports
# --------------------------------------------------------------
import triton.language as tl

# --------------------------------------------------------------
# Helper utilities (RoPE cache, rotate_half)
# --------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((-b, a), dim=-1)


def _rope_cache(drope: int,
               max_len: int,
               dtype: torch.dtype,
               device: torch.device):
    """
    Pre‑compute the RoPE cosine / sine tables (shared across all tokens).

    Returns:
        cos (max_len, drope), sin (max_len, drope) in the requested dtype.
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
    theta_pos = torch.einsum("p,d->pd", pos, theta)          # (max_len, half)
    theta_pos = torch.cat([theta_pos, theta_pos], dim=-1)   # (max_len, drope)

    cos = torch.cos(theta_pos).to(torch.bfloat16)
    sin = torch.sin(theta_pos).to(torch.bfloat16)

    cache[key] = (cos, sin)
    return cos, sin


# --------------------------------------------------------------
# Triton‑fused row‑wise softmax for BF16 (used on the attention scores)
# --------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    row_stride, col_stride,
    rows, cols,
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Row‑wise softmax for a 2‑D BF16 matrix."""
    row = tl.program_id(0)
    if row >= rows:
        return

    col = tl.arange(0, BLOCK)
    mask = col < cols

    # Load BF16 and cast to FP32 for numeric stability
    val = tl.load(in_ptr + row * row_stride + col * col_stride,
                  mask=mask, other=-float("inf"))
    val_f32 = tl.cast(val, tl.float32)

    # Stable softmax
    row_max = tl.max(val_f32, axis=0)
    val_f32 = val_f32 - row_max
    exp_val = tl.exp(val_f32)
    sum_exp = tl.sum(exp_val, axis=0)

    out_f32 = exp_val / sum_exp
    out = tl.cast(out_f32, tl.bfloat16)

    tl.store(out_ptr + row * row_stride + col * col_stride,
             out, mask=mask)


def _rowwise_softmax(x: torch.Tensor) -> torch.Tensor:
    """Apply the fused row‑wise softmax to a 2‑D BF16 tensor."""
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


# --------------------------------------------------------------
# Core forward (compiled once with torch.compile)
# --------------------------------------------------------------
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
    Heavy‑weight sub‑graph for a single generation step (seq_len == 1).

    Returns:
        out            - (bs, 1, dim)   model output for the new token
        new_kv_cache  - (bs, max_seq_len, dkv+drope)  updated cache
        new_kv_len    - int, new length of the cache
    """
    # -----------------------------------------------------------------
    # Unpack config values for readability
    # -----------------------------------------------------------------
    bs   = config.batch_size
    nh   = config.n_heads
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    dkv  = config.kv_lora_rank

    # -----------------------------------------------------------------
    # 1️⃣ Down‑project Q and KV (low‑rank)
    # -----------------------------------------------------------------
    q_lora = F.linear(x_flat, wDQ)               # (bs, dq)
    kv_lora = F.linear(x_flat, wDKV)             # (bs, dkv+drope)

    # -----------------------------------------------------------------
    # 2️⃣ Update KV‑cache (store low‑rank + rope part)
    # -----------------------------------------------------------------
    seq_idx = kv_cache_seq_len                     # current length before write
    kv_cache_data[:, seq_idx:seq_idx + 1, :] = kv_lora.unsqueeze(1)
    seq_idx += 1
    kv_len = seq_idx                               # == old_len + 1

    # -----------------------------------------------------------------
    # 3️⃣ Up‑project queries, split into NoPE / RoPE parts
    # -----------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                  # (bs, nh*(dnope+drope))
    q_up = q_up.view(bs, nh, dnope + drope)      # (bs, nh, dnope+drope)
    q_nope, q_rope_raw = torch.split(q_up, [dnope, drope], dim=-1)   # (bs,nh,dnope), (bs,nh,drope)

    # -----------------------------------------------------------------
    # 4️⃣ Slice KV‑up‑projection once (w_UKV_k and w_UKV_val_T)
    # -----------------------------------------------------------------
    w_UKV = wUKV.view(nh, dnope + dv, dkv)        # (nh, dnope+dv, dkv)

    # -----------------------------------------------------------------
    # 5️⃣ Project q_nope → low‑rank (dkv) using w_UKV_k
    # -----------------------------------------------------------------
    w_UKV_k = w_UKV[:, :dnope, :]                 # (nh, dnope, dkv)
    # Batched einsum: (bs, nh, dnope) x (nh, dnope, dkv) -> (bs, nh, dkv)
    q_nope_proj = torch.einsum('bhd, hdc -> bhc', q_nope, w_UKV_k)   # (bs, nh, dkv)

    # -----------------------------------------------------------------
    # 6️⃣ Retrieve KV low‑rank sequence and the RoPE part from cache
    # -----------------------------------------------------------------
    kv_nope_seq = kv_cache_data[:, :kv_len, :dkv]                # (bs, kv_len, dkv)
    k_rope_raw  = kv_cache_data[:, :kv_len, dkv:]               # (bs, kv_len, drope)

    # -----------------------------------------------------------------
    # 7️⃣ Apply RoPE to the query (single position)
    # -----------------------------------------------------------------
    q_pos = kv_len - 1                                            # position of the new token
    cos_q = rope_cos[q_pos].view(1, 1, drope)                     # (1,1,drope)
    sin_q = rope_sin[q_pos].view(1, 1, drope)                     # (1,1,drope)

    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, drope)

    # -----------------------------------------------------------------
    # 8️⃣ Apply RoPE to all keys (vectorised over the whole cache)
    # -----------------------------------------------------------------
    cos_k = rope_cos[:kv_len].unsqueeze(0)          # (1, kv_len, drope)
    sin_k = rope_sin[:kv_len].unsqueeze(0)          # (1, kv_len, drope)
    # broadcast across batch dimension (bs, kv_len, drope)
    k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k   # (bs, kv_len, drope)

    # -----------------------------------------------------------------
    # 9️⃣ Compute attention scores (both parts) and scale
    # -----------------------------------------------------------------
    # a) RoPE part (bs, nh, drope) @ (bs, kv_len, drope)^T -> (bs, nh, kv_len)
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))

    # b) No‑PE part (bs, nh, dkv) @ (bs, kv_len, dkv)^T -> (bs, nh, kv_len)
    scores_nope = torch.matmul(q_nope_proj, kv_nope_seq.transpose(-2, -1))

    # c) Combine + scaling
    scale = 1.0 / math.sqrt(dnope + drope)
    scores = (scores_nope + scores_rope) * scale                     # (bs, nh, kv_len)

    # -----------------------------------------------------------------
    # 🔟 Row‑wise softmax (fused Triton kernel)
    # -----------------------------------------------------------------
    scores_flat = scores.reshape(bs * nh, kv_len)                # (B*H, K)
    attn_flat = _rowwise_softmax(scores_flat)                   # (B*H, K)
    attn = attn_flat.view(bs, nh, kv_len)                       # (bs, nh, kv_len)

    # -----------------------------------------------------------------
    # 1️⃣1️⃣ Weighted sum in low‑rank space (Z = Σ a_i * v_i)
    # -----------------------------------------------------------------
    # kv_nope_seq contains the low‑rank “value” part (dkv)
    Z = torch.einsum('bhn,bnd->bhd', attn, kv_nope_seq)         # (bs, nh, dkv)

    # -----------------------------------------------------------------
    # 1️⃣2️⃣ Project Z to the final value dimension (dv) – per‑head linear
    # -----------------------------------------------------------------
    w_UKV_val_T = w_UKV[:, dnope:, :].permute(0, 2, 1)          # (nh, dkv, dv)
    y_head = torch.einsum('bhd, hdc -> bhc', Z, w_UKV_val_T)   # (bs, nh, dv)

    # -----------------------------------------------------------------
    # 1️⃣3️⃣ Output projection back to model dimension
    # -----------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)                   # (bs, nh*dv)
    out = F.linear(y_head_flat, wO)                              # (bs, dim)
    out = out.unsqueeze(1)                                       # (bs, 1, dim)

    return out, kv_cache_data, kv_len


# --------------------------------------------------------------
# Public entry point – called by the benchmark harness
# --------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton‑accelerated MLA forward (single‑token generation step).

    Parameters
    ----------
    data : Tuple[Config, torch.Tensor, KVCache]
        - Config instance containing model hyper‑parameters and weight tensors.
        - Input tensor x of shape (batch_size, seq_len, dim).  For generation ``seq_len`` is 1.
        - KVCache instance that stores the low‑rank key/value cache.

    Returns
    -------
    output : torch.Tensor
        Shape (batch_size, 1, dim) – the model output for the current token.
    kv_cache : torch.Tensor
        Updated cache data of shape (batch_size, max_seq_len, kv_lora_rank + qk_rope_head_dim).
    """
    config, x, kv_cache = data

    # -------------------------------------------------------------
    # Resolve shortcuts from config
    # -------------------------------------------------------------
    bs   = config.batch_size
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    max_seq_len = config.max_seq_len

    # -------------------------------------------------------------
    # Weight tensors (already on device, BF16)
    # -------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, dim)
    wDKV  = config.KV_proj_down_weight         # (dkv+drope, dim)
    wUQ   = config.Q_proj_up_weight            # (nh*(dnope+drope), dq)
    wUKV  = config.KV_proj_up_weight           # (nh*(dnope+dv), dkv)
    wO    = config.wo_weight                   # (dim, nh*dv)

    # -------------------------------------------------------------
    # RoPE cache (cosine / sine tables) – allocate once per config
    # -------------------------------------------------------------
    if not hasattr(config, "_rope_cos"):
        cos_tab, sin_tab = _rope_cache(drope, max_seq_len, x.dtype, x.device)
        config._rope_cos, config._rope_sin = cos_tab, sin_tab
    rope_cos = config._rope_cos
    rope_sin = config._rope_sin

    # -------------------------------------------------------------
    # Flatten input (seq_len == 1 for generation) – shape (bs, dim)
    # -------------------------------------------------------------
    x_flat = x.squeeze(1)   # (bs, dim)

    # -------------------------------------------------------------
    # Compile the heavyweight sub‑graph once (torch.compile) to fuse
    # linear layers, einsums, and the Triton softmax.
    # -------------------------------------------------------------
    if not hasattr(custom_kernel, "_compiled"):
        # torch.compile will trace the sub‑graph and fuse supported ops.
        custom_kernel._compiled = torch.compile(
            _core_forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

    # -------------------------------------------------------------
    # Execute the compiled graph
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Update KVCache in‑place so callers see the new cache
    # -------------------------------------------------------------
    kv_cache.data = new_kv_data
    kv_cache.seq_len = new_kv_len

    # -------------------------------------------------------------
    # Return output and the (updated) cache tensor
    # -------------------------------------------------------------
    return out, kv_cache.data