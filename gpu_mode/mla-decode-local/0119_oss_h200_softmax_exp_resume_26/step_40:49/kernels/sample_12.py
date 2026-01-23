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

# --------------------------------------------------------------
# Helper utilities (rotate‑half, RoPE cache, fused row‑wise softmax)
# --------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate the second half of the last dimension (same as RoPE.rotate_half)."""
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((-b, a), dim=-1)


def _rope_cache(drope: int, max_len: int, dtype: torch.dtype, device: torch.device):
    """
    Pre‑compute cosine/sine tables for rotary embeddings.
    Returned tensors have shape (max_len, drope) and dtype bfloat16.
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
    theta_pos = torch.cat([theta_pos, theta_pos], dim=-1)    # (max_len, drope)

    cos = torch.cos(theta_pos).to(torch.bfloat16)
    sin = torch.sin(theta_pos).to(torch.bfloat16)

    cache[key] = (cos, sin)
    return cos, sin


# --------------------------------------------------------------
# Triton fused row‑wise softmax (BF16)
# --------------------------------------------------------------

@triton.jit
def _softmax_fused_kernel(
    out_ptr, in_ptr,                # pointers
    row_stride, col_stride,         # strides (elements, not bytes)
    rows, cols,                     # matrix shape
    BLOCK: tl.constexpr,            # power‑of‑2 ≥ cols
    NUM_STAGES: tl.constexpr,       # software‑pipelining depth
):
    """Fused row‑wise softmax for BF16 tensors."""
    row = tl.program_id(0)
    if row >= rows:
        return

    col = tl.arange(0, BLOCK)                     # (BLOCK,)
    mask = col < cols
    # load input
    in_off = in_ptr + row * row_stride + col * col_stride
    val = tl.load(in_off, mask=mask, other=-float("inf"))
    val_f32 = tl.cast(val, tl.float32)

    # max & subtract
    row_max = tl.max(val_f32, axis=0)
    val_f32 = val_f32 - row_max

    # exp & sum
    exp_val = tl.exp(val_f32)
    sum_exp = tl.sum(exp_val, axis=0)

    # normalize
    out_f32 = exp_val / sum_exp
    out_bf16 = tl.cast(out_f32, tl.bfloat16)

    # write output
    out_off = out_ptr + row * row_stride + col * col_stride
    tl.store(out_off, out_bf16, mask=mask)


def _rowwise_softmax(x: torch.Tensor) -> torch.Tensor:
    """Apply the fused kernel to a 2‑D BF16 tensor."""
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


# --------------------------------------------------------------
# Core MLA forward (torch ops + fused softmax)
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
    kv_cache_len: int,                        # current filled length
    config: Config,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Triton‑accelerated MLA forward for a single generation token.
    Returns:
        out                : (bs, 1, dim)
        new_kv_cache_data : (bs, max_seq_len, dkv+drope)
        new_kv_len        : int
    """
    # ---- unpack config -------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim
    dkv = config.kv_lora_rank
    dq = config.q_lora_rank

    # ---- 1️⃣ low‑rank projections (Q ↓ & KV ↓) ------------------------
    q_lora = F.linear(x_flat, wDQ)                # (bs, dq)
    kv_lora = F.linear(x_flat, wDKV)              # (bs, dkv+drope)

    # ---- 2️⃣ KV‑cache update (append the new token) --------------------
    pos = kv_cache_len
    kv_cache_data[:, pos:pos + 1, :] = kv_lora.unsqueeze(1)   # (bs, 1, dkv+drope)
    new_kv_len = pos + 1

    # ---- 3️⃣ Up‑project Q (split NoPE / RoPE) -------------------------
    q_up = F.linear(q_lora, wUQ)                              # (bs, nh*(dnope+drope))
    q_up = q_up.view(bs, nh, dnope + drope)                  # (bs, nh, dnope+drope)
    q_nope, q_rope_raw = torch.split(q_up, [dnope, drope], dim=-1)

    # ---- 4️⃣ Split KV low‑rank + RoPE pieces from the cache -----------
    kv_nope_seq = kv_cache_data[:, :new_kv_len, :dkv]           # (bs, kv_len, dkv)
    k_rope_raw = kv_cache_data[:, :new_kv_len, dkv:]           # (bs, kv_len, drope)

    # ---- 5️⃣ Prepare up‑projection weights for K‑NoPE and V -----------
    w_UKV = wUKV.view(nh, dnope + dv, dkv)                # (nh, dnope+dv, dkv)
    w_UKV_k = w_UKV[:, :dnope, :]                         # (nh, dnope, dkv)
    w_UKV_val_T = w_UKV[:, dnope:, :].permute(0, 2, 1)   # (nh, dkv, dv)

    # ---- 6️⃣ Project Q‑NoPE into low‑rank space (head‑wise) -----------
    # (bs, nh, dnope) x (nh, dnope, dkv) -> (bs, nh, dkv)
    q_nope_proj = torch.einsum('bhd,hdc->bhc', q_nope, w_UKV_k)

    # ---- 7️⃣ RoPE on queries (single position) -----------------------
    qpos = new_kv_len - 1
    cos_q = rope_cos[qpos].view(1, 1, drope)   # (1,1,drope)
    sin_q = rope_sin[qpos].view(1, 1, drope)
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, drope)

    # ---- 8️⃣ RoPE on keys (vectorised over the whole cache) ----------
    cos_k = rope_cos[:new_kv_len].unsqueeze(0)      # (1,kv_len,drope)
    sin_k = rope_sin[:new_kv_len].unsqueeze(0)      # (1,kv_len,drope)
    k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k   # (bs, kv_len, drope)

    # ---- 9️⃣ Compute raw attention scores (low‑rank + RoPE) ----------
    scores_nope = torch.matmul(q_nope_proj,
                               kv_nope_seq.transpose(-2, -1))          # (bs, nh, kv_len)
    scores_rope = torch.matmul(q_rope,
                               k_rope.transpose(-2, -1))               # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(dnope + drope)
    scores = (scores_nope + scores_rope) * scale                     # (bs, nh, kv_len)

    # ---- 🔟 Row‑wise softmax (fused Triton kernel) --------------------
    scores_flat = scores.reshape(bs * nh, new_kv_len)                # (B*H, kv_len)
    attn_flat = _rowwise_softmax(scores_flat)                        # (B*H, kv_len)
    attn = attn_flat.view(bs, nh, new_kv_len)                        # (bs, nh, kv_len)

    # ---- 1️⃣1️⃣ Weighted sum of low‑rank V -----------------------------
    Z = torch.matmul(attn, kv_nope_seq)                               # (bs, nh, dkv)

    # ---- 1️⃣2️⃣ Project Z to final value dimension (dv) ----------------
    y_head = torch.einsum('bhd,hdc->bhc', Z, w_UKV_val_T)           # (bs, nh, dv)

    # ---- 1️⃣3️⃣ Output projection back to model dimension -------------
    y_head_flat = y_head.reshape(bs, nh * dv)                        # (bs, nh*dv)
    out = F.linear(y_head_flat, wO)                                   # (bs, dim)
    out = out.unsqueeze(1)                                            # (bs, 1, dim)

    return out, kv_cache_data, new_kv_len


# --------------------------------------------------------------
# Public entry point – Triton‑accelerated MLA forward
# --------------------------------------------------------------

def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton‑accelerated MLA forward (single‑token generation step).

    Returns
    -------
    output : torch.Tensor
        Shape [batch_size, seq_len, dim]   (seq_len == 1)
    kv_cache : torch.Tensor
        Updated cache data (shape [bs, max_seq_len, dkv+drope])
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
    # Weight tensors (already on device, BF16)
    # -------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv+drope, dim)
    wUQ  = config.Q_proj_up_weight            # (nh*(dnope+drope), dq)
    wUKV = config.KV_proj_up_weight           # (nh*(dnope+dv), dkv)
    wO   = config.wo_weight                    # (dim, nh*dv)

    # -------------------------------------------------
    # RoPE cache – pre‑compute cosine / sine tables once
    # -------------------------------------------------
    if not hasattr(config, "_rope_cos"):
        cos_tab, sin_tab = _rope_cache(drope, max_seq_len, x.dtype, x.device)
        config._rope_cos, config._rope_sin = cos_tab, sin_tab
    rope_cos = config._rope_cos
    rope_sin = config._rope_sin

    # -------------------------------------------------
    # Flatten input (seq_len is always 1 for generation)
    # -------------------------------------------------
    x_flat = x.squeeze(1)               # (bs, dim)

    # -------------------------------------------------
    # Compile heavyweight sub‑graph once (torch.compile)
    # -------------------------------------------------
    if not hasattr(custom_kernel, "_compiled"):
        # Torch compile will fuse arithmetic; softmax stays external.
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
    # Update KVCache instance in‑place so that the caller sees it
    # -------------------------------------------------
    kv_cache.data = new_kv_data
    kv_cache.seq_len = new_kv_len

    # -------------------------------------------------
    # Return output (shape [bs, 1, dim]) and the cache tensor
    # -------------------------------------------------
    return out, kv_cache.data