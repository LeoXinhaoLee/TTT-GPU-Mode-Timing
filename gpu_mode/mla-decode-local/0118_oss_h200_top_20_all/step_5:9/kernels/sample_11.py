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

# ------------------------------------------------------------
#  Helper utilities (RoPE cache, rotate‑half, Triton softmax)
# ------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


_rope_cache: dict[Tuple[int, int, torch.device],
                 Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta = 10000 ** (-i / half)  (float32 -> bf16)
        theta = (10000.0 ** (-torch.arange(half,
                                          dtype=torch.float32,
                                          device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                # (max_seq_len,1)
        idx = pos * theta[None, :]                                   # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                          # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Row‑wise softmax for a 2‑D bf16 tensor."""
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N_COLS, BLOCK_SIZE):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
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
        BLOCK = 1 << (n_cols - 1).bit_length()
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
#  Optimised MLA forward – fused and trimmed
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns (output, updated_kv_cache) where
      - output : torch.Tensor of shape (batch, seq_len, dim)  (bf16)
      - updated_kv_cache : the tensor stored inside KVCache after the update
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # 1️⃣  Unpack config & weights
    # ------------------------------------------------------------------
    bs   = config.batch_size                # batch size
    sl   = config.seq_len                   # sequence length (always 1 here)
    nh   = config.n_heads                   # number of heads
    d    = config.dim                       # model dimension
    dq   = config.q_lora_rank               # Q‑down rank
    dkv  = config.kv_lora_rank              # KV‑down rank (latent dimension)
    d_nope = config.qk_nope_head_dim        # No‑PE head dim
    d_rope = config.qk_rope_head_dim        # RoPE head dim
    dv   = config.v_head_dim                # value head dim
    msl  = config.max_seq_len               # maximum sequence length

    # weight tensors (already on correct device & dtype==bf16)
    wDQ   = config.Q_proj_down_weight                # (dq, d)
    wDKV  = config.KV_proj_down_weight               # (dkv + d_rope, d)
    wUQ   = config.Q_proj_up_weight                  # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight                 # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                         # (d, nh*dv)

    # ------------------------------------------------------------------
    # 2️⃣  Q down‑projection + up‑projection
    # ------------------------------------------------------------------
    q_lora = F.linear(x, wDQ)                     # (bs, sl, dq)
    q_up   = F.linear(q_lora, wUQ)                 # (bs, sl, (d_nope+d_rope)*nh)
    # reshape to (bs, sl, nh, d_nope+d_rope)
    q_up = q_up.view(bs, sl, nh, d_nope + d_rope)

    # ------------------------------------------------------------------
    # 3️⃣  KV down‑projection + cache update
    # ------------------------------------------------------------------
    kv_lora_in = F.linear(x, wDKV)                # (bs, sl, dkv+d_rope)
    kv_lora, kv_len = kv_cache(kv_lora_in)        # kv_lora : (bs, kv_len, dkv+d_rope)
    kv_len_int = int(kv_len)                      # integer length of cache after insertion

    # split latent part and rope part
    kv_latent = kv_lora[..., :dkv]                # (bs, kv_len, dkv)
    k_rope_raw = kv_lora[..., dkv:]               # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  RoPE tables (cached)
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ------------------------------------------------------------------
    # 5️⃣  Split Q into NOPE & RoPE, apply RoPE
    # ------------------------------------------------------------------
    # q_up : (bs, sl, nh, d_nope+d_rope)
    q_nope_raw = q_up[..., :d_nope]               # (bs, sl, nh, d_nope)
    q_rope_raw = q_up[..., d_nope:]               # (bs, sl, nh, d_rope)

    # remove the (always‑1) time dimension
    if sl != 1:
        raise RuntimeError("MLA implementation assumes seq_len == 1")
    q_nope_raw = q_nope_raw.squeeze(1)            # (bs, nh, d_nope)
    q_rope_raw = q_rope_raw.squeeze(1)            # (bs, nh, d_rope)

    # ----- Q RoPE (single position) -----
    query_pos = kv_len_int - 1                     # absolute position of newest token
    cos_q = cos_table[query_pos]                  # (d_rope,)
    sin_q = sin_table[query_pos]                  # (d_rope,)
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, d_rope)

    # ----- K RoPE (all cached positions) -----
    cos_k = cos_table[:kv_len_int]                # (kv_len, d_rope)
    sin_k = sin_table[:kv_len_int]                # (kv_len, d_rope)
    cos_k_b = cos_k[None, :, :]                   # (1, kv_len, d_rope)
    sin_k_b = sin_k[None, :, :]                   # (1, kv_len, d_rope)
    k_rope = k_rope_raw * cos_k_b + _rotate_half(k_rope_raw) * sin_k_b   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Fast path when there is **no** No‑PE head dimension
    # ------------------------------------------------------------------
    if d_nope == 0:
        #   Q : (bs, 1, nh, d_rope)
        Q = q_rope.unsqueeze(1)                                 # (bs, 1, nh, d_rope)

        #   K : (bs, kv_len, nh, d_rope)  (broadcast rope keys across heads)
        K = k_rope.unsqueeze(2).expand(-1, -1, nh, -1)           # (bs, kv_len, nh, d_rope)

        #   V : (bs, kv_len, nh, dkv)  (broadcast latent values across heads)
        V = kv_latent.unsqueeze(2).expand(-1, -1, nh, -1)       # (bs, kv_len, nh, dkv)

        # Flash‑attention (scaled dot‑product).  Head‑dim = d_rope, so use default
        # scaling (1/√d_rope).  We keep the explicit scale for clarity.
        scale = 1.0 / math.sqrt(d_rope)
        M = F.scaled_dot_product_attention(Q, K, V, scale=scale)   # (bs,1,nh,dkv)
        M = M.squeeze(1)                                            # (bs, nh, dkv)

        # ----- Value projection (per‑head) -----
        # wUKV view -> (nh, d_nope+dv, dkv) -> (nh, dv, dkv) because d_nope==0
        wV = wUKV.view(nh, d_nope + dv, dkv)[:, :, :]                # (nh, dv, dkv)
        wV_T = wV.permute(0, 2, 1)                                 # (nh, dkv, dv)
        y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)             # (bs, nh, dv)

    else:
        # ------------------------------------------------------------------
        # 7️⃣  General case (both NOPE and RoPE parts)
        # ------------------------------------------------------------------
        # view KV up‑projection weight as per‑head blocks
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)

        # ----- linear maps for the NOPE part -----
        wK = wUKV_view[:, :d_nope, :]                             # (nh, d_nope, dkv)
        wV = wUKV_view[:, d_nope:, :]                             # (nh, dv, dkv)

        # ----- project Q_nope into latent space -----
        if d_nope > 0:
            # q_nope_raw : (bs, nh, d_nope)
            q_nope_latent = torch.einsum('bhd,hdm->bhm', q_nope_raw, wK)   # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros(bs, nh, dkv,
                                        dtype=torch.bfloat16,
                                        device=x.device)

        # ----- attention scores -----
        # latent contribution
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-1, -2))            # (bs, nh, kv_len)
        # RoPE contribution
        scores_rope = torch.matmul(q_rope,
                                   k_rope.transpose(-1, -2))               # (bs, nh, kv_len)

        scale = 1.0 / math.sqrt(d_nope + d_rope)
        scores = (scores_nope + scores_rope) * scale                 # (bs, nh, kv_len)

        # ----- soft‑max (Triton) -----
        scores_2d = scores.view(bs * nh, kv_len_int)                 # (B*H, L)
        attn_2d   = _triton_softmax(scores_2d)                       # (B*H, L)
        attn = attn_2d.view(bs, nh, kv_len_int)                     # (bs, nh, kv_len)

        # ----- weighted sum of latent keys -----
        M = torch.matmul(attn, kv_latent)                           # (bs, nh, dkv)

        # ----- value projection -----
        wV_T = wV.permute(0, 2, 1)                                   # (nh, dkv, dv)
        y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)               # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 8️⃣  Output projection (merge heads & final linear)
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, -1)                       # (bs, nh*dv)
    output = F.linear(y, wO)                         # (bs, d)
    output = output.unsqueeze(1)                     # (bs, 1, d)

    # ------------------------------------------------------------------
    # Return the output and the *updated* KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data