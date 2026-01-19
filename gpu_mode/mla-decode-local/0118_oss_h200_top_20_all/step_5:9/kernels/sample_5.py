### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
#  RoPE helper – cached cos/sin tables (bf16)
# ----------------------------------------------------------------------
_rope_cache: dict[Tuple[int, int, torch.device],
                 Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return pre‑computed (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                          dtype=torch.float32,
                                          device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                # (max_seq_len, 1)
        idx = pos * theta[None, :]                # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)      # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ----------------------------------------------------------------------
#  Triton softmax (optional, kept for completeness – not used in the
#  compiled path but available if you want to replace torch.softmax)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
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
#  Compiled forward – the heavy lifting happens here.
# ----------------------------------------------------------------------
_compiled_fwd = None   # will hold the torch.compile‑ed function after first call

def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns (output, updated_kv_cache).
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    #  Unpack config & weights
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
    #  1️⃣  KV cache update (down‑projection only – this part is *outside*
    #  the compiled kernel to keep the in‑place write simple)
    # ------------------------------------------------------------------
    # down‑project x to latent+rope part for KV
    kv_lora_in = F.linear(x, wDKV)                     # (bs, sl, dkv + d_rope)

    # write into cache (in‑place)
    cur_len = kv_cache.seq_len
    kv_cache.data[:, cur_len:cur_len + sl, :] = kv_lora_in
    kv_cache.seq_len = cur_len + sl                     # new length after current token
    kv_len = kv_cache.seq_len                           # scalar we will use later

    # ------------------------------------------------------------------
    #  2️⃣  Build (or fetch) RoPE tables – these are cached globally
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ------------------------------------------------------------------
    #  3️⃣  Compile (or reuse) the heavy part of the forward pass.
    #      We set `dynamic=True` because the KV length varies across calls.
    # ------------------------------------------------------------------
    global _compiled_fwd
    if _compiled_fwd is None:
        def _fwd(x, kv_cache_data, kv_len,
                 wDQ, wUQ, wUKV, wO,
                 cos_tbl, sin_tbl,
                 n_heads, d_nope, d_rope, dkv, dv):
            """
            x                : (bs, sl=1, d)
            kv_cache_data    : (bs, max_seq_len, dkv+d_rope)
            kv_len           : scalar int (current length of the cache)
            wDQ, wUQ, wUKV, wO : weight tensors (bf16)
            cos_tbl, sin_tbl  : (max_seq_len, d_rope) – pre‑computed RoPE tables
            """
            bs = x.shape[0]

            # ---------- Q projection ----------
            q_lora = F.linear(x, wDQ)                               # (bs, 1, dq)
            q_up   = F.linear(q_lora, wUQ)                           # (bs, 1, (d_nope+d_rope)*nh)
            q_up   = q_up.view(bs, 1, n_heads, d_nope + d_rope).squeeze(1)   # (bs, nh, d_nope+d_rope)

            q_nope, q_rope_raw = torch.split(q_up,
                                              [d_nope, d_rope],
                                              dim=-1)                     # (bs, nh, d_nope), (bs, nh, d_rope)

            # ---------- KV sliced from cache ----------
            # latent part
            kv_latent = kv_cache_data[:, :kv_len, :dkv]              # (bs, kv_len, dkv)
            # rope part (the extra d_rope channels)
            k_rope_raw = kv_cache_data[:, :kv_len, dkv:]              # (bs, kv_len, d_rope)

            # ---------- RoPE ----------
            # query position (absolute)
            query_pos = kv_len - 1
            cos_q = cos_tbl[query_pos]                               # (d_rope,)
            sin_q = sin_tbl[query_pos]                               # (d_rope,)

            q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, d_rope)

            # keys – whole cached range
            cos_k = cos_tbl[:kv_len]                                 # (kv_len, d_rope)
            sin_k = sin_tbl[:kv_len]                                 # (kv_len, d_rope)
            cos_k_b = cos_k[None, :, :]                              # (1, kv_len, d_rope)
            sin_k_b = sin_k[None, :, :]                              # (1, kv_len, d_rope)

            k_rope = k_rope_raw * cos_k_b + _rotate_half(k_rope_raw) * sin_k_b   # (bs, kv_len, d_rope)

            # ---------- Project Q‑no‑PE part into latent space ----------
            # wUKV shape: ((d_nope+dv)*nh, dkv) → view as (nh, d_nope+dv, dkv)
            wUKV_view = wUKV.view(n_heads, d_nope + dv, dkv)            # (nh, d_nope+dv, dkv)
            wK = wUKV_view[:, :d_nope, :]                               # (nh, d_nope, dkv)
            # Einsum: (bs, nh, d_nope) • (nh, d_nope, dkv) → (bs, nh, dkv)
            q_nope_lat = torch.einsum('bhd,hdm->bhm', q_nope, wK)      # (bs, nh, dkv)

            # ---------- Attention scores ----------
            scores_nope = torch.matmul(q_nope_lat, kv_latent.transpose(-1, -2))   # (bs, nh, kv_len)
            scores_rope = torch.matmul(q_rope,    k_rope.transpose(-1, -2))      # (bs, nh, kv_len)

            scale = 1.0 / math.sqrt(d_nope + d_rope)
            scores = (scores_nope + scores_rope) * scale                     # (bs, nh, kv_len)

            # ---------- Softmax ----------
            attn = torch.nn.functional.softmax(scores, dim=-1)               # (bs, nh, kv_len)

            # ---------- Weighted sum (M) ----------
            M = torch.matmul(attn, kv_latent)                                 # (bs, nh, dkv)

            # ---------- Project M → values ----------
            wV = wUKV_view[:, d_nope:, :]                                     # (nh, dv, dkv)
            wV_T = wV.permute(0, 2, 1)                                        # (nh, dkv, dv)
            y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)                   # (bs, nh, dv)

            # ---------- Output projection ----------
            y = y_head.reshape(bs, -1)                                        # (bs, nh*dv)
            out = F.linear(y, wO)                                             # (bs, d)
            out = out.unsqueeze(1)                                            # (bs, 1, d)
            return out

        _compiled_fwd = torch.compile(_fwd,
                                      fullgraph=True,
                                      dynamic=True)   # KV length is dynamic

    # ------------------------------------------------------------------
    #  4️⃣  Run the compiled forward
    # ------------------------------------------------------------------
    out = _compiled_fwd(
        x,
        kv_cache.data,
        kv_len,
        wDQ, wUQ, wUKV, wO,
        cos_table, sin_table,
        nh, d_nope, d_rope, dkv, dv,
    )

    # ------------------------------------------------------------------
    #  5️⃣  Return output and the *updated* KV‑cache tensor
    # ------------------------------------------------------------------
    return out, kv_cache.data