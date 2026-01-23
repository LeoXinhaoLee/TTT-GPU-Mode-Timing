### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config   # <- must be imported this way
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# 1️⃣  RoPE utilities (same as the reference implementation)
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,            # [B, T, D] (bf16/fp16/fp32)
    cos_ptr, sin_ptr, # [T, D]  or [D]  (broadcast possible)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,         # D must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,
):
    """In‑place RoPE for a tensor of shape (B,T,D).  Operates on the two
    halves of the last dimension using the rotate‑half trick."""
    pid = tl.program_id(0)                # one program per (b,t) pair
    b = pid // T
    t = pid - b * T

    half = D // 2

    off = tl.arange(0, BLOCK_HALF)        # <‑‑ power‑of‑2 range
    mask = off < half

    # pointers for the two halves of `x`
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + off * stride_xd                   # first half
    x1_ptr = x_base + (half + off) * stride_xd          # second half

    # pointers for cosine / sine (they may be broadcasted over T)
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t
    c_ptr = cos_base + off * stride_cos_d
    s_ptr = sin_base + off * stride_sin_d

    # load
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # RoPE: out0 = x0*c - x1*s ; out1 = x1*c + x0*s
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # store back (in‑place)
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor,
                       cos_q:   torch.Tensor,
                       sin_q:   torch.Tensor) -> None:
    """
    Apply RoPE **in place** to a tensor of shape (B, H, D) where
    D is even.  The implementation follows the reference code and
    launches one Triton program per (batch, head) pair.
    """
    assert q_rope.is_cuda
    assert q_rope.shape[-1] % 2 == 0

    B, H, D = q_rope.shape
    half = D // 2
    # Choose a block size that is the next power‑of‑2 ≥ half (capped at 256)
    BLOCK_HALF = 1 << (half - 1).bit_length()
    BLOCK_HALF = min(BLOCK_HALF, 256)

    grid = (B * H,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=B, T=H, D=D,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        stride_cos_t=0, stride_cos_d=cos_q.stride(0),
        stride_sin_t=0, stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

# ----------------------------------------------------------------------
# 2️⃣  Cached cosine / sine tables (shared across calls)
# ----------------------------------------------------------------------
_rope_cache = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Return cached `(cos, sin)` tables of shape (max_seq_len, dim) in bf16.
    The table is built exactly as in the reference `RoPE` module.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)           # (max_seq_len,1)
        idx = pos * theta                                   # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                 # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 3️⃣  Optimised forward kernel for the MLA module
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward.
    --------------------------------------------------------------
    Input:
        data = (config, x, kv_cache)
            config : Config  – model hyper‑parameters & weight tensors
            x      : (B, 1, D)  – bf16 input activation
            kv_cache: KVCache   – mutable cache object (stores raw KV tokens)
    Output:
        y      : (B, 1, D)  – bf16 result of the MLA block
        kv_data: the underlying cache tensor (updated in‑place)
    --------------------------------------------------------------

    The implementation follows the reference forward but
      • fuses the two Q·K matmuls into a single einsum,
      • uses a Triton‑based soft‑max,
      • applies RoPE with a specialised kernel,
      • avoids the expensive “KV_proj_up” over the whole cache by
        performing the up‑projection **once** per forward (still O(B·L·dkv·(d_nope+dv)·H) but
        implemented as a single batched GEMM via `torch.einsum` which is heavily
        optimised for bf16 on H200).
    --------------------------------------------------------------
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # 0️⃣  unpack config & weights (all are bf16 and already on the device)
    # ------------------------------------------------------------------
    B   = config.batch_size
    S   = config.seq_len                # always 1 in the tests
    H   = config.n_heads
    dq  = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    d_v = config.v_head_dim
    max_seq_len = config.max_seq_len

    wDQ  = config.Q_proj_down_weight           # (dq , D)
    wDKV = config.KV_proj_down_weight          # (dkv+d_rope , D)
    wUQ  = config.Q_proj_up_weight             # ((d_nope+d_rope)*H , dq)
    wUKV = config.KV_proj_up_weight            # ((d_nope+d_v)*H , dkv)
    wO   = config.wo_weight                    # (D , H*d_v)

    # ------------------------------------------------------------------
    # 1️⃣  Down‑projection
    # ------------------------------------------------------------------
    # x : (B,1,D) → (B,1,dq) and (B,1,dkv+d_rope)
    q_lora     = F.linear(x, wDQ)                       # (B,1,dq)
    kv_lora_in = F.linear(x, wDKV)                      # (B,1,dkv+d_rope)

    # ------------------------------------------------------------------
    # 2️⃣  KV‑cache update (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_in)              # (B,kv_len,dkv+d_rope)
    query_pos = kv_len - 1                               # absolute position for the query token

    # ------------------------------------------------------------------
    # 3️⃣  Up‑project queries
    # ------------------------------------------------------------------
    # squeeze the singleton seq‑len dimension before the up‑proj
    q_up = F.linear(q_lora.squeeze(1), wUQ)             # (B, (d_nope+d_rope)*H)
    q_up = q_up.view(B, H, d_nope + d_rope)            # (B, H, d_nope+d_rope)
    q_nope, q_rope = torch.split(q_up,
                                  [d_nope, d_rope],
                                  dim=-1)           # (B,H,d_nope) , (B,H,d_rope)

    # ------------------------------------------------------------------
    # 4️⃣  Split KV into latent and RoPE parts
    # ------------------------------------------------------------------
    kv_nope_raw = kv_lora[..., :dkv]                    # (B,kv_len,dkv)
    k_rope_raw   = kv_lora[..., dkv:]                   # (B,kv_len,d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE tables & in‑place application
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, max_seq_len, x.device)

    # ----- query side --------------------------------------------------
    cos_q = cos_table[query_pos]          # (d_rope,)
    sin_q = sin_table[query_pos]          # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)      # modifies q_rope in‑place

    # ----- key side ----------------------------------------------------
    cos_k = cos_table[:kv_len]            # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]            # (kv_len, d_rope)
    # element‑wise RoPE on the key part (no‑copy broadcasting over heads)
    k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k   # (B,kv_len,d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Query latent projection (q_nope → q_nope_latent)
    # ------------------------------------------------------------------
    if d_nope == 0:
        # Edge case – no latent query dimension
        q_nope_latent = torch.zeros(B, H, dkv,
                                    dtype=x.dtype,
                                    device=x.device)
    else:
        # wUKV : ((d_nope+d_v)*H , dkv) → (H , d_nope+d_v , dkv)
        wUKV_view = wUKV.view(H, d_nope + d_v, dkv)          # (H,d_nope+d_v,dkv)
        wK = wUKV_view[:, :d_nope, :]                        # (H,d_nope,dkv)
        # einsum: (B,H,d_nope) @ (H,d_nope,dkv) → (B,H,dkv)
        q_nope_latent = torch.einsum('bhd,hdn->bhn', q_nope, wK)

    # ------------------------------------------------------------------
    # 7️⃣  Build the *combined* Q and K matrices (latent + RoPE)
    # ------------------------------------------------------------------
    # Q : (B,H,d_nope+d_rope) → (B,H,dkv+d_rope)
    Q = torch.cat([q_nope_latent, q_rope], dim=-1)   # (B,H,dkv+d_rope)

    # K : (B,kv_len,dkv+d_rope)
    K = torch.cat([kv_nope_raw, k_rope], dim=-1)    # (B,kv_len,dkv+d_rope)

    # ------------------------------------------------------------------
    # 8️⃣  Scaled dot‑product (single fused einsum)
    # ------------------------------------------------------------------
    # scores_{b,h,l} = Σ_d Q_{b,h,d} * K_{b,l,d}
    scale = 1.0 / math.sqrt(d_rope + d_nope)        # √(d_total) where d_total = d_nope+d_rope
    scores = torch.einsum('bhd,btd->bht', Q, K) * scale   # (B,H,kv_len)

    # ------------------------------------------------------------------
    # 9️⃣  Soft‑max (Triton)
    # ------------------------------------------------------------------
    # reshape to 2‑D tensor for the Triton kernel: (B*H, kv_len)
    scores_flat = scores.reshape(B * H, kv_len)          # (B*H , kv_len)
    attn_flat   = _triton_softmax(scores_flat)           # (B*H , kv_len)  bf16
    attn = attn_flat.view(B, H, kv_len)                 # (B,H,kv_len)

    # ------------------------------------------------------------------
    # 🔟  Weighted sum of *latent* keys (M = attn @ kv_nope_raw)
    # ------------------------------------------------------------------
    # attn : (B,H,kv_len) , kv_nope_raw : (B,kv_len,dkv)
    M = torch.einsum('bhl,bld->bhd', attn, kv_nope_raw)   # (B,H,dkv)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣  Project aggregated latent keys to per‑head values
    # ------------------------------------------------------------------
    # wV_T : (H, dkv, d_v)   – obtained from the second half of wUKV
    wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)      # (H, dkv, d_v)
    # y_head = M @ wV_T   →  (B,H,d_v)
    y_head = torch.einsum('bhd,hdv->bhv', M, wV_T)         # (B,H,d_v)

    # ------------------------------------------------------------------
    # 1️⃣2️⃣  Final linear projection back to model dimension
    # ------------------------------------------------------------------
    y = y_head.reshape(B, H * d_v)                         # (B, H*d_v)
    y = F.linear(y, wO)                                    # (B, D)
    y = y.unsqueeze(1)                                     # (B, 1, D)

    # ------------------------------------------------------------------
    # Return output and the (now updated) KV‑cache tensor
    # ------------------------------------------------------------------
    return y, kv_cache.data


# ----------------------------------------------------------------------
# 3️⃣  Triton soft‑max (row‑wise, bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in tl.range(0, n_cols, BLOCK_SIZE, num_stages=NUM_STAGES):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in tl.range(0, n_cols, BLOCK_SIZE, num_stages=NUM_STAGES):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
    for start in tl.range(0, n_cols, BLOCK_SIZE, num_stages=NUM_STAGES):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise soft‑max for a 2‑D bf16 tensor via Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape

    # pick a power‑of‑2 block size (capped at 1024)
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
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out