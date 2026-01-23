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
# 0️⃣  Cached cosine / sine tables for RoPE (global LRU cache)
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bf16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )                                   # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
        idx = pos * theta[None, :]          # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1) # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 1️⃣  RoPE kernels (in‑place for queries, out‑of‑place for keys)
# ----------------------------------------------------------------------
@triton.jit
def _rope_swap_halves_kernel(
    x_ptr, out_ptr,               # x and out: [B,T,D] bf16
    cos_ptr, sin_ptr,             # [T,D] tables (broadcast over B)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,              # even
    stride_xb, stride_xt, stride_xd,
    stride_ob, stride_ot, stride_od,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    bt = pid
    b = bt // T
    t = bt - b * T

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # ----- load x (first half / second half) -----
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + offs * stride_xd                # first half
    x1_ptr = x_base + (half + offs) * stride_xd       # second half
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)

    # ----- load cos / sin -----
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t
    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d
    c = tl.load(c_ptr, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr, mask=mask, other=0.0).to(tl.float32)

    # ----- rotate‑half (out0 = x0*c - x1*s; out1 = x1*c + x0*s) -----
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # ----- store result (either back‑in‑place or into out_ptr) -----
    out_base = out_ptr + b * stride_ob + t * stride_ot
    out0_ptr = out_base + offs * stride_od
    out1_ptr = out_base + (half + offs) * stride_od
    tl.store(out0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(out1_ptr, out1.to(tl.bfloat16), mask=mask)

def _rope_inplace_query(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE in‑place to a tensor of shape (B, H, D)."""
    assert q.is_cuda and q.dtype == torch.bfloat16
    B, H, D = q.shape
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    grid = (B * H,)
    _rope_swap_halves_kernel[grid](
        q, q,                     # in‑place
        cos, sin,
        B=B, T=H, D=D,
        stride_xb=q.stride(0), stride_xt=q.stride(1), stride_xd=q.stride(2),
        stride_ob=q.stride(0), stride_ot=q.stride(1), stride_od=q.stride(2),
        stride_cos_t=0, stride_cos_d=cos.stride(0),
        stride_sin_t=0, stride_sin_d=sin.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )

def _rope_out_of_place_key(key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Return rotated keys. `key` shape = (B, T, D)."""
    B, T, D = key.shape
    assert D % 2 == 0
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    out = torch.empty_like(key)
    grid = (B * T,)
    _rope_swap_halves_kernel[grid](
        key, out,
        cos, sin,
        B=B, T=T, D=D,
        stride_xb=key.stride(0), stride_xt=key.stride(1), stride_xd=key.stride(2),
        stride_ob=out.stride(0), stride_ot=out.stride(1), stride_od=out.stride(2),
        stride_cos_t=cos.stride(0), stride_cos_d=cos.stride(1),
        stride_sin_t=sin.stride(0), stride_sin_d=sin.stride(1),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# 2️⃣  Triton softmax (row‑wise, bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # ---- max reduction ----
    max_val = tl.full([BLOCK_M], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_M)
    for start in range(0, N_COLS, BLOCK_M):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---- exp & sum ----
    sum_val = tl.full([BLOCK_M], 0.0, tl.float32)
    for start in range(0, N_COLS, BLOCK_M):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---- normalize ----
    for start in range(0, N_COLS, BLOCK_M):
        cur = start + col
        mask = cur < N_COLS
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
    # choose a power‑of‑2 block size (capped at 1024)
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
        N_COLS=n_cols,
        BLOCK_M=BLOCK,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# 3️⃣  Main custom kernel (MLA forward)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of Multi‑head Latent Attention (MLA).
    Returns
    -------
    output   : torch.Tensor   # (batch, seq_len=1, dim) , bf16
    kv_data  : torch.Tensor   # new KV‑cache tensor
    """
    config, x, kv_cache = data
    device = x.device

    # ------------------------------------------------------------------
    # Unpack configuration
    # ------------------------------------------------------------------
    bs   = config.batch_size                     # 128
    sl   = config.seq_len                        # always 1
    nh   = config.n_heads                        # 128
    dim  = config.dim                            # 7168
    dq   = config.q_lora_rank                    # 1536
    dkv  = config.kv_lora_rank                   # 512
    d_nope = config.qk_nope_head_dim            # e.g. 64 (provided by config)
    d_rope = config.qk_rope_head_dim            # 64
    dv   = config.v_head_dim                     # 128
    msl  = config.max_seq_len                    # 8192

    # ------------------------------------------------------------------
    # Lazy‑fusion of static weight matrices (executed only once)
    # ------------------------------------------------------------------
    # 1) fused Q up‑projection (Q_proj_up ∘ Q_proj_down)
    if not hasattr(config, "_wQU_fused"):
        # ((d_nope+d_rope)*nh , dim)  =  (Q_up_weight @ Q_down_weight)
        config._wQU_fused = torch.matmul(config.Q_proj_up_weight, config.Q_proj_down_weight)
    wQU = config._wQU_fused                     # (nh*(d_nope+d_rope), dim)

    # 2) split KV‑up weight into (no‑PE) & value parts and fuse value+output
    if not hasattr(config, "_wK"):
        # view shape (nh, d_nope+dv, dkv)
        wUKV_view = config.KV_proj_up_weight.view(nh, d_nope + dv, dkv)
        config._wK = wUKV_view[:, :d_nope, :]                # (nh, d_nope, dkv)
        wV = wUKV_view[:, d_nope:, :]                       # (nh, dv, dkv)

        # fuse value projection with final output projection:
        # wO : (dim, nh*dv)  -> reshape to (dim, nh, dv)
        wO_view = config.wo_weight.view(dim, nh, dv)        # (dim, nh, dv)
        # bring WV to (nh, dkv, dv) so we can do a batched matmul
        wV_T = wV.permute(0, 2, 1)                          # (nh, dkv, dv)
        wO_T = wO_view.permute(1, 2, 0)                     # (nh, dv, dim)
        # (nh, dkv, dim)
        config._wVO = torch.bmm(wV_T, wO_T)                 # fused weight
        config._wK = config._wK
    wK   = config._wK          # (nh, d_nope, dkv)
    wVO  = config._wVO         # (nh, dkv, dim)

    # ------------------------------------------------------------------
    # 1️⃣ Down‑projection & KV‑cache update
    # ------------------------------------------------------------------
    # Q down‑proj is now fused with up‑proj (see wQU)
    # KV down‑proj (still needed for cache)
    kv_lora_input = F.linear(x, config.KV_proj_down_weight)   # (bs, 1, dkv + d_rope)
    kv_lora, kv_len = kv_cache(kv_lora_input)                # kv_lora: (bs, kv_len, dkv+d_rope)

    # ------------------------------------------------------------------
    # 2️⃣ Q up‑projection (fused) and split into No‑PE / RoPE parts
    # ------------------------------------------------------------------
    # x : (bs,1,dim)  -> (bs, nh, d_nope+d_rope)
    q_all = F.linear(x, wQU)                  # (bs, 1, nh*(d_nope+d_rope))
    q_all = q_all.view(bs, nh, d_nope + d_rope)  # (bs, nh, d_total)
    q_nope = q_all[..., :d_nope]              # (bs, nh, d_nope)
    q_rope = q_all[..., d_nope:]              # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 3️⃣ Split KV into latent (no‑PE) and RoPE parts
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]         # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]          # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣ RoPE for queries (in‑place) and keys (out‑of‑place)
    # ------------------------------------------------------------------
    # pre‑compute cosine / sine tables once per config
    cos_tab, sin_tab = _get_rope_tables(d_rope, msl, device)

    # query side – single position (absolute position = kv_len-1)
    qpos = kv_len - 1
    cos_q = cos_tab[qpos]                     # (d_rope,)
    sin_q = sin_tab[qpos]                     # (d_rope,)
    _rope_inplace_query(q_rope, cos_q, sin_q)  # q_rope mutated in‑place

    # key side – whole cache
    cos_k = cos_tab[:kv_len]                  # (kv_len, d_rope)
    sin_k = sin_tab[:kv_len]                  # (kv_len, d_rope)
    k_rope = _rope_out_of_place_key(k_rope_input, cos_k, sin_k)   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣ Compute attention scores (latent + RoPE)
    # ------------------------------------------------------------------
    # latent part: q_nope (bs,nh,d_nope) -> latent space via wK (nh,d_nope,dkv)
    # result q_nope_latent : (bs, nh, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)       # (bs, nh, dkv)

    # scores from latent part
    scores_nope = torch.einsum('bhd,bld->bhl', q_nope_latent, kv_nope_input)  # (bs, nh, kv_len)

    # scores from RoPE part
    scores_rope = torch.einsum('bhd,bld->bhl', q_rope, k_rope)                # (bs, nh, kv_len)

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale

    # ------------------------------------------------------------------
    # 6️⃣ Softmax (Triton) → attention weights
    # ------------------------------------------------------------------
    scores_flat = scores.view(bs * nh, kv_len)           # (B*H, Kv)
    attn_flat = _triton_softmax(scores_flat)            # (B*H, Kv)  bf16
    attn = attn_flat.view(bs, nh, kv_len)               # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 7️⃣ Weighted sum of latent keys  (M = Σ attn * kv_nope_input)
    # ------------------------------------------------------------------
    M = torch.einsum('bhl,bld->bhd', attn, kv_nope_input)   # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 8️⃣ Final projection (M @ wVO) – l fused value + output projection
    # ------------------------------------------------------------------
    # wVO : (nh, dkv, dim)
    output = torch.einsum('bhd,hdk->bk', M, wVO)        # (bs, dim)
    output = output.unsqueeze(1)                       # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data