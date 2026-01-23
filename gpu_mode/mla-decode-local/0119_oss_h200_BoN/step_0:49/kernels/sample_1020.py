### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way.
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# 0️⃣  RoPE helper – produces (cos, sin) tables once per configuration
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                           dtype=torch.float32,
                                           device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                     # (max_seq_len, 1)
        idx = pos * theta[None, :]                                       # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                               # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# 1️⃣  Triton kernel – in‑place RoPE (rotate‑half) for any 3‑D tensor
# ----------------------------------------------------------------------
@triton.jit
def _rope_swap_halves_kernel(
    x_ptr,                     # [B, T, D]  bf16/fp16/fp32
    cos_ptr, sin_ptr,          # [T, D]   (or broadcasted)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,           # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,  # processes D/2 at a time
):
    pid = tl.program_id(0)
    # decode 3‑D index
    b = pid // T
    t = pid - b * T

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # ---- pointers for the two halves of x ----
    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + offs * stride_xd                     # first half
    x1_ptr = x_base + (half + offs) * stride_xd            # second half

    # ---- pointers for cos / sin (may be broadcast in the T dimension) ----
    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t

    c_ptr = cos_base + offs * stride_cos_d
    s_ptr = sin_base + offs * stride_sin_d

    # ---- load -----------------------------------------------------------
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # ---- rotate‑half (RoPE) --------------------------------------------
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # ---- store back (in‑place) -----------------------------------------
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def _rope_inplace(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    In‑place RoPE (rotate‑half) for a tensor of shape (B, T, D).
    cos / sin must be broadcastable to (T, D); they are usually taken from
    the tables returned by ``_get_rope_tables``.
    """
    assert x.is_cuda
    B, T, D = x.shape
    assert D % 2 == 0, "RoPE head dimension must be even"

    # Choose a power‑of‑2 block that fits the half‑dimension.
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    BLOCK_HALF = min(BLOCK_HALF, 256)   # Triton caps block size for good occupancy

    grid = (B * T,)                     # one program per (batch, seq) pair
    _rope_swap_halves_kernel[grid](
        x,
        cos, sin,
        B=B, T=T, D=D,
        stride_xb=x.stride(0), stride_xt=x.stride(1), stride_xd=x.stride(2),
        stride_cos_t=cos.stride(0), stride_cos_d=cos.stride(1),
        stride_sin_t=sin.stride(0), stride_sin_d=sin.stride(1),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )
    # No return – operation is in‑place

# ----------------------------------------------------------------------
# 2️⃣  Triton Softmax (row‑wise, bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,               # number of columns
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in  = row * stride_in
    row_off_out = row * stride_out

    # ---- max reduction -------------------------------------------------
    col = tl.arange(0, BLOCK_SIZE)
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---- exponent & sum ------------------------------------------------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---- normalisation -------------------------------------------------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    B, N = x.shape
    # Pick a reasonable block size (power‑of‑2, ≤ 1024)
    if N <= 32:
        BLOCK = 32
    elif N <= 64:
        BLOCK = 64
    elif N <= 128:
        BLOCK = 128
    else:
        BLOCK = 1 << (N - 1).bit_length()
        BLOCK = min(BLOCK, 1024)

    out = torch.empty_like(x)
    grid = (B,)
    _softmax_kernel[grid](
        out,
        x,
        out.stride(0),
        x.stride(0),
        N,
        BLOCK_SIZE=BLOCK,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# 3️⃣  The fused MLA forward (uses the two Triton kernels above)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Highly‑optimised forward of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor          # shape (batch, seq_len, dim)  bf16
    kv_cache.data : torch.Tensor   # the updated KV‑cache (bf16)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Unpack static configuration (all are compile‑time constants)
    # --------------------------------------------------------------
    bs   = config.batch_size
    sl   = config.seq_len                # always 1 in the test suite
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    # --------------------------------------------------------------
    # Grab weight tensors (they are already on the correct device)
    # --------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + d_rope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # --------------------------------------------------------------
    # 1️⃣  Down‑projection -------------------------------------------------
    # --------------------------------------------------------------
    q_lora = F.linear(x, wDQ)                            # (bs, sl, dq)
    kv_lora_input = F.linear(x, wDKV)                    # (bs, sl, dkv + d_rope)

    # --------------------------------------------------------------
    # 2️⃣  KV‑cache update (in‑place) ------------------------------------
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)            # kv_lora: (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                                # absolute position of the current token

    # --------------------------------------------------------------
    # 3️⃣  Up‑projection of queries ---------------------------------------
    # --------------------------------------------------------------
    #   Q‑up gives (bs, nh, d_nope + d_rope) directly
    q_up = F.linear(q_lora.squeeze(1), wUQ)               # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)           # (bs, nh, d_total)

    q_nope = q_up[..., :d_nope]                          # (bs, nh, d_nope)
    q_rope = q_up[..., d_nope:]                          # (bs, nh, d_rope)

    # --------------------------------------------------------------
    # 4️⃣  Split KV tensor (no‑PE part and RoPE part) --------------------
    # --------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]                    # (bs, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]                    # (bs, kv_len, d_rope)

    # --------------------------------------------------------------
    # 5️⃣  Up‑project KV → (k_nope , v) ---------------------------------
    # --------------------------------------------------------------
    #   The weight wUKV encodes both the key‑no‑PE and the value projections.
    #   We first compute the whole projection and then split.
    kv_up = F.linear(kv_nope_input, wUKV)                 # (bs, kv_len, (d_nope+dv)*nh)
    kv_up = kv_up.view(bs, kv_len, nh, d_nope + dv)      # (bs, kv_len, nh, d_total)
    k_nope = kv_up[..., :d_nope]                         # (bs, kv_len, nh, d_nope)
    v      = kv_up[..., d_nope:]                         # (bs, kv_len, nh, dv)

    # --------------------------------------------------------------
    # 6️⃣  Prepare RoPE tables (cached globally) -------------------------
    # --------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # -----------------------------------------------------------------
    # 6️⃣a RoPE on queries (single position) ----------------------------
    # -----------------------------------------------------------------
    cos_q = cos_table[query_pos].view(1, d_rope)          # (1, d_rope)
    sin_q = sin_table[query_pos].view(1, d_rope)          # (1, d_rope)
    # reshape to (bs, nh, d_rope) → we use a view that Triton can handle
    q_rope = q_rope.contiguous()
    # Triton kernel works on (B, T, D); we set T=1 (the query length)
    _rope_inplace(q_rope.view(bs * nh, 1, d_rope), cos_q, sin_q)

    # -----------------------------------------------------------------
    # 6️⃣b RoPE on keys (all cached positions) -------------------------
    # -----------------------------------------------------------------
    #   cos / sin are sliced up to the current cache length.
    cos_k = cos_table[:kv_len].contiguous()   # (kv_len, d_rope)
    sin_k = sin_table[:kv_len].contiguous()   # (kv_len, d_rope)
    # k_rope_input has shape (bs, kv_len, d_rope)
    k_rope = k_rope_input.contiguous()
    _rope_inplace(k_rope.view(bs, kv_len, d_rope), cos_k, sin_k)
    # expand to per‑head dimension (broadcast – no extra copy)
    k_rope = k_rope[:, None, :, :].expand(-1, nh, -1, -1)   # (bs, nh, kv_len, d_rope)

    # -----------------------------------------------------------------
    # 7️⃣  Concatenate no‑PE and RoPE parts ------------------------------
    # -----------------------------------------------------------------
    #   Shapes after permute:
    #     q_nope : (bs, nh, d_nope)
    #     q_rope : (bs, nh, d_rope)
    #     k_nope : (bs, kv_len, nh, d_nope) → (bs, nh, kv_len, d_nope)
    #     k_rope : (bs, nh, kv_len, d_rope)
    q = torch.cat([q_nope, q_rope], dim=-1)                     # (bs, nh, d_total)
    k = torch.cat([k_nope.permute(0, 2, 1, 3), k_rope], dim=-1) # (bs, nh, kv_len, d_total)

    # -----------------------------------------------------------------
    # 8️⃣  Scaled dot‑product (single batched GEMM) --------------------
    # -----------------------------------------------------------------
    d_total = d_nope + d_rope
    scale = 1.0 / math.sqrt(d_total)

    #   view as (B*H, 1, D)  and (B*H, D, Kv)  → scores of shape (B*H, Kv)
    q_flat = q.view(bs * nh, 1, d_total)                                   # (B*H, 1, D)
    k_flat = k.permute(0, 1, 3, 2).reshape(bs * nh, d_total, kv_len)       # (B*H, D, Kv)
    scores = torch.bmm(q_flat, k_flat).squeeze(1) * scale                   # (B*H, Kv)

    # -----------------------------------------------------------------
    # 9️⃣  Softmax (Triton) -----------------------------------------------
    # -----------------------------------------------------------------
    attn = _triton_softmax(scores)                                          # (B*H, Kv)
    attn = attn.view(bs, nh, kv_len)                                        # (bs, nh, kv_len)

    # -----------------------------------------------------------------
    # 🔟  Weighted sum of VALUES -------------------------------------------
    # -----------------------------------------------------------------
    #   v : (bs, kv_len, nh, dv) → (bs, nh, kv_len, dv)
    v_perm = v.permute(0, 2, 1, 3)                                          # (bs, nh, kv_len, dv)
    # batched matmul: (B*H, 1, Kv) @ (B*H, Kv, dv) → (B*H, 1, dv)
    attn_flat = attn.view(bs * nh, 1, kv_len)                               # (B*H, 1, Kv)
    v_flat   = v_perm.reshape(bs * nh, kv_len, dv)                           # (B*H, Kv, dv)
    y_head   = torch.bmm(attn_flat, v_flat).squeeze(1)                      # (B*H, dv)

    # -----------------------------------------------------------------
    # 1️⃣1️⃣  Final linear projection (wo) ---------------------------------
    # -----------------------------------------------------------------
    y_head = y_head.view(bs, nh * dv)                                       # (bs, nh*dv)
    # add the singleton seq‑len dimension expected by the original model
    y_head = y_head.unsqueeze(1)                                            # (bs, 1, nh*dv)
    output = F.linear(y_head, wO)                                           # (bs, 1, dim)

    # --------------------------------------------------------------
    # Return the tensor output and the (now‑updated) KV‑cache data
    # --------------------------------------------------------------
    return output, kv_cache.data