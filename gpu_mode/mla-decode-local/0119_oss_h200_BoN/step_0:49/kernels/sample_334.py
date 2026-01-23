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
#  RoPE utilities -------------------------------------------------------
# ----------------------------------------------------------------------
_rope_cache = {}

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return cached (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(torch.bfloat16)  # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)                      # (max_seq_len, 1)
        idx = pos * theta[None, :]                     # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)           # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Triton kernel for in‑place RoPE on the queries -----------------------
# ----------------------------------------------------------------------
@triton.jit
def rope_swap_halves_kernel(
    x_ptr,                      # [B, T, D] bf16/fp16/fp32
    cos_ptr, sin_ptr,            # [D] (broadcasted)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,             # must be even
    stride_xb, stride_xt, stride_xd,
    stride_cos, stride_sin,
    BLOCK_HALF: tl.constexpr,    # processes D/2 in blocks
):
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T

    half = D // 2
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half

    # base address for the token
    x_base = x_ptr + b * stride_xb + t * stride_xt
    # first half of the vector
    x0_ptr = x_base + offs * stride_xd
    # second half of the vector
    x1_ptr = x_base + (half + offs) * stride_xd

    # load halves
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)

    # load cos / sin (broadcasted over the head dimension)
    c = tl.load(cos_ptr + offs * stride_cos, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + offs * stride_sin, mask=mask, other=0.0).to(tl.float32)

    # RoPE (rotate‑half) : out0 = x0*c - x1*s ; out1 = x1*c + x0*s
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # store back
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    """
    In‑place RoPE for queries.
    q_rope : (B, H, D)  bf16,  D even
    cos_q / sin_q : (D,)  bf16
    """
    assert q_rope.is_cuda
    bs, nh, d = q_rope.shape
    assert d % 2 == 0

    half = d // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    BLOCK_HALF = min(BLOCK_HALF, 256)

    grid = (bs * nh,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=bs, T=nh, D=d,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        stride_cos=cos_q.stride(0),
        stride_sin=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )
# ----------------------------------------------------------------------
#  Optimised MLA forward ------------------------------------------------
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi‑head Latent Attention (MLA) module.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim), bf16
    kv_cache_tensor : torch.Tensor   # updated KV‑cache tensor (B, max_seq_len, d_kv+d_rope)
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Unpack configuration (readability)
    # ------------------------------------------------------------------
    bs   = config.batch_size                 # batch size
    sl   = config.seq_len                    # =1 during inference
    nh   = config.n_heads
    d    = config.dim
    dq   = config.q_lora_rank               # Q‑LoRA rank
    dkv  = config.kv_lora_rank              # KV‑LoRA rank
    d_nope = config.qk_nope_head_dim        # size of the “no‑PE’’ part in Q/K
    d_rope = config.qk_rope_head_dim        # size of the RoPE part in Q/K
    dv   = config.v_head_dim                # value dimension per head
    msl  = config.max_seq_len

    # ------------------------------------------------------------------
    # Extract weight tensors (already on device & bf16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight                     # (dq, dim)
    wDKV  = config.KV_proj_down_weight                    # (dkv + d_rope, dim)
    wUQ   = config.Q_proj_up_weight                       # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight                      # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                              # (dim, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣ Down‑project
    # ------------------------------------------------------------------
    # (B, 1, dq)
    q_lora = F.linear(x, wDQ)
    # (B, 1, dkv + d_rope)
    kv_lora_input = F.linear(x, wDKV)

    # ------------------------------------------------------------------
    # 2️⃣ Update KV‑cache (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_input)   # kv_lora: (B, kv_len, dkv+d_rope)
    kv_len = int(kv_len)                        # python int for indexing
    query_pos = kv_len - 1                       # absolute position of the new token

    # ------------------------------------------------------------------
    # 3️⃣ Up‑project queries
    # ------------------------------------------------------------------
    # (B, (d_nope+d_rope)*nh)
    q_up = F.linear(q_lora.squeeze(1), wUQ)
    # (B, nh, d_nope+d_rope)
    q_up = q_up.view(bs, nh, d_nope + d_rope)

    q_nope = q_up[..., :d_nope]          # (B, nh, d_nope)
    q_rope = q_up[..., d_nope:]          # (B, nh, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣ Split KV into latent and RoPE parts
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]          # (B, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]          # (B, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣ RoPE for queries (in‑place, Triton)
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]               # (d_rope,)
    sin_q = sin_table[query_pos]               # (d_rope,)
    rope_inplace_query(q_rope, cos_q, sin_q)   # q_rope now contains RoPE‑encoded values

    # ------------------------------------------------------------------
    # 6️⃣ RoPE for keys (vectorised PyTorch – cheap compared to attention)
    # ------------------------------------------------------------------
    cos_k = cos_table[:kv_len]                 # (kv_len, d_rope)
    sin_k = sin_table[:kv_len]                 # (kv_len, d_rope)

    # broadcast to batch dimension
    cos_k = cos_k[None, :, :]                  # (1, kv_len, d_rope)
    sin_k = sin_k[None, :, :]                  # (1, kv_len, d_rope)

    # rotate‑half + apply sin/cos
    k_rotated = _rotate_half(k_rope_input)    # (B, kv_len, d_rope)
    k_rope = k_rope_input * cos_k + k_rotated * sin_k   # (B, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 7️⃣ Project the “no‑PE’’ query part into the latent space
    # ------------------------------------------------------------------
    # reshape up‑projection weight for easy indexing
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)          # (nh, d_nope+dv, dkv)

    # (nh, d_nope, dkv)
    wK = wUKV_view[:, :d_nope, :]

    # q_nope: (B, nh, d_nope)  ;  wK: (nh, d_nope, dkv)
    # result: (B, nh, dkv)
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # ------------------------------------------------------------------
    # 8️⃣  Concatenate latent & RoPE parts → full Q / K for attention
    # ------------------------------------------------------------------
    # Q : (B, nh, dkv + d_rope)
    q_full = torch.cat([q_nope_latent, q_rope], dim=-1)   # (B, nh, d_total)

    # K : (B, kv_len, dkv + d_rope)
    k_full = torch.cat([kv_nope_input, k_rope], dim=-1)   # (B, kv_len, d_total)

    # ------------------------------------------------------------------
    # 9️⃣  Scaled‑dot‑product attention (FlashAttention via torch.nn.functional)
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)   # same scaling as original code

    # reshape Q for SDPA: (B, nh, 1, d_total)
    q_full = q_full.unsqueeze(2)

    # broadcast K and V across the head dimension (no extra memory copy)
    k_exp = k_full.unsqueeze(1).expand(-1, nh, -1, -1)          # (B, nh, kv_len, d_total)
    v_exp = kv_nope_input.unsqueeze(1).expand(-1, nh, -1, -1)   # (B, nh, kv_len, dkv)

    # flash‑attention: returns (B, nh, 1, dkv)
    attn_out = F.scaled_dot_product_attention(
        q_full, k_exp, v_exp,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )   # (B, nh, 1, dkv)

    # squeeze sequence dimension → (B, nh, dkv)
    M = attn_out.squeeze(2)

    # ------------------------------------------------------------------
    # 🔟  Project aggregated latent keys to per‑head values
    # ------------------------------------------------------------------
    # wV : (nh, dv, dkv)   → transpose to (nh, dkv, dv) for einsum
    wV = wUKV_view[:, d_nope:, :]          # (nh, dv, dkv)
    wV_T = wV.permute(0, 2, 1)            # (nh, dkv, dv)

    # (B, nh, dv)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)

    # ------------------------------------------------------------------
    # 1️⃣1️⃣ Merge heads & final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, -1)            # (B, nh*dv)
    y = y.unsqueeze(1)                    # (B, 1, nh*dv)
    output = F.linear(y, wO)              # (B, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the updated KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data