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
# Helper – rotate‑half (used for RoPE)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the last dimension (used by RoPE)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ----------------------------------------------------------------------
# RoPE tables (cos / sin) – cached lazily
# ----------------------------------------------------------------------
_rope_cache = {}

def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Returns (cos, sin) tables of shape (max_seq_len, dim) in bf16.
    `dim` must be even.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta[i] = 10000^{-i/half}
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (L,1)
        idx = pos * theta[None, :]                      # (L, half)
        idx = torch.cat([idx, idx], dim=-1)            # (L, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# Triton row‑wise soft‑max (operates on a 2‑D bf16 tensor)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,               # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # ---------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # ---------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # ---------- normalize ----------
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
    # pick a reasonable block size (power‑of‑2)
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
        N=N,
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------------
# Optimised MLA forward – custom_kernel
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward pass.
    Returns
    -------
    output : torch.Tensor      # (batch, seq_len, dim) – bf16
    kv_cache_data : torch.Tensor   # updated raw kv‑cache (same tensor stored in KVCache)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Local aliases (avoid repeated attribute look‑ups)
    # --------------------------------------------------------------
    bs = config.batch_size          # e.g. 128
    sl = config.seq_len             # always 1 here
    msl = config.max_seq_len        # max sequence length
    nh = config.n_heads             # 128
    d  = config.dim                 # 7168
    dq = config.q_lora_rank         # 1536
    dkv = config.kv_lora_rank       # 512
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim          # 128

    wDQ  = config.Q_proj_down_weight             # (dq, d)
    wDKV = config.KV_proj_down_weight            # (dkv+d_rope, d)
    wUQ  = config.Q_proj_up_weight               # ((d_nope+d_rope)*nh, dq)
    wUKV = config.KV_proj_up_weight              # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                      # (d, nh*dv)

    # ------------------------------------------------------------------
    # 0️⃣ Down‑projections (Linear without bias)
    # ------------------------------------------------------------------
    # x : (bs, sl, d) → (bs, sl, dq) and (bs, sl, dkv+d_rope)
    # sl == 1, we can squeeze the seq‑dim early
    q_lora = F.linear(x, wDQ)                     # (bs, sl, dq)
    q_lora = q_lora.squeeze(1)                    # (bs, dq)

    kv_lora0 = F.linear(x, wDKV)                  # (bs, sl, dkv+d_rope)
    kv_lora0 = kv_lora0.squeeze(1).unsqueeze(1)   # (bs, 1, dkv+d_rope)

    # ------------------------------------------------------------------
    # 1️⃣ KV‑cache update (in‑place)
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora0)          # kv_lora : (bs, kv_len, dkv+d_rope)
    query_pos = kv_len - 1                        # absolute position of the current token

    # ------------------------------------------------------------------
    # 2️⃣ Up‑project queries (Q) and split NoPE / RoPE
    # ------------------------------------------------------------------
    # (bs, dq) → (bs, (d_nope+d_rope)*nh) → (bs, nh, d_nope+d_rope)
    q_up = F.linear(q_lora, wUQ)                  # (bs, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope)    # (bs, nh, d_nope+d_rope)

    if d_nope > 0:
        q_nope = q_up[..., :d_nope]              # (bs, nh, d_nope)
        q_rope_raw = q_up[..., d_nope:]          # (bs, nh, d_rope)
    else:
        q_nope = None
        q_rope_raw = q_up                       # (bs, nh, d_rope)

    # ------------------------------------------------------------------
    # 3️⃣ Split KV‑latent / KV‑RoPE (no up‑projection of the whole cache!)
    # ------------------------------------------------------------------
    kv_nope_raw = kv_lora[..., :dkv]              # (bs, kv_len, dkv)
    kv_rope_raw = kv_lora[..., dkv:]              # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣ RoPE (queries & keys) – cached cos / sin tables
    # ------------------------------------------------------------------
    if d_rope > 0:
        cos_tbl, sin_tbl = _get_rope_tables(d_rope, msl, x.device)

        # ----- query (single position) -----
        # cos/sin for the current token position
        cq = cos_tbl[query_pos].view(1, 1, d_rope)   # (1,1,d_rope)
        sq = sin_tbl[query_pos].view(1, 1, d_rope)   # (1,1,d_rope)
        q_rope = q_rope_raw * cq + _rotate_half(q_rope_raw) * sq   # (bs, nh, d_rope)

        # ----- keys (entire prefix) -----
        ck = cos_tbl[:kv_len]                         # (kv_len, d_rope)
        sk = sin_tbl[:kv_len]                         # (kv_len, d_rope)
        # broadcast batch dimension
        ck = ck.unsqueeze(0).expand(bs, -1, -1)       # (bs, kv_len, d_rope)
        sk = sk.unsqueeze(0).expand(bs, -1, -1)       # (bs, kv_len, d_rope)
        k_rope = kv_rope_raw * ck + _rotate_half(kv_rope_raw) * sk   # (bs, kv_len, d_rope)
    else:
        q_rope = q_rope_raw
        k_rope = kv_rope_raw

    # ------------------------------------------------------------------
    # 5️⃣ Latent scores (q_nope -> dkv)  – only if d_nope > 0
    # ------------------------------------------------------------------
    if d_nope > 0:
        # wK : (nh, d_nope, dkv)  – first d_nope rows of KV up‑proj weight
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
        wK = wUKV_view[:, :d_nope, :]               # (nh, d_nope, dkv)

        # q_latent : (bs, nh, dkv)  = q_nope @ wK    (per‑head)
        # Using einsum which maps nicely onto cuBLAS
        q_latent = torch.einsum('bhd, hdk -> bhk', q_nope, wK)   # (bs, nh, dkv)
    else:
        q_latent = None

    # ------------------------------------------------------------------
    # 6️⃣ Compute the two score matrices (rope & latent) and combine
    # ------------------------------------------------------------------
    # 6.1 rope scores  (bs, nh, kv_len)
    # we use torch.einsum which internally launches a batched GEMM
    scores_rope = torch.einsum('bhd,bnd->bhn', q_rope, k_rope)   # (bs, nh, kv_len)

    if d_nope > 0:
        # 6.2 latent scores  (bs, nh, kv_len)
        #   kv_nope_raw : (bs, kv_len, dkv)
        scores_lat = torch.einsum('bhk,bnk->bhn', q_latent, kv_nope_raw)  # (bs, nh, kv_len)
        scores = (scores_rope + scores_lat) * (1.0 / math.sqrt(d_nope + d_rope))
    else:
        scores = scores_rope * (1.0 / math.sqrt(d_rope))

    # ------------------------------------------------------------------
    # 7️⃣ Soft‑max (row‑wise) via Triton
    # ------------------------------------------------------------------
    B, H, S = scores.shape
    attn = _triton_softmax(scores.view(B * H, S)).view(B, H, S)   # (bs, nh, kv_len)

    # ------------------------------------------------------------------
    # 8️⃣ Weighted sum of latent vectors (Z)  –  Z = attn @ kv_nope_raw
    # ------------------------------------------------------------------
    Z = torch.einsum('bhn,bnd->bhd', attn, kv_nope_raw)          # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 9️⃣ Project Z to the value space (wV_T)
    # ------------------------------------------------------------------
    # wV_T : (nh, dkv, dv)   – second slice of KV up‑proj weight
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)                  # (nh, d_nope+dv, dkv)
    wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)            # (nh, dkv, dv)

    y_head = torch.einsum('bhd, hdf -> bhf', Z, wV_T)           # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 10️⃣ Output projection back to model dimension
    # ------------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)                    # (bs, nh*dv)
    output = F.linear(y_head_flat, wO)                           # (bs, dim)
    output = output.unsqueeze(1)                                 # (bs, 1, dim)

    # ------------------------------------------------------------------
    # Return the output and the (now‑updated) KV‑cache buffer
    # ------------------------------------------------------------------
    return output, kv_cache.data