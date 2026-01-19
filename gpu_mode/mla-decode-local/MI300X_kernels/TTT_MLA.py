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
# Global caches (shared across kernel calls)
# ----------------------------------------------------------------------
_cached_cos = None          # (max_seq_len, rope_dim)  bfloat16
_cached_sin = None          # (max_seq_len, rope_dim)  bfloat16
_rope_cache = {}           # (dim, max_seq_len, device) -> (cos, sin)

_combined_qkv_weight = {}   # (id(wUQ), id(wDQ), id(wDKV)) -> fused weight (not used in the new fast path)
_wV_T_cache = {}            # (id(wUKV), d_nope) -> (nh, dkv, dv) tensor

_combined_latent_to_output_weight = {}   # (id(wUKV), id(wO)) -> (nh, dkv, dim) (used by generic fallback)

_wO_T_cache = {}            # (id(wO)) -> wO.T (contiguous)

_compiled_forward = None    # compiled generic forward (fallback)

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables for rotary embeddings, cached globally."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half,
                                          dtype=torch.float32,
                                          device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                         # (max_seq_len, 1)
        idx = pos * theta                                                  # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16),
                            idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


# ----------------------------------------------------------------------
# Triton GEMM – used for the large output projection (y_head_flat @ wO.T)
# ----------------------------------------------------------------------
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,                     # pointers
    M, N, K,                                 # problem size
    stride_am, stride_ak,                    # strides for A
    stride_bk, stride_bn,                    # strides for B
    stride_cm, stride_cn,                    # strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am
                          + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0)
        a = a.to(tl.float32)

        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk
                          + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm
                      + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs,
             acc.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


def _matmul_bf16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A : (M, K)   bfloat16
    B : (K, N)   bfloat16
    returns C : (M, N) bfloat16
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

    # tile sizes – tuned for A100/H200
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _matmul_kernel[grid](
        A,
        B,
        C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return C


# ----------------------------------------------------------------------
# Generic forward (compiled with torch.compile) – fallback for d_nope > 0
# ----------------------------------------------------------------------
def _build_compiled_forward():
    """Build the generic (no‑PE) MLA forward with torch‑compile."""
    import torch.nn.functional as F

    def _inner(x: torch.Tensor,
               kv_data: torch.Tensor,
               cur_len: int,
               cos_tbl: torch.Tensor,
               sin_tbl: torch.Tensor,
               wDQ: torch.Tensor,
               wDKV: torch.Tensor,
               wUQ: torch.Tensor,
               wUKV: torch.Tensor,
               wO: torch.Tensor,
               nh: int,
               d_nope: int,
               d_rope: int,
               dkv: int,
               dv: int):
        # -------------------------- 1️⃣ down‑project --------------------------
        q_lora = F.linear(x, wDQ)               # (bs, sl, dq)
        kv_lora0 = F.linear(x, wDKV)            # (bs, sl, dkv + d_rope)

        # -------------------------- 2️⃣ KV‑cache write -------------------------
        new_len = cur_len + kv_lora0.shape[1]
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]       # (bs, kv_len, dkv + d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        # -------------------------- 3️⃣ up‑project Q ---------------------------
        q_up = F.linear(q_lora.squeeze(1), wUQ)               # (bs, nh*(d_nope+d_rope))
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)    # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        # -------------------------- 4️⃣ split KV ------------------------------
        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)  # kv_nope unused
        kv_latent = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        # -------------------------- 5️⃣ “no‑PE” query -------------------------
        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk', q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                         dtype=torch.bfloat16,
                                         device=x.device)

        # -------------------------- 6️⃣ RoPE on Q & K -------------------------
        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, nh, kv_len, d_rope)

        # -------------------------- 7️⃣ scores & soft‑max --------------------
        scores_rope = torch.matmul(q_rope_rot,
                                   k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

        bh = x.shape[0] * nh
        scores_flat = scores.reshape(bh, -1)
        attn = F.softmax(scores_flat, dim=-1).view(x.shape[0], nh, -1)

        # -------------------------- 8️⃣ weighted sum -------------------------
        latent_agg = torch.einsum('bhn,bnd->bhd', attn, kv_latent)

        # -------------------------- 9️⃣ project to value --------------------
        y_head = torch.einsum('bhd, hdf -> bhf', latent_agg, wV_T)   # (bs, nh, dv)

        # -------------------------- 🔟 output projection --------------------
        y_head_flat = y_head.reshape(x.shape[0], nh * dv)       # (bs, nh*dv)
        out = F.linear(y_head_flat, wO)                         # (bs, dim)
        out = out.unsqueeze(1)                                   # (bs, 1, dim)

        return out, kv_data, new_len

    return torch.compile(
        _inner,
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward.
    - When qk_nope_head_dim == 0 and seq_len == 1 we use a
      specialised implementation that:
        * runs separate Q‑down + Q‑up and KV‑down projections,
        * stores the rotated key directly into the KV cache,
        * computes attention using flash‑SDPA,
        * projects values with a per‑head matmul,
        * fuses the huge output projection using cuBLAS (torch.matmul) instead of a custom Triton GEMM.
    - All other configurations fall back to the compiled generic implementation.
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # Convenience aliases (plain ints)
    # ------------------------------------------------------------------
    bs = config.batch_size
    sl = config.seq_len          # always 1 for the fast path
    msl = config.max_seq_len
    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim
    d = config.dim

    # ------------------------------------------------------------------
    # Weight handles (already on device, bfloat16)
    # ------------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight         # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope+drope)*nh, dq)
    wUKV = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO   = config.wo_weight                   # (dim, nh*dv)

    # ------------------------------------------------------------------
    # RoPE tables (global, cached)
    # ------------------------------------------------------------------
    global _cached_cos, _cached_sin
    if _cached_cos is None or _cached_cos.shape[0] < msl:
        _cached_cos, _cached_sin = _get_rope_tables(drope, msl, x.device)

    # ------------------------------------------------------------------
    # Fast path: d_nope == 0 and processing a single token (seq_len == 1)
    # ------------------------------------------------------------------
    if d_nope == 0 and sl == 1:
        # --------------------------------------------------------------
        # 1️⃣ Down‑project Q and KV in one go (still separate for clarity)
        # --------------------------------------------------------------
        x_flat = x.squeeze(1)                     # (bs, dim)

        # Q down‑projection
        q_lora = F.linear(x_flat, wDQ)            # (bs, dq)

        # KV down‑projection (produces latent + rope part)
        kv_down = F.linear(x_flat, wDKV)          # (bs, dkv + drope)
        kv_latent = kv_down[:, :dkv]               # (bs, dkv)
        rope_raw   = kv_down[:, dkv:]               # (bs, drope)

        # --------------------------------------------------------------
        # 2️⃣ Write new token into KV‑cache (latent + rotated key)
        # --------------------------------------------------------------
        cur_len = kv_cache.seq_len                  # current length
        # Rotate rope part for the key at position cur_len
        cos_k = _cached_cos[cur_len].view(1, drope)   # (1, drope)
        sin_k = _cached_sin[cur_len].view(1, drope)   # (1, drope)
        rope_rot = rope_raw * cos_k + _rotate_half(rope_raw) * sin_k   # (bs, drope)

        kv_cache.data[:, cur_len, :dkv] = kv_latent
        kv_cache.data[:, cur_len, dkv:] = rope_rot
        kv_cache.seq_len = cur_len + 1
        kv_len = cur_len + 1

        # --------------------------------------------------------------
        # 3️⃣ Up‑project Q (rope part only, because d_nope == 0)
        # --------------------------------------------------------------
        q_up = F.linear(q_lora, wUQ)                # (bs, nh * drope)
        q_up = q_up.view(bs, nh, drope)            # (bs, nh, drope)

        # --------------------------------------------------------------
        # 4️⃣ Apply RoPE to the current query
        # --------------------------------------------------------------
        q_pos = kv_len - 1
        cos_q = _cached_cos[q_pos].view(1, 1, drope)   # (1,1,drope)
        sin_q = _cached_sin[q_pos].view(1, 1, drope)   # (1,1,drope)
        q_rot = q_up * cos_q + _rotate_half(q_up) * sin_q   # (bs, nh, drope)

        # --------------------------------------------------------------
        # 5️⃣ Gather full KV cache (rotated keys + latent values)
        # --------------------------------------------------------------
        kv_slice = kv_cache.data[:, :kv_len, :]          # (bs, kv_len, dkv + drope)
        k_rope   = kv_slice[..., dkv:]                 # (bs, kv_len, drope)
        v_latent = kv_slice[..., :dkv]                 # (bs, kv_len, dkv)

        # --------------------------------------------------------------
        # 6️⃣ Flash‑SDPA (scaled‑dot‑product attention)
        # --------------------------------------------------------------
        # Expected shapes:
        #   Q : (bs, 1, nh, drope)
        #   K : (bs, kv_len, 1, drope)
        #   V : (bs, kv_len, 1, dkv)
        q_exp = q_rot.unsqueeze(1)          # (bs, 1, nh, drope)
        k_exp = k_rope.unsqueeze(2)         # (bs, kv_len, 1, drope)
        v_exp = v_latent.unsqueeze(2)       # (bs, kv_len, 1, dkv)

        latent_agg = F.scaled_dot_product_attention(
            q_exp, k_exp, v_exp,
            dropout_p=0.0,
            is_causal=False
        ).squeeze(1)                        # (bs, nh, dkv)

        # --------------------------------------------------------------
        # 7️⃣ Value‑projection (latent → per‑head value vectors)
        # --------------------------------------------------------------
        key_wvt = (id(wUKV), d_nope)
        if key_wvt not in _wV_T_cache:
            with torch.no_grad():
                # wUKV shape: ((d_nope+dv)*nh, dkv) -> (nh, d_nope+dv, dkv)
                wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
                # d_nope == 0 ⇒ we take the dv slice directly
                wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1).contiguous()  # (nh, dkv, dv)
                _wV_T_cache[key_wvt] = wV_T
        wV_T = _wV_T_cache[key_wvt]

        # (bs, nh, dkv) x (nh, dkv, dv) -> (bs, nh, dv)
        y_head = torch.einsum('bhd,hdk->bhk', latent_agg, wV_T)   # (bs, nh, dv)

        # --------------------------------------------------------------
        # 8️⃣ Fused output projection (value → model dimension) via cuBLAS
        # --------------------------------------------------------------
        y_head_flat = y_head.reshape(bs, nh * dv)    # (bs, nh*dv)

        key_wot = id(wO)
        if key_wot not in _wO_T_cache:
            _wO_T_cache[key_wot] = wO.t().contiguous()    # (nh*dv, dim)
        wO_T = _wO_T_cache[key_wot]

        # Use torch.matmul (cuBLAS) – generally faster than the custom Triton GEMM for this shape
        out_flat = torch.matmul(y_head_flat, wO_T)   # (bs, dim)

        out = out_flat.unsqueeze(1)                  # (bs, 1, dim)

        return out, kv_cache.data

    # ------------------------------------------------------------------
    # General case – fall back to the compiled generic implementation
    # ------------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,                                   # (bs, sl, dim)
        kv_cache.data,                       # (bs, max_seq_len, dkv+d_rope)
        kv_cache.seq_len,                    # current cache length
        _cached_cos,
        _cached_sin,
        wDQ,
        wDKV,
        wUQ,
        wUKV,
        wO,
        nh,
        d_nope,
        drope,
        dkv,
        dv,
    )

    # update cache state
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    return out, kv_cache.data