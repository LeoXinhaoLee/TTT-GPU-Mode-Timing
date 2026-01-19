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
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# RoPE cache (cos / sin tables) – stored lazily
# ----------------------------------------------------------------------
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Returns (cos, sin) tables of shape (max_seq_len, dim) in bfloat16.
    `dim` must be even.
    """
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        # theta is computed in float32 for precision, then cast to bf16
        theta = (10000.0 ** (-torch.arange(half,
                                          dtype=torch.float32,
                                          device=device) / half)).to(torch.bfloat16)   # (half,)
        pos = torch.arange(max_seq_len,
                           dtype=torch.int64,
                           device=device).unsqueeze_(1)                     # (max_seq_len, 1)
        idx = pos * theta[None, :]                                          # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)                                 # (max_seq_len, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
# Triton row‑wise softmax (bf16)
# ----------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    N: tl.constexpr,              # number of columns
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    # -------- max ----------
    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    # -------- exp & sum ----------
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # -------- normalize ----------
    for start in range(0, N, BLOCK_SIZE):
        cur = start + col
        mask = cur < N
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)

def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax for a 2‑D bf16 tensor using Triton."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape
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
        N=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out

# ----------------------------------------------------------------------
# The compiled MLA kernel
# ----------------------------------------------------------------------
def _compile_mla():
    """
    Returns a torch‑compiled function that performs the heavy part of the
    MLA forward pass. The function signature is:
        _mlafwd(q_lora, kv_lora, kv_len, query_start,
                wUQ, wUKV, wO,
                cos_tbl, sin_tbl,
                dnope, drope, dh, nh, dv)
    The first three tensors are the down‑projected queries and KV‑lora,
    the remaining arguments are static scalars and weight buffers.
    """
    @torch.compile
    def _mlafwd(
        q_lora: torch.Tensor,          # (B, S, Dq)   – down‑projected queries
        kv_lora: torch.Tensor,          # (B, K, Dkv+drope) – cache (including just‑added tokens)
        kv_len: int,                    # total length of the cache after the update
        query_start: int,               # absolute start position of the current query block
        wUQ: torch.Tensor,              # ((dnope+drope)*Nhead , Dq)
        wUKV: torch.Tensor,             # ((dnope+dv)*Nhead , Dkv)
        wO: torch.Tensor,               # (Dim , Nhead*Dv)
        cos_tbl: torch.Tensor,          # (max_seq_len , drope) – pre‑computed cos
        sin_tbl: torch.Tensor,          # (max_seq_len , drope) – pre‑computed sin
        dnope: int,
        drope: int,
        dh: int,                        # = dnope + drope   (size of Q per head)
        nh: int,
        dv: int,
    ) -> torch.Tensor:
        """
        Returns the attention output (B, S, Dim) in bfloat16.
        """
        B, S, _ = q_lora.shape                     # batch and query length
        K = kv_len                                # total cache length after insert

        # --------------------------------------------------------------
        # 1️⃣  Up‑project queries (Q)
        # --------------------------------------------------------------
        # combine batch & seq for a single matmul
        q_flat = q_lora.view(B * S, -1)               # (B*S , Dq)
        q_up = F.linear(q_flat, wUQ)                  # (B*S , (dnope+drope)*Nhead)
        q_up = q_up.view(B, S, nh, dh)                # (B , S , Nhead , dh)
        q_nope, q_rope_raw = torch.split(q_up, [dnope, drope], dim=-1)   # (B,S,Nhead,dnope) , (B,S,Nhead,drope)

        # --------------------------------------------------------------
        # 2️⃣  Prepare KV cache slices
        # --------------------------------------------------------------
        # kv_lora already contains (dkv + drope) for the entire cache
        kv_latent = kv_lora[..., :wUKV.shape[1]]      # (B , K , Dkv)
        kv_rope_raw = kv_lora[..., wUKV.shape[1]:]   # (B , K , drope)

        # --------------------------------------------------------------
        # 3️⃣  RoPE for queries
        # --------------------------------------------------------------
        # slice the pre‑computed tables for the current query block
        # query positions are [query_start , query_start + S)
        cos_q = cos_tbl[query_start:query_start + S]               # (S , drope)
        sin_q = sin_tbl[query_start:query_start + S]               # (S , drope)
        # broadcast to (1,1,S,drope)
        cos_q = cos_q.view(1, 1, S, drope)
        sin_q = sin_q.view(1, 1, S, drope)

        # apply RoPE (BF16)
        q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (B,S,Nhead,drope)
        # move head dimension before seq for later matmuls
        q_rope = q_rope.permute(0, 2, 1, 3)               # (B,Nhead,S,drope)

        # --------------------------------------------------------------
        # 4️⃣  RoPE for keys (whole cache)
        # --------------------------------------------------------------
        cos_k = cos_tbl[:K]                                 # (K , drope)
        sin_k = sin_tbl[:K]                                 # (K , drope)
        k_rope = kv_rope_raw * cos_k + _rotate_half(kv_rope_raw) * sin_k   # (B , K , drope)

        # --------------------------------------------------------------
        # 5️⃣  Split the up‑projected Q into NoPE / RoPE parts
        # --------------------------------------------------------------
        q_nope = q_nope.permute(0, 2, 1, 3)                # (B,Nhead,S,dnope)

        # --------------------------------------------------------------
        # 6️⃣  Extract per‑head matrices from wUKV
        # --------------------------------------------------------------
        wUKV_view = wUKV.view(nh, dnope + dv, -1)          # (Nhead , dnope+dv , Dkv)
        wK = wUKV_view[:, :dnope, :]                       # (Nhead , dnope , Dkv)
        wV = wUKV_view[:, dnope:, :]                       # (Nhead , dv , Dkv)
        wV_T = wV.permute(0, 2, 1)                         # (Nhead , Dkv , dv)

        # --------------------------------------------------------------
        # 7️⃣  Project NoPE queries into the latent space
        # --------------------------------------------------------------
        if dnope > 0:
            # q_nope : (B,Nhead,S,dnope) , wK : (Nhead,dnope,Dkv)
            # result : (B,Nhead,S,Dkv)
            q_nope_latent = torch.einsum('bhsd, hde -> bhse', q_nope, wK)
        else:
            q_nope_latent = torch.zeros((B, nh, S, wK.shape[-1]), dtype=torch.bfloat16,
                                        device=q_nope.device)

        # --------------------------------------------------------------
        # 8️⃣  Compute attention scores (NoPE + RoPE)
        # --------------------------------------------------------------
        # NoPE part: (B,Nhead,S,Dkv) • (B,K,Dkv)ᵀ  → (B,Nhead,S,K)
        scores_nope = torch.einsum('bhse, bte -> bhst', q_nope_latent, kv_latent)

        # RoPE part: (B,Nhead,S,drope) • (B,K,drope)ᵀ → (B,Nhead,S,K)
        scores_rope = torch.einsum('bhsc, btc -> bhst', q_rope, k_rope)

        # Combine and scale
        scale = 1.0 / math.sqrt(dnope + drope)
        scores = (scores_nope + scores_rope) * scale          # (B,Nhead,S,K)

        # --------------------------------------------------------------
        # 9️⃣  Row‑wise softmax (Triton) – flatten the first three axes
        # --------------------------------------------------------------
        scores_flat = scores.view(B * nh * S, K)               # (B*nh*S , K)
        attn_flat = _triton_softmax(scores_flat)               # same shape
        attn = attn_flat.view(B, nh, S, K)                     # (B,Nhead,S,K)

        # --------------------------------------------------------------
        # 🔟  Attention‑weighted aggregation of latent vectors
        # --------------------------------------------------------------
        # attn : (B,Nhead,S,K)   ,   kv_latent : (B,K,Dkv)
        # result latent_agg : (B,Nhead,S,Dkv)
        latent_agg = torch.einsum('bhsk, bkd -> bhsd', attn, kv_latent)

        # --------------------------------------------------------------
        # 1️⃣1️⃣  Project aggregated latents to value space
        # --------------------------------------------------------------
        # latent_agg : (B,Nhead,S,Dkv)   ,   wV_T : (Nhead,Dkv,Dv)
        y_head = torch.einsum('bhsd, hdf -> bhsf', latent_agg, wV_T)   # (B,Nhead,S,Dv)

        # --------------------------------------------------------------
        # 1️⃣2️⃣  Final linear projection back to model dimension
        # --------------------------------------------------------------
        y_head_flat = y_head.reshape(B, S, nh * dv)            # (B,S,Nhead*Dv)
        out = F.linear(y_head_flat, wO)                        # (B,S,Dim)
        return out
    return _mlafwd

# ----------------------------------------------------------------------
# Custom kernel entry point – called by the test harness
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast MLA forward pass.
    Returns
    -------
    output : torch.Tensor    # shape (batch, seq_len, dim) – bfloat16
    kv_cache.data : torch.Tensor   # the up‑to‑date cache tensor (raw kv‑lora)
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Unpack scalar configuration values (avoid attribute look‑ups later)
    # --------------------------------------------------------------
    bs = config.batch_size
    sl = config.seq_len                     # may be >1 for pre‑fill
    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim
    dh = dnope + drope                     # size of a single query head before splitting
    msl = config.max_seq_len

    # --------------------------------------------------------------
    # Extract weight matrices (already on CUDA, BF16)
    # --------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq , dim)
    wDKV  = config.KV_proj_down_weight         # (dkv+drope , dim)
    wUQ   = config.Q_proj_up_weight            # ((dnope+drope)*nh , dq)
    wUKV  = config.KV_proj_up_weight           # ((dnope+dv)*nh , dkv)
    wO    = config.wo_weight                   # (dim , nh*dv)

    # --------------------------------------------------------------
    # 1️⃣  Combine the two down‑projections into a single matmul
    # --------------------------------------------------------------
    # (dq + dkv + drope) x dim   – the concatenation is cheap compared to the matmul
    wD_combined = torch.cat([wDQ, wDKV], dim=0)          # (dq + dkv + drope , dim)
    down = F.linear(x, wD_combined)                     # (B , S , dq + dkv + drope)

    q_lora = down[..., :dq]                             # (B , S , dq)
    kv_lora_tok = down[..., dq:]                        # (B , S , dkv + drope)

    # --------------------------------------------------------------
    # 2️⃣  KV‑cache update (in‑place).  It returns the full cache tensor and
    #     the new total length.
    # --------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_tok)             # kv_lora : (B , K , dkv+drope)
    # absolute start position of the current query block (may be >0 when S>1)
    query_start = kv_len - sl

    # --------------------------------------------------------------
    # 3️⃣  Prepare RoPE tables (cached globally)
    # --------------------------------------------------------------
    cos_tbl, sin_tbl = _get_rope_tables(drope, msl, x.device)

    # --------------------------------------------------------------
    # 4️⃣  Call the heavy‑lifting compiled routine
    # --------------------------------------------------------------
    # compile the inner routine lazily – only once per Python process
    if not hasattr(custom_kernel, "_mlafwd"):
        custom_kernel._mlafwd = _compile_mla()

    output = custom_kernel._mlafwd(
        q_lora,                     # (B , S , dq)
        kv_lora,                    # (B , K , dkv+drope)
        kv_len,                     # int
        query_start,                # int
        wUQ,
        wUKV,
        wO,
        cos_tbl,
        sin_tbl,
        dnope,
        drope,
        dh,
        nh,
        dv,
    )                               # (B , S , dim)  – BF16

    # --------------------------------------------------------------
    # Return the attention output and the (now‑updated) KV‑cache raw tensor
    # --------------------------------------------------------------
    return output, kv_cache.data