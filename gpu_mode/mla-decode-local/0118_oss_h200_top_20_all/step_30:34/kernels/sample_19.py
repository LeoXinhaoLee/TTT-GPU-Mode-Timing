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
# Helper – rotate‑half (identical to the reference implementation)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Same as the one used in the reference."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# Global RoPE tables (cos / sin) – lazily built
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None          # (max_seq_len, rope_dim)   bfloat16
_cached_sin: torch.Tensor = None          # (max_seq_len, rope_dim)   bfloat16


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Create (cos, sin) tables for rotary embeddings (bfloat16)."""
    half = dim // 2
    # theta : (half,)
    theta = (10000.0 ** (-torch.arange(half,
                                      dtype=torch.float32,
                                      device=device) / half)).to(torch.bfloat16)
    # pos : (max_seq_len, 1)
    pos = torch.arange(max_seq_len,
                       dtype=torch.int64,
                       device=device).unsqueeze_(1)
    # idx : (max_seq_len, half)
    idx = pos * theta
    # duplicate to get full dim
    idx = torch.cat([idx, idx], dim=-1)          # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel – fused attention & per‑head value projection (no‑NoPE case)
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_fused_kernel(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,            # (B, H, Dq)            bfloat16 – already rotary‑applied
    K_raw_ptr,        # (B, L, Dq)            bfloat16 – raw rope part (not rotated)
    V_ptr,            # (B, L, Dv_lat)        bfloat16 – latent values (kv‑lora)
    cos_ptr,          # (max_seq_len, Dq)     bfloat16
    sin_ptr,          # (max_seq_len, Dq)     bfloat16
    wV_T_ptr,         # (H, Dv_lat, Dv)       bfloat16
    Y_ptr,            # (B, H, Dv)            bfloat16 – output per‑head

    # ------------------------------------------------------------------
    # Strides (in elements, not bytes)
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len,  stride_k_dim,
    stride_cos_seq, stride_cos_dim,
    stride_sin_seq, stride_sin_dim,
    stride_v_batch, stride_v_len, stride_v_dim,
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
    stride_y_batch, stride_y_head, stride_y_dv,

    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,                # batch size
    H: tl.constexpr,                # total heads
    L: tl.constexpr,                # KV length (after insertion)
    Dq: tl.constexpr,               # rope dimension (e.g. 64)
    Dv_lat: tl.constexpr,           # latent value dimension (kv_lora_rank, e.g. 512)
    Dv: tl.constexpr,               # per‑head output dim (v_head_dim, e.g. 128)
    scale: tl.constexpr,            # 1/√(Dq)
    HEADS_PER_BLOCK: tl.constexpr,  # how many heads a program handles
    BLOCK_K: tl.constexpr,          # KV positions processed per iteration
    BLOCK_DV: tl.constexpr,         # tile size on the latent‑value dimension
):
    """
    * All heads of a single batch element are processed by one program.
    * The kernel implements a numerically‑stable soft‑max and directly
      accumulates the final per‑head output:
          y = Σ_i (exp(score_i) * (V_i @ wV_T)) / Σ_i exp(score_i)
      This removes the intermediate “latent” buffer and the second
      weight‑projection pass, saving memory traffic.
    * RoPE is applied on‑the‑fly to the keys, while queries are already
      rotated outside the kernel (see the Python wrapper).
    """

    # ------------------------------------------------------------------
    # Program / tile identifiers
    # ------------------------------------------------------------------
    pid = tl.program_id(0)                    # batch index
    head_block = tl.program_id(1)              # head‑block index

    b = pid                                    # concrete batch index
    head_start = head_block * HEADS_PER_BLOCK   # first head handled by this block
    hrange = tl.arange(0, HEADS_PER_BLOCK)     # 0 … HEADS_PER_BLOCK‑1
    hidx = head_start + hrange                  # absolute head IDs

    # ------------------------------------------------------------------
    # Mask for “real” heads (the last block may be partial)
    # ------------------------------------------------------------------
    head_mask = hidx < H                         # (HEADS_PER_BLOCK,)

    # ------------------------------------------------------------------
    # Load Q (already rotated) – shape (HPB, Dq)
    # ------------------------------------------------------------------
    offs_q = (
        b * stride_q_batch
        + hidx[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_mask[:, None],
                 other=0.0)                     # (HPB, Dq)

    # --------------------------------------------------------------
    # Split Q into its two rotary halves
    # --------------------------------------------------------------
    half = Dq // 2
    q_left  = q[:, :half]                       # (HPB, half)
    q_right = q[:, half:]                       # (HPB, half)

    # --------------------------------------------------------------
    # Stable‑softmax accumulators (float32)
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HPB,)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (HPB,)

    # --------------------------------------------------------------
    # Accumulator for the final per‑head output (float32)
    # --------------------------------------------------------------
    y_head = tl.zeros([HEADS_PER_BLOCK, Dv], dtype=tl.float32)        # (HPB, Dv)

    # --------------------------------------------------------------
    # Main loop over KV positions (single‑pass soft‑max)
    # --------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k   = start_k + tl.arange(0, BLOCK_K)                     # (BLOCK_K,)
        k_mask  = cur_k < L

        # ----------------------------------------------------------
        # Load raw K‑block (rope part, not yet rotated)
        # ----------------------------------------------------------
        offs_k_raw = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_raw = tl.load(K_raw_ptr + offs_k_raw,
                         mask=k_mask[:, None],
                         other=0.0)                                  # (BK, Dq)

        # ----------------------------------------------------------
        # Load cosine / sine for the same positions
        # ----------------------------------------------------------
        offs_cos = (
            cur_k[:, None] * stride_cos_seq
            + tl.arange(0, Dq)[None, :] * stride_cos_dim
        )
        cos_block = tl.load(cos_ptr + offs_cos,
                            mask=k_mask[:, None],
                            other=0.0)                                # (BK, Dq)

        offs_sin = (
            cur_k[:, None] * stride_sin_seq
            + tl.arange(0, Dq)[None, :] * stride_sin_dim
        )
        sin_block = tl.load(sin_ptr + offs_sin,
                            mask=k_mask[:, None],
                            other=0.0)                                # (BK, Dq)

        # ----------------------------------------------------------
        # Split K / cos / sin into halves
        # ----------------------------------------------------------
        k_left   = k_raw[:, :half]
        k_right  = k_raw[:, half:]

        cos_left = cos_block[:, :half]
        cos_right= cos_block[:, half:]

        sin_left = sin_block[:, :half]
        sin_right= sin_block[:, half:]

        # ----------------------------------------------------------
        # Apply RoPE on‑the‑fly (coefficients for the dot product)
        # ----------------------------------------------------------
        coeff_left  = q_left * cos_left + q_right * sin_right          # (HPB, half)
        coeff_right = -q_left * sin_left + q_right * cos_right          # (HPB, half)

        # ----------------------------------------------------------
        # Dot‑products (heads × K‑block) = rotated‑dot‑product
        # ----------------------------------------------------------
        term_left  = tl.dot(coeff_left,  k_left,  trans_b=True)        # (HPB, BK)
        term_right = tl.dot(coeff_right, k_right, trans_b=True)        # (HPB, BK)
        prod = term_left + term_right
        score_f32 = tl.cast(prod, tl.float32) * scale                # (HPB, BK)

        # ----------------------------------------------------------
        # Stable‑softmax bookkeeping
        # ----------------------------------------------------------
        block_max = tl.max(score_f32, axis=1)                         # (HPB,)
        new_max   = tl.maximum(max_score, block_max)                  # (HPB,)

        # rescale previous accumulators
        scale_factor = tl.exp(max_score - new_max)                    # (HPB,)
        sum_exp   = sum_exp * scale_factor
        y_head    = y_head * tl.cast(scale_factor, tl.float32)[:, None]

        exp_score_f32 = tl.exp(score_f32 - new_max[:, None])          # (HPB, BK)
        sum_exp = sum_exp + tl.sum(exp_score_f32, axis=1)             # (HPB,)

        # ----------------------------------------------------------
        # Accumulate weighted V × wV_T (fused)
        # ----------------------------------------------------------
        for start_d in range(0, Dv_lat, BLOCK_DV):
            cur_d   = start_d + tl.arange(0, BLOCK_DV)
            d_mask  = cur_d < Dv_lat

            # Load V slice (latent values)
            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(V_ptr + offs_v,
                              mask=k_mask[:, None] & d_mask[None, :],
                              other=0.0)                              # (BK, BDV)
            v_slice_f32 = tl.cast(v_slice, tl.float32)                # (BK, BDV)

            # Load transposed wV_T slice (DKV → Dv) for the current latent tile
            # shape of wV_T slice: (HPB, BDV, Dv)
            offs_wV = (
                hidx[:, None, None] * stride_wV_T_head
                + cur_d[None, :, None] * stride_wV_T_lat
                + tl.arange(0, Dv)[None, None, :] * stride_wV_T_out
            )
            wV_block = tl.load(wV_T_ptr + offs_wV,
                               mask=head_mask[:, None] & d_mask[None, :],
                               other=0.0)                         # (HPB, BDV, Dv)

            # V_proj = V_slice @ wV_block        -> (BK, Dv) per head
            # Broadcasting rules give (HPB, BK, Dv)
            v_proj = tl.dot(v_slice_f32, wV_block)                    # (HPB, BK, Dv)

            # Weighted sum over the KV positions
            # exp_score_f32 : (HPB, BK) → expand to (HPB, BK, 1)
            contrib = tl.sum(exp_score_f32[:, :, None] * v_proj, axis=1)  # (HPB, Dv)

            y_head = y_head + contrib

        # ----------------------------------------------------------
        # End of KV‑position loop – update max_score
        # ----------------------------------------------------------
        max_score = new_max

    # ------------------------------------------------------------------
    # Normalise the per‑head output (divide by Σexp)
    # ------------------------------------------------------------------
    y_head = y_head / sum_exp[:, None]                               # (HPB, Dv)

    # ------------------------------------------------------------------
    # Store the results
    # ------------------------------------------------------------------
    offs_y = (
        b * stride_y_batch
        + hidx[:, None] * stride_y_head
        + tl.arange(0, Dv)[None, :] * stride_y_dv
    )
    tl.store(Y_ptr + offs_y,
             tl.cast(y_head, tl.bfloat16),
             mask=head_mask[:, None])


# ----------------------------------------------------------------------
# Fast‑path for the common configuration (qk_nope_head_dim == 0)
# ----------------------------------------------------------------------
def _fast_forward_multihead_opt(
    config: Config,
    x: torch.Tensor,
    kv_cache: KVCache,
    wDQ: torch.Tensor,
    wDKV: torch.Tensor,
    wUQ: torch.Tensor,
    wUKV: torch.Tensor,
    wO: torch.Tensor,
    cos_tbl: torch.Tensor,
    sin_tbl: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass when there is **no** NoPE part.
    It uses a fused Triton kernel that computes the attention scores,
    performs a single‑pass stable soft‑max and directly accumulates the
    per‑head output (avoiding a separate latent‑value buffer).
    """
    bs = config.batch_size
    nh = config.n_heads
    dim = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # --------------------------------------------------------------
    # 1️⃣ Down‑projection + KV cache update
    # --------------------------------------------------------------
    # x : (B, 1, Dim) → squeeze for the linear layers
    x2 = x.squeeze(1)                                   # (B, Dim)

    q_lora  = F.linear(x2, wDQ)                         # (B, dq)
    kv_lora = F.linear(x2, wDKV)                        # (B, dkv + drope)

    # ----------------------------------------------------------------
    # 2️⃣ KV‑cache (store raw rope part, keep latent part)
    # ----------------------------------------------------------------
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1

    kv_latent_new = kv_lora[:, :dkv]                     # (B, dkv)
    rope_raw_new  = kv_lora[:, dkv:]                     # (B, drope)

    # store into cache (already in bfloat16)
    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_raw_new
    kv_cache.seq_len = new_len

    # --------------------------------------------------------------
    # 3️⃣ Query up‑projection + RoPE (queries are already rotated)
    # --------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                        # (B, nh * drope)
    q_up = q_up.view(bs, nh, drope)                     # (B, nh, drope)

    # query position (the token we are generating now)
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                               # (drope,)
    sin_q = sin_tbl[q_pos]                               # (drope,)

    # RoPE for queries – identical to the reference implementation
    q_rope = q_up * cos_q + _rotate_half(q_up) * sin_q   # (B, nh, drope)
    q = q_rope                                            # (B, nh, drope)

    # --------------------------------------------------------------
    # 4️⃣ Gather KV from the cache (latent values + raw rope part)
    # --------------------------------------------------------------
    kv_all   = kv_cache.data[:, :new_len, :]            # (B, L, dkv + drope)
    k_rope_raw = kv_all[..., dkv:]                      # (B, L, drope) – not rotated yet
    v_latent   = kv_all[..., :dkv]                      # (B, L, dkv)

    # --------------------------------------------------------------
    # 5️⃣ Prepare weight for the fused kernel (transpose for fast access)
    # --------------------------------------------------------------
    # wUKV shape: ((d_nope + dv) * nh , dkv)  ; d_nope == 0
    # → view as (nh, dv, dkv) and then transpose to (nh, dkv, dv)
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)

    # --------------------------------------------------------------
    # 6️⃣ Allocate buffer for per‑head outputs
    # --------------------------------------------------------------
    y_head = torch.empty((bs, nh, dv), dtype=torch.bfloat16, device=x.device)

    # --------------------------------------------------------------
    # 7️⃣ Strides (all tensors are contiguous → can use .stride())
    # --------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim = q.stride()
    stride_k_batch, stride_k_len , stride_k_dim  = k_rope_raw.stride()
    stride_cos_seq, stride_cos_dim = cos_tbl.stride()
    stride_sin_seq, stride_sin_dim = sin_tbl.stride()
    stride_v_batch, stride_v_len, stride_v_dim = v_latent.stride()
    stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out = wV_T.stride()
    stride_y_batch, stride_y_head, stride_y_dv = y_head.stride()

    # --------------------------------------------------------------
    # 8️⃣ Triton launch configuration
    # --------------------------------------------------------------
    # Tune block sizes: a modest HEADS_PER_BLOCK works well on Hopper.
    HEADS_PER_BLOCK = 32                                 # 4 programs per batch
    BLOCK_K = 256                                        # positions per inner iteration
    BLOCK_DV = 128 if dkv >= 256 else 64                # latent‑tile size

    # Number of head‑blocks needed
    n_head_blocks = (nh + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    grid = (bs, n_head_blocks)                          # one program per (batch, head‑block)

    scale = 1.0 / math.sqrt(drope)                       # Dq == drope (no‑NoPE dim)

    _triton_attn_fused_kernel[grid](
        # pointers
        q,                    # (B, H, Dq) – already rotated
        k_rope_raw,
        v_latent,
        cos_tbl,
        sin_tbl,
        wV_T,
        y_head,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len , stride_k_dim,
        stride_cos_seq, stride_cos_dim,
        stride_sin_seq, stride_sin_dim,
        stride_v_batch, stride_v_len, stride_v_dim,
        stride_wV_T_head, stride_wV_T_lat, stride_wV_T_out,
        stride_y_batch, stride_y_head, stride_y_dv,
        # compile‑time constants
        bs, nh, new_len, drope, dkv, dv,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DV,
        # launch config
        num_warps=8, num_stages=4,
    )

    # --------------------------------------------------------------
    # 9️⃣ Final linear projection (per‑head outputs → model dim)
    # --------------------------------------------------------------
    y_head_flat = y_head.view(bs, nh * dv)               # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                      # (B, Dim)
    out = out.unsqueeze(1)                               # (B, 1, Dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Fallback – compiled reference implementation (handles d_nope > 0)
# ----------------------------------------------------------------------
_compiled_forward: torch._dynamo.eval_frame.OptimizedModule = None


def _build_compiled_forward():
    """Build the generic (d_nope > 0) implementation using torch.compile."""
    import torch.nn.functional as F

    def _inner(
        x: torch.Tensor,
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
        dv: int,
    ):
        # ---- identical to the reference forward (see the prompt) ----
        q_lora = F.linear(x, wDQ)               # (bs, 1, dq)
        kv_lora0 = F.linear(x, wDKV)            # (bs, 1, dkv + d_rope)

        new_len = cur_len + kv_lora0.shape[1]
        kv_data[:, cur_len:new_len, :] = kv_lora0.to(kv_data.dtype)
        kv_lora = kv_data[:, :new_len, :]       # (bs, kv_len, dkv + d_rope)
        kv_len = new_len
        query_pos = kv_len - 1

        q_up = F.linear(q_lora.squeeze(1), wUQ)               # (bs, nh*d_nope+d_rope)
        q_up = q_up.view(x.shape[0], nh, d_nope + d_rope)    # (bs, nh, d_nope+d_rope)
        q_nope, q_rope = torch.split(q_up, [d_nope, d_rope], dim=-1)

        kv_nope, k_rope = torch.split(kv_lora, [dkv, d_rope], dim=-1)  # kv_nope unused
        kv_latent = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)               # (nh, d_nope+dv, dkv)
        wK = wUKV_view[:, :d_nope, :] if d_nope > 0 else None   # (nh, d_nope, dkv)
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1)          # (nh, dkv, dv)

        if d_nope > 0:
            q_nope_latent = torch.einsum('bhd, hdk -> bhk', q_nope, wK)               # (bs, nh, dkv)
        else:
            q_nope_latent = torch.zeros((x.shape[0], nh, dkv),
                                         dtype=torch.bfloat16,
                                         device=x.device)

        cos_q = cos_tbl[query_pos].view(1, 1, d_rope)
        sin_q = sin_tbl[query_pos].view(1, 1, d_rope)
        q_rope_rot = q_rope * cos_q + _rotate_half(q_rope) * sin_q   # (bs, nh, d_rope)

        cos_k = cos_tbl[:kv_len].unsqueeze(0)   # (1, kv_len, d_rope)
        sin_k = sin_tbl[:kv_len].unsqueeze(0)
        k_rope_rot = k_rope * cos_k + _rotate_half(k_rope) * sin_k   # (bs, kv_len, d_rope)

        scores_rope = torch.matmul(q_rope_rot,
                                   k_rope_rot.transpose(-2, -1))      # (bs, nh, kv_len)
        scores_nope = torch.matmul(q_nope_latent,
                                   kv_latent.transpose(-2, -1))         # (bs, nh, kv_len)
        scores = (scores_rope + scores_nope) * (1.0 / math.sqrt(d_nope + d_rope))

        bh = x.shape[0] * nh
        scores_flat = scores.reshape(bh, -1)
        attn = F.softmax(scores_flat, dim=-1).to(torch.bfloat16).view(x.shape[0], nh, -1)

        latent_agg = torch.matmul(attn, kv_latent)               # (bs, nh, dkv)

        y_head = torch.einsum('bhd, hdf -> bhf', latent_agg, wV_T)   # (bs, nh, dv)

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
# Main entry point (custom_kernel)
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expected entry point for the benchmark harness.
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Extract scalar configuration values
    # --------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dim = config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # --------------------------------------------------------------
    # Weights (already stored in the Config instance)
    # --------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight        # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh , dq)
    wUKV = config.KV_proj_up_weight          # ((d_nope + dv)  * nh , dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

    # --------------------------------------------------------------
    # Build / fetch RoPE tables (cached globally)
    # --------------------------------------------------------------
    global _cached_cos, _cached_sin
    if (_cached_cos is None) or (_cached_cos.shape[0] < config.max_seq_len):
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # --------------------------------------------------------------
    # Fast‑path when there is no “NoPE” component (the most common case)
    # --------------------------------------------------------------
    if d_nope == 0:
        return _fast_forward_multihead_opt(
            config, x, kv_cache,
            wDQ, wDKV, wUQ, wUKV, wO,
            _cached_cos, _cached_sin,
        )

    # --------------------------------------------------------------
    # General case – fall back to the compiled reference implementation
    # --------------------------------------------------------------
    global _compiled_forward
    if _compiled_forward is None:
        _compiled_forward = _build_compiled_forward()

    out, new_kv_data, new_len = _compiled_forward(
        x,
        kv_cache.data,
        kv_cache.seq_len,
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
    # update KVCache state
    kv_cache.data = new_kv_data
    kv_cache.seq_len = int(new_len)

    # --------------------------------------------------------------
    # Return the result and the (updated) KV cache tensor
    # --------------------------------------------------------------
    return out, kv_cache.data