### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config   # noqa: F401
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
# Helper – rotate‑half (identical to the reference implementation)
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the last dimension (RoPE helper)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# ----------------------------------------------------------------------
# Cached rotary tables (cos / sin) – generated once per device / max seq‑len
# ----------------------------------------------------------------------
_cached_cos: torch.Tensor = None   # (max_seq_len, rope_dim)   bfloat16
_cached_sin: torch.Tensor = None   # (max_seq_len, rope_dim)   bfloat16


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """
    Produce the (cos, sin) tables for RoPE in bfloat16.
    This mirrors the reference implementation.
    """
    half = dim // 2
    theta = (10000.0 ** (-torch.arange(half,
                                      dtype=torch.float32,
                                      device=device) / half)).to(torch.bfloat16)   # (half,)
    pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)   # (max_seq_len,1)
    idx = pos * theta                                                   # (max_seq_len, half)
    idx = torch.cat([idx, idx], dim=-1)                                # (max_seq_len, dim)
    return idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel – computes:
#   * stable soft‑max over the rotary part,
#   * weighted sum of the latent KV values (size dkv),
#   * stores the latent per‑head result into *latent_out*.
# The per‑head value‑projection is performed later with a batched
# torch.einsum (leveraging cuBLAS for the GEMM).
# ----------------------------------------------------------------------
@triton.jit
def _triton_attn_latent_kernel_opt_rope(
    # ------------------------------------------------------------------
    # Pointers
    # ------------------------------------------------------------------
    Q_ptr,               # (B, H, Dq)               bfloat16 – already rotated
    K_raw_ptr,           # (B, L, Dq)               bfloat16 – raw rope part
    V_ptr,               # (B, L, Dkv)              bfloat16
    cos_ptr,             # (max_seq_len, Dq)        bfloat16
    sin_ptr,             # (max_seq_len, Dq)        bfloat16
    latent_ptr,          # (B, H, Dkv)              bfloat16 – output
    # ------------------------------------------------------------------
    # Strides (in elements)
    # ------------------------------------------------------------------
    stride_q_batch, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_len , stride_k_dim,
    stride_cos_seq, stride_cos_dim,
    stride_sin_seq, stride_sin_dim,
    stride_v_batch, stride_v_len , stride_v_dim,
    stride_lat_batch, stride_lat_head, stride_lat_dim,
    # ------------------------------------------------------------------
    # Compile‑time constants
    # ------------------------------------------------------------------
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # total heads
    L: tl.constexpr,          # KV length (after insert)
    Dq: tl.constexpr,         # rope dim (e.g. 64)
    Dkv: tl.constexpr,        # kv‑lora rank (e.g. 512)
    scale: tl.constexpr,      # 1 / sqrt(Dq)
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,    # KV positions per iteration
    BLOCK_DKV: tl.constexpr,  # tile size on the latent dimension
):
    """
    One program processes ``HEADS_PER_BLOCK`` heads for a single batch element.
    The kernel:
      1. Loads the (already‑rotated) query Q for those heads.
      2. Iterates over KV positions in chunks of ``BLOCK_K``:
         * Loads the raw rope part of K, applies rotary on‑the‑fly.
         * Computes the dot‑product Q·K, builds a numerically‑stable soft‑max.
         * Loads the latent V values and accumulates the attention‑weighted sum.
      3. After the loop, normalises by the softmax denominator and writes
         the per‑head latent vector (size ``Dkv``) to ``latent_ptr``.
    """
    pid = tl.program_id(0)                # batch index
    pid_h = tl.program_id(1)              # head‑tile index

    b = pid
    head_start = pid_h * HEADS_PER_BLOCK
    head_range = tl.arange(0, HEADS_PER_BLOCK)
    head_idx   = head_start + head_range
    head_mask  = head_idx < H

    # --------------------------------------------------------------
    # Load the already‑rotated queries (B, H, Dq)
    # --------------------------------------------------------------
    offs_q = (
        b * stride_q_batch
        + head_idx[:, None] * stride_q_head
        + tl.arange(0, Dq)[None, :] * stride_q_dim
    )
    q = tl.load(Q_ptr + offs_q,
                 mask=head_mask[:, None],
                 other=0.0)                     # (HEADS_PER_BLOCK, Dq)

    # split Q for rotary
    half = Dq // 2
    q_l = q[:, :half]
    q_r = q[:, half:]

    # --------------------------------------------------------------
    # Stable‑softmax accumulators (float32)
    # --------------------------------------------------------------
    max_score = tl.full([HEADS_PER_BLOCK], -float("inf"), tl.float32)   # (HPB,)
    sum_exp   = tl.full([HEADS_PER_BLOCK], 0.0, tl.float32)            # (HPB,)

    # accumulator for the latent (attention‑weighted sum of V)
    latent_acc = tl.zeros([HEADS_PER_BLOCK, Dkv], dtype=tl.float32)    # (HPB, Dkv)

    # --------------------------------------------------------------
    # Main loop over KV positions
    # --------------------------------------------------------------
    for start_k in range(0, L, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)            # (BLOCK_K,)
        k_mask = cur_k < L

        # --------------------  K (raw rope part)  --------------------
        offs_k_raw = (
            b * stride_k_batch
            + cur_k[:, None] * stride_k_len
            + tl.arange(0, Dq)[None, :] * stride_k_dim
        )
        k_raw = tl.load(K_raw_ptr + offs_k_raw,
                         mask=k_mask[:, None],
                         other=0.0)                      # (BLOCK_K, Dq)

        # --------------------  cos / sin tables  --------------------
        offs_cos = (
            cur_k[:, None] * stride_cos_seq
            + tl.arange(0, Dq)[None, :] * stride_cos_dim
        )
        cos_block = tl.load(cos_ptr + offs_cos,
                            mask=k_mask[:, None],
                            other=0.0)                      # (BLOCK_K, Dq)

        offs_sin = (
            cur_k[:, None] * stride_sin_seq
            + tl.arange(0, Dq)[None, :] * stride_sin_dim
        )
        sin_block = tl.load(sin_ptr + offs_sin,
                            mask=k_mask[:, None],
                            other=0.0)                      # (BLOCK_K, Dq)

        # split halves ------------------------------------------------
        k_l = k_raw[:, :half]
        k_r = k_raw[:, half:]

        cos_l = cos_block[:, :half]
        cos_r = cos_block[:, half:]

        sin_l = sin_block[:, :half]
        sin_r = sin_block[:, half:]

        # rotate K on‑the‑fly ----------------------------------------
        coeff_l = q_l * cos_l + q_r * sin_r           # (HPB, half)
        coeff_r = -q_l * sin_l + q_r * cos_r           # (HPB, half)

        # dot‑product -------------------------------------------------
        term_l = tl.dot(coeff_l, k_l, trans_b=True)   # (HPB, BLOCK_K)
        term_r = tl.dot(coeff_r, k_r, trans_b=True)   # (HPB, BLOCK_K)
        scores = tl.cast(term_l + term_r, tl.float32) * scale   # (HPB, BLOCK_K)

        # -------- stable soft‑max update (max / sum) ---------------
        block_max = tl.max(scores, axis=1)            # (HPB,)
        new_max   = tl.maximum(max_score, block_max)  # (HPB,)

        # rescale previous accumulators
        scale_factor = tl.exp(max_score - new_max)    # (HPB,)
        sum_exp   = sum_exp * scale_factor
        latent_acc = latent_acc * tl.cast(scale_factor, tl.float32)[:, None]

        exp_score = tl.exp(scores - new_max[:, None])   # (HPB, BLOCK_K)
        sum_exp += tl.sum(exp_score, axis=1)

        # ---------------- V (latent) accumulation -----------------
        # exp_score : (HPB, BLOCK_K)
        # V slice  : (BLOCK_K, Dkv)   – we tile on Dkv to stay within registers
        for start_d in range(0, Dkv, BLOCK_DKV):
            cur_d   = start_d + tl.arange(0, BLOCK_DKV)
            d_mask  = cur_d < Dkv

            offs_v = (
                b * stride_v_batch
                + cur_k[:, None] * stride_v_len
                + cur_d[None, :] * stride_v_dim
            )
            v_slice = tl.load(V_ptr + offs_v,
                              mask=k_mask[:, None] & d_mask[None, :],
                              other=0.0)                     # (BLOCK_K, BLOCK_DKV)
            v_f32 = tl.cast(v_slice, tl.float32)               # (BLOCK_K, BLOCK_DKV)

            # latent_acc[:, start_d:start_d+BLOCK_DKV] += exp_score @ v_f32
            # Using tl.dot: (HPB, BLOCK_K) x (BLOCK_K, BLOCK_DKV) -> (HPB, BLOCK_DKV)
            latent_acc[:, start_d:start_d + BLOCK_DKV] += tl.dot(exp_score, v_f32)

        # update max_score for next iteration
        max_score = new_max

    # --------------------------------------------------------------
    # Normalise latent (divide by Σexp) – still float32
    # --------------------------------------------------------------
    latent_f32 = latent_acc / sum_exp[:, None]            # (HPS, Dkv)

    # cast to bfloat16 for the output buffer
    latent_bf = tl.cast(latent_f32, tl.bfloat16)

    # --------------------------------------------------------------
    # Store per‑head latent vectors
    # --------------------------------------------------------------
    for start_d in range(0, Dkv, BLOCK_DKV):
        cur_d   = start_d + tl.arange(0, BLOCK_DKV)
        d_mask  = cur_d < Dkv

        offs_lat = (
            b * stride_lat_batch
            + head_idx[:, None] * stride_lat_head
            + cur_d[None, :] * stride_lat_dim
        )
        block = latent_bf[:, start_d:start_d + BLOCK_DKV]      # (HPB, BLOCK_DKV)
        tl.store(latent_ptr + offs_lat,
                 block,
                 mask=head_mask[:, None] & d_mask[None, :])


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
    Optimised implementation for the case `qk_nope_head_dim == 0`.
    The heavy lifting (attention + latent aggregation) is performed by a
    Triton kernel; the per‑head value projection is delegated to a
    batched GEMM via ``torch.einsum``.
    """
    # ------------------------------------------------------------------
    # 0️⃣ basic shapes / constants
    # ------------------------------------------------------------------
    bs = config.batch_size
    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim
    dim = config.dim

    # ------------------------------------------------------------------
    # 1️⃣ down‑projection and KV‑cache update
    # ------------------------------------------------------------------
    x2 = x.squeeze(1)                                    # (B, Dim)
    q_lora = F.linear(x2, wDQ)                            # (B, dq)
    kv_lora = F.linear(x2, wDKV)                          # (B, dkv + drope)

    # write new token into the cache
    cur_len = kv_cache.seq_len
    new_len = cur_len + 1
    kv_latent_new = kv_lora[:, :dkv]                      # (B, dkv)
    rope_raw_new  = kv_lora[:, dkv:]                      # (B, drope)

    kv_cache.data[:, cur_len:new_len, :dkv] = kv_latent_new
    kv_cache.data[:, cur_len:new_len, dkv:] = rope_raw_new
    kv_cache.seq_len = new_len

    # ------------------------------------------------------------------
    # 2️⃣ up‑project queries and apply RoPE (single token)
    # ------------------------------------------------------------------
    q_up = F.linear(q_lora, wUQ)                          # (B, nh*drope)
    q_up = q_up.view(bs, nh, drope)                      # (B, nh, drope)

    # query position = newest token index
    q_pos = new_len - 1
    cos_q = cos_tbl[q_pos]                                # (drope,)
    sin_q = sin_tbl[q_pos]                                # (drope,)
    q_rope = q_up * cos_q + _rotate_half(q_up) * sin_q    # (B, nh, drope)

    # ------------------------------------------------------------------
    # 3️⃣ gather cached KV (latent + raw rope part)
    # ------------------------------------------------------------------
    kv_all = kv_cache.data[:, :new_len, :]                # (B, L, dkv+drope)
    k_rope_raw = kv_all[..., dkv:]                        # (B, L, drope)   <-- raw
    v_latent   = kv_all[..., :dkv]                        # (B, L, dkv)

    # ------------------------------------------------------------------
    # 4️⃣ allocate buffer for the latent (attention‑weighted sum of V)
    # ------------------------------------------------------------------
    latent_buf = torch.empty((bs, nh, dkv), dtype=torch.bfloat16, device=x.device)

    # ------------------------------------------------------------------
    # 5️⃣ launch Triton kernel
    # ------------------------------------------------------------------
    # ---- compile‑time block sizes (tuned experimentally) ----
    HEADS_PER_BLOCK = 32                 # 128 heads → 4 program tiles per batch
    BLOCK_K = 256                         # KV positions per iteration
    BLOCK_DKV = 64                        # tile on the latent dimension

    # strides (in elements)
    stride_q_batch, stride_q_head, stride_q_dim = q_rope.stride()
    stride_k_batch, stride_k_len , stride_k_dim  = k_rope_raw.stride()
    stride_cos_seq, stride_cos_dim = cos_tbl.stride()
    stride_sin_seq, stride_sin_dim = sin_tbl.stride()
    stride_v_batch, stride_v_len , stride_v_dim = v_latent.stride()
    stride_lat_batch, stride_lat_head, stride_lat_dim = latent_buf.stride()

    scale = 1.0 / math.sqrt(drope)        # d_nope == 0

    grid = (bs, math.ceil(nh / HEADS_PER_BLOCK))

    _triton_attn_latent_kernel_opt_rope[grid](
        # pointers
        q_rope,
        k_rope_raw,
        v_latent,
        cos_tbl,
        sin_tbl,
        latent_buf,
        # strides
        stride_q_batch, stride_q_head, stride_q_dim,
        stride_k_batch, stride_k_len , stride_k_dim,
        stride_cos_seq, stride_cos_dim,
        stride_sin_seq, stride_sin_dim,
        stride_v_batch, stride_v_len , stride_v_dim,
        stride_lat_batch, stride_lat_head, stride_lat_dim,
        # compile‑time constants
        bs, nh, new_len, drope, dkv,
        scale,
        HEADS_PER_BLOCK, BLOCK_K, BLOCK_DKV,
        # launch configuration
        num_warps=8, num_stages=4,
    )

    # ------------------------------------------------------------------
    # 6️⃣ per‑head value projection (batched GEMM via einsum)
    # ------------------------------------------------------------------
    # wV_T : (nh, dkv, dv) – transpose of the original up‑projection weight
    wV_T = wUKV.view(nh, dv, dkv).permute(0, 2, 1).contiguous()   # (nh, dkv, dv)

    # latent_buf : (B, nh, dkv)  (bfloat16)
    # result y_head : (B, nh, dv)
    y_head = torch.einsum('bhd,hdf->bhf', latent_buf, wV_T)       # bfloat16 matmul

    # ------------------------------------------------------------------
    # 7️⃣ final linear projection back to model dimension
    # ------------------------------------------------------------------
    y_head_flat = y_head.reshape(bs, nh * dv)      # (B, nh*dv)
    out = F.linear(y_head_flat, wO)                # (B, dim)
    out = out.unsqueeze(1)                         # (B, 1, dim)

    return out, kv_cache.data


# ----------------------------------------------------------------------
# Compiled fallback for the general case (d_nope > 0)
# ----------------------------------------------------------------------
_compiled_forward: torch._dynamo.eval_frame.OptimizedModule = None


def _build_compiled_forward():
    """Compiled reference implementation handling the generic case."""
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
        # identical to the reference version – unchanged
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
        kv_latent   = kv_lora[..., :dkv]                                 # (bs, kv_len, dkv)

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
    Expected signature by the benchmark harness.
    """
    config, x, kv_cache = data

    # --------------------------------------------------------------
    # Extract scalar config values
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
    # Rename weight tensors for brevity (they are already stored in the Config)
    # --------------------------------------------------------------
    wDQ  = config.Q_proj_down_weight          # (dq, dim)
    wDKV = config.KV_proj_down_weight          # (dkv + drope, dim)
    wUQ  = config.Q_proj_up_weight            # ((d_nope + drope) * nh, dq)
    wUKV = config.KV_proj_up_weight          # ((d_nope + dv) * nh, dkv)
    wO   = config.wo_weight                   # (dim, nh * dv)

    # --------------------------------------------------------------
    # Build / fetch the global RoPE tables (cos / sin) – one‑time per device
    # --------------------------------------------------------------
    global _cached_cos, _cached_sin
    if (_cached_cos is None) or (_cached_cos.shape[0] < config.max_seq_len):
        _cached_cos, _cached_sin = _get_rope_tables(drope,
                                                    config.max_seq_len,
                                                    x.device)

    # --------------------------------------------------------------
    # Fast‑path – the common configuration has d_nope == 0
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

    # output already shaped (B, 1, Dim)
    return out, kv_cache.data