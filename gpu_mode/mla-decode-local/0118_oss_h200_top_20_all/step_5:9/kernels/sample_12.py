### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config   # noqa: F401
### END OF IMPORT STATEMENTS BLOCK ###

# ----------------------------------------------------------------------
#  Helper: RoPE cache (cos & sin tables)
# ----------------------------------------------------------------------
_rope_cache: dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_rope_tables(head_dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, head_dim) in bf16."""
    key = (head_dim, max_seq_len, device)
    if key not in _rope_cache:
        half = head_dim // 2
        # theta = 10000 ** (-i/half)   (float32 → bf16)
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )                                   # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (max_seq_len,1)
        idx = pos * theta[None, :]         # (max_seq_len, half)
        idx = torch.cat([idx, idx], dim=-1)  # (max_seq_len, head_dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]

# ----------------------------------------------------------------------
#  Helper: rotate‑half used by RoPE
# ----------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of the last dimension and negate the second half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

# ----------------------------------------------------------------------
#  Triton soft‑max (fallback – not used when we take the SDPA path)
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
#  Main custom kernel – highly optimised forward of MLA
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward pass of the Multi‑head Latent Attention (MLA) module.
    Returns (output, updated_kv_cache_tensor) where
      - output : torch.Tensor of shape (batch, seq_len, dim)  (bf16)
      - updated_kv_cache_tensor : the tensor stored inside KVCache after the update
    """
    config, x, kv_cache = data

    # ------------------------------------------------------------------
    # 1️⃣  Unpack config & weights
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

    # weight tensors (already on the correct device & dtype==bf16)
    wDQ   = config.Q_proj_down_weight                # (dq, d)
    wDKV  = config.KV_proj_down_weight               # (dkv + d_rope, d)
    wUQ   = config.Q_proj_up_weight                  # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight                 # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                         # (d, nh*dv)

    # ------------------------------------------------------------------
    # 2️⃣  Q & KV down‑projections
    # ------------------------------------------------------------------
    # Down‑project Q
    q_lora = F.linear(x, wDQ)                 # (bs, sl, dq)

    # Down‑project KV (latent + rope part)
    kv_lora_in = F.linear(x, wDKV)            # (bs, sl, dkv + d_rope)

    # ------------------------------------------------------------------
    # 3️⃣  KV cache update
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora_in)    # kv_lora: (bs, kv_len, dkv+d_rope)
    kv_len_int = int(kv_len)                  # scalar – total length after the append

    # ------------------------------------------------------------------
    # 4️⃣  Split latent / rope parts
    # ------------------------------------------------------------------
    # Q up‑projection → (bs, sl, nh, d_nope+d_rope)
    q_up = F.linear(q_lora, wUQ)               # (bs, sl, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, sl, nh, d_nope + d_rope)

    # Because seq_len == 1 we can drop that dimension early
    q_up = q_up.squeeze(1)                    # (bs, nh, d_nope+d_rope)

    # Split Q into no‑PE & RoPE parts
    if d_nope > 0:
        q_nope, q_rope_raw = torch.split(q_up, [d_nope, d_rope], dim=-1)   # (bs, nh, d_nope), (bs, nh, d_rope)
    else:
        q_nope = torch.empty(bs, nh, 0, dtype=torch.bfloat16, device=x.device)
        q_rope_raw = q_up                              # (bs, nh, d_rope)

    # KV split (latent part stays as‑is, rope part will get RoPE)
    kv_latent = kv_lora[..., :dkv]                # (bs, kv_len, dkv)
    k_rope_raw = kv_lora[..., dkv:]               # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣  RoPE tables (cached) – apply to queries and keys
    # ------------------------------------------------------------------
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # ----- Q RoPE (single position) -----
    query_pos = kv_len_int - 1                     # absolute position of newest token
    cos_q = cos_table[query_pos]                   # (d_rope,)
    sin_q = sin_table[query_pos]                   # (d_rope,)
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q   # (bs, nh, d_rope)

    # ----- K RoPE (all cached positions) -----
    # cos/sin for all positions 0 .. kv_len-1
    cos_k = cos_table[:kv_len_int]                 # (kv_len, d_rope)
    sin_k = sin_table[:kv_len_int]                 # (kv_len, d_rope)
    # broadcast to batch dimension
    cos_k = cos_k[None, :, :]                      # (1, kv_len, d_rope)
    sin_k = sin_k[None, :, :]                      # (1, kv_len, d_rope)
    k_rope = k_rope_raw * cos_k + _rotate_half(k_rope_raw) * sin_k   # (bs, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 6️⃣  Attention – use fused Flash‑Attention when no‑PE path is empty
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)

    if d_nope == 0:
        # --------------------------------------------------------------
        # 6a️⃣  Pure RoPE path → torch.nn.functional.scaled_dot_product_attention
        # --------------------------------------------------------------

        # Q : (bs, nh, 1, d_rope)
        q = q_rope.unsqueeze(2)                                   # (bs, nh, 1, d_rope)

        # K : (bs, nh, kv_len, d_rope) – broadcast the same K to every head
        k = k_rope.unsqueeze(1).expand(-1, nh, -1, -1)            # (bs, nh, kv_len, d_rope)

        # V : (bs, nh, kv_len, dkv) – same latent values for every head
        v = kv_latent.unsqueeze(1).expand(-1, nh, -1, -1)         # (bs, nh, kv_len, dkv)

        # Flash‑Attention (single‑query case)
        #   output shape → (bs, nh, 1, dkv)
        m = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False, dropout_p=0.0
        )
        m = m.squeeze(2)                                          # (bs, nh, dkv)

    else:
        # --------------------------------------------------------------
        # 6b️⃣  General case – compute separate No‑PE scores + RoPE scores
        # --------------------------------------------------------------

        # --- Q no‑PE -----
        # Map Q_nope → latent space via per‑head weight wK (first d_nope rows of wUKV)
        wK = wUKV.view(nh, d_nope + dv, dkv)[:, :d_nope, :]       # (nh, d_nope, dkv)
        # (bs, nh, dkv) = (bs, nh, d_nope) @ (nh, d_nope, dkv)
        q_nope_latent = torch.einsum('bhd,hdm->bhm', q_nope, wK)   # (bs, nh, dkv)

        # --- Scores (latent part) ---
        scores_nope = torch.matmul(q_nope_latent, kv_latent.transpose(-1, -2))   # (bs, nh, kv_len)

        # --- RoPE scores ---
        scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))              # (bs, nh, kv_len)

        # --- Combine & scale ---
        scores = (scores_nope + scores_rope) * scale
        scores = scores.to(torch.bfloat16)

        # --- Softmax (Triton) ---
        scores_2d = scores.view(bs * nh, kv_len_int)               # (B*H, L)
        attn_2d = _triton_softmax(scores_2d)                      # (B*H, L)
        attn = attn_2d.view(bs, nh, kv_len_int)                   # (bs, nh, kv_len)

        # --- Weighted sum of latent keys → M (bs, nh, dkv) ---
        m = torch.matmul(attn, kv_latent)                         # (bs, nh, dkv)

    # ------------------------------------------------------------------
    # 7️⃣  Project latent result to per‑head values (dv)
    # ------------------------------------------------------------------
    # wV_T : (nh, dkv, dv)
    wV_T = wUKV.view(nh, d_nope + dv, dkv)[:, d_nope:, :].permute(0, 2, 1)   # (nh, dkv, dv)

    # y_head : (bs, nh, dv)
    y_head = torch.einsum('bhd,hdv->bhv', m, wV_T)            # (bs, nh, dv)

    # ------------------------------------------------------------------
    # 8️⃣  Merge heads & final linear projection
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, -1)                                # (bs, nh*dv)
    output = F.linear(y, wO)                                  # (bs, d)
    output = output.unsqueeze(1)                               # (bs, 1, d)

    # ------------------------------------------------------------------
    # Return the output and the *updated* KV‑cache tensor
    # ------------------------------------------------------------------
    return output, kv_cache.data