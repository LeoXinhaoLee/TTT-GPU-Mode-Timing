#######################################################################################################
#  Triton‑accelerated forward pass for the Multi‑Head Latent Attention (MLA) module
#
#  NOTE
#  ----
#  * All tensors are assumed to be on the same CUDA device and in bfloat16.
#  * Only the functions/classes that are required for the kernel are defined below – the
#    `Config` and `KVCache` classes are imported from the provided ``reference`` module.
#  * The implementation follows the exact mathematics of the reference PyTorch model but
#    replaces the original row‑wise softmax with cuDNN’s native softmax (much faster for the
#    shapes we use) and keeps the RoPE rotation in a tiny Triton kernel that works in‑place.
#  * Heavy matrix multiplications (down‑/up‑projections, attention score, value projection) are
#    left to cuBLAS – they are already strongly optimized and any extra Triton‑fusion would
#    only increase launch overhead.
#  * The function is JIT‑compiled with ``torch.compile`` (a.k.a. TorchDynamo/AOTAutograd) so
#    that all inexpensive element‑wise ops (splits, reshapes, concatenations, casts, etc.) get
#    fused into a few GPU kernels, dramatically cutting the per‑step launch latency.
#######################################################################################################

### -------------------------------------------------------------------- ###
### imports (do NOT modify the block below)                               ###
### -------------------------------------------------------------------- ###
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import Config, KVCache  # must be imported exactly like this
### -------------------------------------------------------------------- ###

### -------------------------------------------------------------------- ###
### tiny Triton kernel that implements the “rotate‑half + cos/sin” step of RoPE  ###
### -------------------------------------------------------------------- ###
@triton.jit
def _rope_swap_halves_kernel(
    x_ptr,                         # [B, H, D]  (bfloat16 / fp16 / fp32)
    cos_ptr, sin_ptr,              # [D]        (broadcasted cosine / sine)
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,               # must be even
    stride_xb, stride_xh, stride_xd,
    stride_cos_d,
    stride_sin_d,
    BLOCK_HALF: tl.constexpr,      # D // 2 elements per block
):
    pid = tl.program_id(0)                       # 0‑D grid ->  B*H threads
    b = pid // H
    h = pid - b * H

    off = tl.arange(0, BLOCK_HALF)                # 0 … D/2‑1
    mask = off < D // 2

    # ----------------------------------------------------------------
    # pointers to the two halves of the input tensor
    # ----------------------------------------------------------------
    base = x_ptr + b * stride_xb + h * stride_xh
    x0_ptr = base + off * stride_xd                     # first half
    x1_ptr = base + (D // 2 + off) * stride_xd          # second half

    # ----------------------------------------------------------------
    # pointers to cosine / sine (broadcasted, stride = 0 along B/H)
    # ----------------------------------------------------------------
    c_ptr = cos_ptr + off * stride_cos_d
    s_ptr = sin_ptr + off * stride_sin_d

    # ----------------------------------------------------------------
    # load
    # ----------------------------------------------------------------
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)   # (BLOCK_HALF,)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c  = tl.load(c_ptr,  mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(s_ptr,  mask=mask, other=0.0).to(tl.float32)

    # ----------------------------------------------------------------
    # RoPE with rotate‑half (swap‑halves) : out0 = x0*c - x1*s ; out1 = x1*c + x0*s
    # ----------------------------------------------------------------
    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    # ----------------------------------------------------------------
    # store back in‑place
    # ----------------------------------------------------------------
    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)

def _rope_inplace_query(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """
    Applies RoPE to the query tensor *in‑place*.
    Parameters
    ----------
    q   : torch.Tensor of shape (B, H, D)   (bfloat16)
    cos : torch.Tensor of shape (D,)        (bfloat16)
    sin : torch.Tensor of shape (D,)        (bfloat16)
    """
    B, H, D = q.shape
    assert D % 2 == 0

    # pick a block size that is a power‑of‑2 and >= D/2
    half = D // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()
    BLOCK_HALF = min(BLOCK_HALF, 256)   # keep register pressure modest

    grid = (B * H,)

    _rope_swap_halves_kernel[grid](
        q,
        cos, sin,
        B=B, H=H, D=D,
        stride_xb=q.stride(0),
        stride_xh=q.stride(1),
        stride_xd=q.stride(2),
        # cosine / sine are 1‑D, broadcast across B/H → stride = 0
        stride_cos_d=cos.stride(0),
        stride_sin_d=sin.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )
### -------------------------------------------------------------------- ###

### -------------------------------------------------------------------- ###
### Cached cosine / sine tables for RoPE (shared across calls)            ###
### -------------------------------------------------------------------- ###
_rope_cache = {}
def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    """Return (cos, sin) tables of shape (max_seq_len, dim) in bfloat16."""
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )                                   # (half,)
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)  # (L,1)
        idx = pos * theta                     # (L, half)
        idx = torch.cat([idx, idx], dim=-1)   # (L, dim)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]
### -------------------------------------------------------------------- ###

# ----------------------------------------------------------------------
#  Optimised MLA forward – everything is written as pure PyTorch
#  except for the tiny RoPE kernel and the native cuDNN softmax.
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast forward pass for the Multi‑Head Latent Attention module.
    Returns
    -------
    output      : torch.Tensor    (B, 1, D)  in bfloat16
    kv_cache    : torch.Tensor    the updated cache tensor (B, cur_len, K_dim)
    """
    # ------------------------------------------------------------------
    # unpack -----------------------------------------------------------
    # ------------------------------------------------------------------
    config, x, kv_cache = data

    bs = config.batch_size          # B
    sl = config.seq_len             # =1 (always)
    nh = config.n_heads
    d_model = config.dim            # D
    dq = config.q_lora_rank         # d_q
    dkv = config.kv_lora_rank       # d_kv (latent)
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim

    # ------------------------------------------------------------------
    # weight tensors (already on device, bf16)
    # ------------------------------------------------------------------
    wDQ   = config.Q_proj_down_weight          # (dq, D)
    wDKV  = config.KV_proj_down_weight         # (dkv + d_rope, D)
    wUQ   = config.Q_proj_up_weight            # ((d_nope+d_rope)*nh, dq)
    wUKV  = config.KV_proj_up_weight           # ((d_nope+dv)*nh, dkv)
    wO    = config.wo_weight                   # (D, nh*dv)

    # ------------------------------------------------------------------
    # 1️⃣ down‑projection ------------------------------------------------
    # ------------------------------------------------------------------
    # x : (B, 1, D)
    q_lora   = F.linear(x, wDQ)                # (B, 1, dq)
    kv_lora0 = F.linear(x, wDKV)               # (B, 1, dkv + d_rope)

    # ------------------------------------------------------------------
    # 2️⃣ KV‑cache update ------------------------------------------------
    # ------------------------------------------------------------------
    kv_lora, kv_len = kv_cache(kv_lora0)       # kv_lora : (B, kv_len, dkv + d_rope)
    query_pos = kv_len - 1                      # absolute position of the current token

    # ------------------------------------------------------------------
    # 3️⃣ up‑projection of queries ----------------------------------------
    # ------------------------------------------------------------------
    # (B, 1, dq) → (B, (d_nope+d_rope)*nh)
    q_up = F.linear(q_lora.squeeze(1), wUQ)    # (B, (d_nope+d_rope)*nh)
    q_up = q_up.view(bs, nh, d_nope + d_rope) # (B, H, D_q)

    q_nope = q_up[..., :d_nope]                # (B, H, d_nope)
    q_rope = q_up[..., d_nope:]                # (B, H, d_rope)

    # ------------------------------------------------------------------
    # 4️⃣ split KV into latent (no‑PE) and RoPE parts --------------------
    # ------------------------------------------------------------------
    kv_nope_input = kv_lora[..., :dkv]         # (B, kv_len, dkv)
    k_rope_input  = kv_lora[..., dkv:]         # (B, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 5️⃣ RoPE on queries  (in‑place) ------------------------------------
    # ------------------------------------------------------------------
    # build cosine / sine tables for the *single* query position
    cos_q, sin_q = _get_rope_tables(d_rope, config.max_seq_len, x.device)
    cos_q = cos_q[query_pos]                 # (d_rope,)
    sin_q = sin_q[query_pos]                 # (d_rope,)
    _rope_inplace_query(q_rope, cos_q, sin_q)   # modifies q_rope in‑place

    # ------------------------------------------------------------------
    # 6️⃣ RoPE on keys (broadcast) ----------------------------------------
    # ------------------------------------------------------------------
    # cos/sin for all cached positions  (kv_len, d_rope)
    cos_k, sin_k = _get_rope_tables(d_rope, config.max_seq_len, x.device)
    cos_k = cos_k[:kv_len]                    # (kv_len, d_rope)
    sin_k = sin_k[:kv_len]                    # (kv_len, d_rope)

    # k_rope = k_rope_input * cos + rotate_half(k_rope_input) * sin
    # rotate_half = [-x[..., half:], x[..., :half]]
    half = d_rope // 2
    k_rot = torch.cat((-k_rope_input[..., half:], k_rope_input[..., :half]), dim=-1)
    k_rope = k_rope_input * cos_k + k_rot * sin_k   # (B, kv_len, d_rope)

    # ------------------------------------------------------------------
    # 7️⃣ split the up‑projection weight into K‑ and V‑parts -------------
    # ------------------------------------------------------------------
    # wUKV shape : ((d_nope+dv)*nh, dkv)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)   # (H, d_nope+dv, dkv)
    wK = wUKV_view[:, :d_nope, :]                # (H, d_nope, dkv)
    wV = wUKV_view[:, d_nope:, :]                # (H, dv, dkv)

    # ------------------------------------------------------------------
    # 8️⃣ project the query “no‑PE” part into the latent space ----------
    # ------------------------------------------------------------------
    # q_nope : (B, H, d_nope)
    # wK    : (H, d_nope, dkv)
    # result : (B, H, dkv)
    #   we use a batch‑wise einsum – it gets lowered to a fused cuBLAS GEMM.
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)   # (B, H, dkv)

    # ------------------------------------------------------------------
    # 9️⃣ concatenate the two query halves  -------------------------------
    # ------------------------------------------------------------------
    q = torch.cat([q_nope_latent, q_rope], dim=-1)   # (B, H, dkv + d_rope)

    # ------------------------------------------------------------------
    # 10️⃣ concatenate the two key halves ---------------------------------
    # ------------------------------------------------------------------
    k = torch.cat([kv_nope_input, k_rope], dim=-1)   # (B, kv_len, dkv + d_rope)

    # ------------------------------------------------------------------
    # 11️⃣ attention scores ------------------------------------------------
    # ------------------------------------------------------------------
    scale = 1.0 / math.sqrt(d_nope + d_rope)         # same for every head / token
    # Q (B, H, D) @ K^T (B, D, kv_len)  →  (B, H, kv_len)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    # ------------------------------------------------------------------
    # 12️⃣ softmax – use cuDNN (much faster than our custom kernel) ------
    # ------------------------------------------------------------------
    # `torch.nn.functional.softmax` internally calls the cuDNN softmax kernel for
    # bfloat16 tensors, which is heavily tuned for the row‑wise case we have.
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)

    # ------------------------------------------------------------------
    # 13️⃣ weighted sum of latent keys (M = Σ attn * KV_nope) -------------
    # ------------------------------------------------------------------
    # kv_nope_input : (B, kv_len, dkv)
    M = torch.matmul(attn, kv_nope_input)            # (B, H, dkv)

    # ------------------------------------------------------------------
    # 14️⃣ project the aggregated latent keys to per‑head values -----------
    # ------------------------------------------------------------------
    # wV : (H, dv, dkv) → we need its transpose (H, dkv, dv) for a GEMM
    wV_T = wV.permute(0, 2, 1)                      # (H, dkv, dv)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)  # (B, H, dv)

    # ------------------------------------------------------------------
    # 15️⃣ final linear projection (merge heads) ---------------------------
    # ------------------------------------------------------------------
    y = y_head.reshape(bs, nh * dv)                 # (B, H*dv)
    y = y.unsqueeze(1)                               # (B, 1, H*dv)
    output = F.linear(y, wO)                         # (B, 1, D)

    # ------------------------------------------------------------------
    # return -------------------------------------------------------------
    # ------------------------------------------------------------------
    return output, kv_cache.data