"""
TriMul (outgoing) forward pass with a Triton mask kernel.

The heavy reduction
    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
is realized as a batched half‑precision GEMM (fp16) which on an H100
executes with Tensor‑Core speed.  The optional pairwise mask is applied
with a small Triton kernel that multiplies a (B,N,N,H) tensor by a
(B,N,N) mask broadcasting over the hidden dimension.  All linear
projections, gates and LayerNorms are performed in fp16 (wherever
possible) and the final result is returned in float32.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mask_mul_kernel(x_ptr, mask_ptr, out_ptr,
                    total_elems, hidden_dim,
                    BLOCK_SIZE: tl.constexpr):
    """
    out = x * mask    (mask is broadcast over the hidden dim)
    x : [total_elems]   (flattened B,N,N,H)
    mask : [total_elems // hidden_dim]   (flattened B,N,N)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # bounds
    valid = offs < total_elems
    mask_len = total_elems // hidden_dim
    mask_idx = offs // hidden_dim
    mask_valid = mask_idx < mask_len

    # load
    x = tl.load(x_ptr + offs, mask=valid, other=0.0)
    m = tl.load(mask_ptr + mask_idx, mask=mask_valid, other=0.0)

    # store
    tl.store(out_ptr + offs, x * m, mask=valid)


def _apply_mask_triton(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a (B,N,N) mask to a (B,N,N,H) tensor using Triton.
    Both tensors must be on the same CUDA device and dtype float16.
    """
    assert tensor.is_cuda and mask.is_cuda
    assert tensor.dtype == torch.float16 and mask.dtype == torch.float16
    total = tensor.numel()
    hidden = tensor.shape[-1]

    # Ensure contiguous layout
    x = tensor.contiguous()
    m = mask.contiguous().view(-1)          # flatten to (B*N*N,)
    out = torch.empty_like(x)

    BLOCK = 65536  # tuned for H100 (≈2^16 threads per kernel launch)
    grid = (triton.cdiv(total, BLOCK),)
    mask_mul_kernel[grid](x, m, out,
                          total, hidden,
                          BLOCK_SIZE=BLOCK)
    return out


def custom_kernel(data):
    """
    Triton‑accelerated forward of the TriMul “outgoing” operator.

    Args:
        data: Tuple (input, mask, weights, config)
            - input: Tensor [B, N, N, dim] (float32)
            - mask:  Tensor [B, N, N]   (float32; may be ignored)
            - weights: dict of model parameters
            - config: dict containing "dim", "hidden_dim", "nomask"

    Returns:
        Tensor [B, N, N, dim] (float32)
    """
    # unpack
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-6

    # ------------------------------------------------------------------
    # 1. Input LayerNorm (float32)
    # ------------------------------------------------------------------
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
        eps=eps,
    )  # [B,N,N,dim] float32

    # ------------------------------------------------------------------
    # 2. Cast to fp16 for the remaining heavy ops
    # ------------------------------------------------------------------
    x_fp16 = x.to(torch.float16)

    B, N, _, _ = x.shape
    flat_x = x_fp16.view(-1, dim)                # (B*N*N, dim)

    # ------------------------------------------------------------------
    # 3. Linear projections (no bias)
    # ------------------------------------------------------------------
    left_proj_w = weights["left_proj.weight"].to(torch.float16)
    right_proj_w = weights["right_proj.weight"].to(torch.float16)

    left = F.linear(flat_x, left_proj_w)        # (B*N*N, hidden_dim)
    right = F.linear(flat_x, right_proj_w)

    left = left.view(B, N, N, hidden_dim)       # (B,N,N,hidden_dim)
    right = right.view(B, N, N, hidden_dim)

    # ------------------------------------------------------------------
    # 4. Optional pairwise mask (Triton kernel)
    # ------------------------------------------------------------------
    if not config.get("nomask", True):
        # mask is float32 → fp16
        mask_fp16 = mask.to(torch.float16, device=device)
        left = _apply_mask_triton(left, mask_fp16)
        right = _apply_mask_triton(right, mask_fp16)

    # ------------------------------------------------------------------
    # 5. Gates (sigmoid(linear(x)))
    # ------------------------------------------------------------------
    left_gate_w = weights["left_gate.weight"].to(torch.float16)
    right_gate_w = weights["right_gate.weight"].to(torch.float16)
    out_gate_w = weights["out_gate.weight"].to(torch.float16)

    left_gate = torch.sigmoid(F.linear(flat_x, left_gate_w)).view(B, N, N, hidden_dim)
    right_gate = torch.sigmoid(F.linear(flat_x, right_gate_w)).view(B, N, N, hidden_dim)
    out_gate = torch.sigmoid(F.linear(flat_x, out_gate_w)).view(B, N, N, hidden_dim)

    # ------------------------------------------------------------------
    # 6. Apply gates to the projected tensors
    # ------------------------------------------------------------------
    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 7. Batched GEMM: out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    #    Implemented as torch.matmul on (B, hidden_dim, N, N) tensors.
    # ------------------------------------------------------------------
    #   left : (B,N,N,H)  → (B,H,N,N)
    #   right: (B,N,N,H) → (B,H,N,N) after transposing the k‑axis
    left_perm = left.permute(0, 3, 1, 2)          # (B, H, N, N)
    right_perm = right.permute(0, 3, 2, 1)        # (B, H, N, N)
    out_perm = torch.matmul(left_perm, right_perm)  # (B, H, N, N)
    out = out_perm.permute(0, 2, 3, 1).contiguous()  # (B, N, N, H)

    # ------------------------------------------------------------------
    # 8. Output LayerNorm over hidden_dim (still fp16)
    # ------------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(torch.float16)
    to_out_norm_b = weights["to_out_norm.bias"].to(torch.float16)
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=eps,
    )

    # ------------------------------------------------------------------
    # 9. Multiply by the output gate (sigmoid)
    # ------------------------------------------------------------------
    out = out * out_gate

    # ------------------------------------------------------------------
    # 10. Final linear projection back to original dim (half‑precision)
    # ------------------------------------------------------------------
    to_out_w = weights["to_out.weight"].to(torch.float16)
    out_flat = out.view(-1, hidden_dim)         # (B*N*N, hidden_dim)
    out_proj = F.linear(out_flat, to_out_w)     # (B*N*N, dim)
    out_proj = out_proj.view(B, N, N, dim)

    # ------------------------------------------------------------------
    # 11. Return float32 tensor
    # ------------------------------------------------------------------
    return out_proj.to(torch.float32)