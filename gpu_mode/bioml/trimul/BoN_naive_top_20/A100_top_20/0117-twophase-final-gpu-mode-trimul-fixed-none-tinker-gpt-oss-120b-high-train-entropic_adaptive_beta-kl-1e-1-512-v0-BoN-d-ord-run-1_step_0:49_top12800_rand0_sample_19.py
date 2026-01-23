"""
TriMul "outgoing" forward pass (AlphaFold3) with a custom Triton kernel.

Algorithm
---------
1. Layer‑norm the 4‑D input tensor (shape [B, N, N, C]).
2. Cast to float16 for speed and apply linear projections:
   left = x @ W_left   (C -> H)
   right = x @ W_right (C -> H)
   left_gate  = sigmoid(x @ W_left_gate)
   right_gate = sigmoid(x @ W_right_gate)
   out_gate   = sigmoid(x @ W_out_gate)
3. If a mask is present, fuse mask application and gating in a Triton kernel:
      left  = left  * left_gate  * mask
      right = right * right_gate * mask
   (mask is broadcast over the hidden dimension H.)
   If no mask, simply multiply by the gates (pure PyTorch).
4. Compute the outgoing pairwise multiplication:
      out[i, j] = Σ_k left[i, k] * right[j, k]
   This is implemented as a batched matrix‑multiply:
      (B, H, N, N) @ (B, H, N, N)^T → (B, H, N, N)
   and then permuted back to shape [B, N, N, H].
5. Layer‑norm over the hidden dimension H (to_out_norm),
   multiply by the out‑gate, and final linear projection back to C
   (to_out).
6. Cast the result to float32 and return.

Only the mask‑gate fusion (step 3) is executed in a custom Triton kernel;
all heavy matrix‑multiplications use highly‑optimized cuBLAS kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_gate_kernel(
    src_ptr, gate_ptr, mask_ptr,
    B, N, H,
    stride_src_b, stride_src_i, stride_src_j, stride_src_h,
    stride_gate_b, stride_gate_i, stride_gate_j, stride_gate_h,
    stride_mask_b, stride_mask_i, stride_mask_j,
    BLOCK_SIZE: tl.constexpr,
):
    """In‑place: src = src * gate * mask (mask broadcasted on H)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * N * N * H
    active = offs < total

    # ---- 4‑D index from flat offset ----
    b = offs // (N * N * H)
    rem = offs % (N * N * H)
    i = rem // (N * H)
    rem = rem % (N * H)
    j = rem // H
    h = rem % H

    # ---- compute addresses ----
    src = src_ptr + b * stride_src_b + i * stride_src_i + j * stride_src_j + h * stride_src_h
    gate = gate_ptr + b * stride_gate_b + i * stride_gate_i + j * stride_gate_j + h * stride_gate_h
    mask_addr = mask_ptr + b * stride_mask_b + i * stride_mask_i + j * stride_mask_j

    # ---- loads (masked) ----
    src_val = tl.load(src, mask=active, other=0.0)
    gate_val = tl.load(gate, mask=active, other=0.0)
    # mask is scalar per (b,i,j); broadcast over h
    mask_val = tl.load(mask_addr, mask=active, other=1.0)

    out_val = src_val * gate_val * mask_val
    tl.store(src, out_val, mask=active)


def _apply_mask_gate(src: torch.Tensor,
                     gate: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """
    Apply gate and mask in-place using the Triton kernel.
    All tensors must be on the same device, contiguous and of the same dtype.
    """
    assert src.is_contiguous() and gate.is_contiguous() and mask.is_contiguous()
    B, N, _, H = src.shape
    total = B * N * N * H
    BLOCK = 1024  # max threads per block on H100
    grid = (triton.cdiv(total, BLOCK),)

    _mask_gate_kernel[grid](
        src,
        gate,
        mask,
        B,
        N,
        H,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        src.stride(3),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        gate.stride(3),
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return src


def custom_kernel(data):
    """
    Forward pass of the TriMul “outgoing” operator.

    Args:
        data: Tuple (input_tensor, mask, weights, config)
            - input_tensor: torch.Tensor of shape [B, N, N, C]
            - mask: torch.Tensor of shape [B, N, N] (may be None)
            - weights: dict of model weights
            - config: dict with keys "dim", "hidden_dim", "nomask" (optional)

    Returns:
        torch.Tensor of shape [B, N, N, C] (float32)
    """
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    # ----- extract weights (they are float32) -----
    norm_w = weights["norm.weight"]
    norm_b = weights["norm.bias"]
    left_proj_w = weights["left_proj.weight"]
    right_proj_w = weights["right_proj.weight"]
    left_gate_w = weights["left_gate.weight"]
    right_gate_w = weights["right_gate.weight"]
    out_gate_w = weights["out_gate.weight"]
    to_out_norm_w = weights["to_out_norm.weight"]
    to_out_norm_b = weights["to_out_norm.bias"]
    to_out_w = weights["to_out.weight"]

    # ----- 1️⃣ LayerNorm over the last dim (C) -----
    x = torch.nn.functional.layer_norm(
        input_tensor, (dim,), weight=norm_w, bias=norm_b
    )  # [B, N, N, C]  (float32)

    # ----- 2️⃣ Cast to fp16 for the heavy compute -----
    compute_dtype = torch.float16
    x = x.to(compute_dtype)

    # ----- cast all linear weights to fp16 -----
    left_proj_w = left_proj_w.to(compute_dtype)
    right_proj_w = right_proj_w.to(compute_dtype)
    left_gate_w = left_gate_w.to(compute_dtype)
    right_gate_w = right_gate_w.to(compute_dtype)
    out_gate_w = out_gate_w.to(compute_dtype)
    to_out_norm_w = to_out_norm_w.to(compute_dtype)
    to_out_norm_b = to_out_norm_b.to(compute_dtype)
    to_out_w = to_out_w.to(compute_dtype)

    # ----- 3️⃣ Linear projections -----
    # each returns [B, N, N, hidden_dim]
    left = torch.nn.functional.linear(x, left_proj_w)
    right = torch.nn.functional.linear(x, right_proj_w)

    left_gate = torch.sigmoid(torch.nn.functional.linear(x, left_gate_w))
    right_gate = torch.sigmoid(torch.nn.functional.linear(x, right_gate_w))
    out_gate = torch.sigmoid(torch.nn.functional.linear(x, out_gate_w))

    # ----- 4️⃣ Apply mask + gate (Triton) or just gate -----
    if not nomask and mask is not None:
        # ensure mask has same device and dtype as compute tensors
        mask = mask.to(compute_dtype)
        left = _apply_mask_gate(left.contiguous(), left_gate.contiguous(), mask)
        right = _apply_mask_gate(right.contiguous(), right_gate.contiguous(), mask)
    else:
        left = left * left_gate
        right = right * right_gate

    # ----- 5️⃣ Outgoing pairwise multiplication (einsum) -----
    # reshape to (B, H, N, N) and use batched matmul
    B, N, _, H = left.shape
    left_perm = left.permute(0, 3, 1, 2)   # [B, H, N, N]
    right_perm = right.permute(0, 3, 1, 2) # [B, H, N, N]

    # out_perm: [B, H, N, N] = left_perm @ right_perm^T
    out_perm = torch.matmul(left_perm, right_perm.transpose(-1, -2))
    # back to [B, N, N, H]
    out = out_perm.permute(0, 2, 3, 1)

    # ----- 6️⃣ to_out_norm (LayerNorm over hidden_dim) -----
    out = torch.nn.functional.layer_norm(
        out, (hidden_dim,), weight=to_out_norm_w, bias=to_out_norm_b
    )

    # ----- 7️⃣ Apply final out‑gate -----
    out = out * out_gate

    # ----- 8️⃣ Final linear projection back to dim -----
    out = torch.nn.functional.linear(out, to_out_w)

    # ----- 9️⃣ Cast back to float32 for the returned tensor -----
    out = out.to(torch.float32)
    return out