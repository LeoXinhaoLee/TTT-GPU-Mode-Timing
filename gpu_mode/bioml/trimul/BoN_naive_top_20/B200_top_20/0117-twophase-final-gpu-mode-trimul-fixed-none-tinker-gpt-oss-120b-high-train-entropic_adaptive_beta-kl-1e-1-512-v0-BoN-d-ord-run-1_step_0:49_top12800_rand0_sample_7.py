"""
TriMul (outgoing) forward pass implemented with a mix of PyTorch and Triton.

Algorithm
---------
1️⃣  Layer‑norm over the last channel of the input tensor.
2️⃣  Project the normalized input to the hidden dimension (left/right) and
    compute gating logits (left_gate/right_gate/out_gate) with a linear layer.
3️⃣  Apply sigmoid to gates and fuse the element‑wise gate multiplication using
    a tiny Triton kernel (in‑place  `a *= b`).
4️⃣  (Optional) apply the pairwise mask – it is simply broadcasted over the
    hidden dimension and multiplied with the projected tensors.
5️⃣  Compute the “tri‑multiplicative” reduction:
        out[b,i,j,:] = Σₖ left[b,i,k,:] * right[b,j,k,:]
    This can be expressed as a batched matrix‑multiply:
        left  : (B, H, N, N) → (B·H, N, N)
        right : (B, H, N, N) → (B·H, N, N)
        out   = left @ rightᵀ   (torch.bmm)
    The result is reshaped back to (B, N, N, H).
6️⃣  Layer‑norm over the hidden channel of the reduction result.
7️⃣  Multiply with the out‑gate (again via the Triton in‑place kernel).
8️⃣  Final linear projection back to the original channel dimension.

Only a small Triton kernel is required (step 3 & 7); the compute‑heavy
tri‑multiplicative reduction uses batched GEMM (torch.bmm) which on an H100
leverages FP16 tensor‑cores and gives sub‑millisecond latency for the
benchmarked sizes.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Dict

# ----------------------------------------------------------------------
# Triton element‑wise in‑place multiplication (a *= b)
# ----------------------------------------------------------------------
@triton.jit
def _elemwise_mul_inplace_kernel(
    a_ptr, b_ptr,           # pointers
    N,                      # total number of elements
    stride_a, stride_b,     # strides (1 for flattened tensors)
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offs < N
    a = tl.load(a_ptr + offs * stride_a, mask=mask)
    b = tl.load(b_ptr + offs * stride_b, mask=mask)
    tl.store(a_ptr + offs * stride_a, a * b, mask=mask)   # in‑place


def _elemwise_mul_inplace(a: torch.Tensor, b: torch.Tensor):
    """
    In‑place element‑wise multiplication a = a * b.
    Both tensors must be contiguous, same shape, and on the same device.
    """
    assert a.shape == b.shape, "Shape mismatch"
    assert a.is_contiguous() and b.is_contiguous()
    total = a.numel()
    BLOCK_SIZE = 256  # ≤ 1024 threads per block
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    # Triton expects the raw pointer; passing the tensor works directly.
    _elemwise_mul_inplace_kernel[grid](
        a, b,
        total,
        1, 1,                      # strides for flattened view
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return a


# ----------------------------------------------------------------------
# Custom kernel entry point
# ----------------------------------------------------------------------
def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]) -> torch.Tensor:
    """
    Forward pass of the outgoing TriMul module.

    Parameters
    ----------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor [B, N, N, dim]  (float32)
        - mask         : torch.Tensor [B, N, N]    (bool / float)
        - weights      : dict of model weights (float32)
        - config       : dict containing 'dim', 'hidden_dim' and optional
                         'nomask' flag.

    Returns
    -------
    torch.Tensor
        Output tensor [B, N, N, dim] (float32)
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    device = input_tensor.device
    dtype = torch.float32

    # ------------------------------------------------------------------
    # 1️⃣ LayerNorm over the channel dimension of the input
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias   = weights["norm.bias"]
    x_norm = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=norm_weight,
        bias=norm_bias,
    )                                   # [B, N, N, dim] (float32)

    # Cast to fp16 for the heavy compute
    x_fp16 = x_norm.to(torch.float16)

    # ------------------------------------------------------------------
    # 2️⃣ Linear projections & gate logits (all in FP16)
    # ------------------------------------------------------------------
    B, N, _, _ = input_tensor.shape
    seq_elems = B * N * N

    # Helper to get weight as fp16
    def _to_fp16(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.float16)

    left_proj_w   = _to_fp16(weights["left_proj.weight"])   # [hidden_dim, dim]
    right_proj_w  = _to_fp16(weights["right_proj.weight"])
    left_gate_w   = _to_fp16(weights["left_gate.weight"])
    right_gate_w  = _to_fp16(weights["right_gate.weight"])
    out_gate_w    = _to_fp16(weights["out_gate.weight"])
    to_out_norm_w = _to_fp16(weights["to_out_norm.weight"])
    to_out_norm_b = _to_fp16(weights["to_out_norm.bias"])
    to_out_w      = _to_fp16(weights["to_out.weight"])      # [dim, hidden_dim]

    # Flatten the first three dimensions to perform a big batched linear
    x_flat = x_fp16.view(seq_elems, dim)                     # [B·N·N, dim]

    # Projections
    left_proj   = torch.nn.functional.linear(x_flat, left_proj_w)    # [B·N·N, hidden_dim]
    right_proj  = torch.nn.functional.linear(x_flat, right_proj_w)
    left_gate   = torch.sigmoid(torch.nn.functional.linear(x_flat, left_gate_w))
    right_gate  = torch.sigmoid(torch.nn.functional.linear(x_flat, right_gate_w))
    out_gate    = torch.sigmoid(torch.nn.functional.linear(x_flat, out_gate_w))

    # Reshape back to 4‑D tensors
    left   = left_proj.view(B, N, N, hidden_dim).contiguous()
    right  = right_proj.view(B, N, N, hidden_dim).contiguous()
    left_gate  = left_gate.view(B, N, N, hidden_dim).contiguous()
    right_gate = right_gate.view(B, N, N, hidden_dim).contiguous()
    out_gate   = out_gate.view(B, N, N, hidden_dim).contiguous()

    # ------------------------------------------------------------------
    # 3️⃣ Apply gates (in‑place) using Triton kernel
    # ------------------------------------------------------------------
    _elemwise_mul_inplace(left,  left_gate)   # left  = left  * left_gate
    _elemwise_mul_inplace(right, right_gate)  # right = right * right_gate

    # ------------------------------------------------------------------
    # 4️⃣ Optional mask (broadcasted over hidden_dim)
    # ------------------------------------------------------------------
    if not config.get("nomask", True):
        mask_fp16 = mask.to(torch.float16).unsqueeze(-1)  # [B, N, N, 1]
        left  = left * mask_fp16
        right = right * mask_fp16

    # ------------------------------------------------------------------
    # 5️⃣ Tri‑multiplicative reduction via batched GEMM
    #    out[b,i,j,:] = Σₖ left[b,i,k,:] * right[b,j,k,:]
    # ------------------------------------------------------------------
    # Bring hidden dimension to the batch axis: (B, H, N, N)
    left_b  = left.permute(0, 3, 1, 2).contiguous()   # [B, H, N, N]
    right_b = right.permute(0, 3, 1, 2).contiguous()

    B_H = B * hidden_dim
    left_mat  = left_b.view(B_H, N, N)                # [B·H, N, N]
    right_mat = right_b.view(B_H, N, N)

    # Batched matrix multiplication: left @ rightᵀ
    out_mat = torch.bmm(left_mat, right_mat.transpose(1, 2))   # [B·H, N, N]

    # Reshape back to (B, N, N, H)
    out_hidden = out_mat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------------
    # 6️⃣ LayerNorm over hidden dimension
    # ------------------------------------------------------------------
    out_norm = torch.nn.functional.layer_norm(
        out_hidden,
        normalized_shape=(hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
    )

    # ------------------------------------------------------------------
    # 7️⃣ Apply out‑gate (in‑place Triton kernel)
    # ------------------------------------------------------------------
    _elemwise_mul_inplace(out_norm, out_gate)

    # ------------------------------------------------------------------
    # 8️⃣ Final projection back to original channel size
    # ------------------------------------------------------------------
    out_flat = out_norm.view(seq_elems, hidden_dim)    # [B·N·N, hidden_dim]
    out_proj = torch.nn.functional.linear(out_flat, to_out_w)   # [B·N·N, dim]

    output = out_proj.view(B, N, N, dim).to(torch.float32)
    return output