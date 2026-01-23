import torch
import triton
import triton.language as tl
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Triton kernel: fuse mask, left‑gate and right‑gate into the projected tensors
# ---------------------------------------------------------------------------
@triton.jit
def apply_mask_gate_kernel(
    left_ptr,                # pointer to left   [B,N,N,H]
    right_ptr,               # pointer to right  [B,N,N,H]
    left_gate_ptr,           # pointer to left_gate  [B,N,N,H]
    right_gate_ptr,          # pointer to right_gate [B,N,N,H]
    mask_ptr,                # pointer to mask   [B,N,N]
    B, N, H,                 # runtime ints
    stride_l_b, stride_l_i, stride_l_j, stride_l_h,
    stride_g_b, stride_g_i, stride_g_j, stride_g_h,
    stride_m_b, stride_m_i, stride_m_j,
    BLOCK_H: tl.constexpr,   # compile‑time block size (≥ hidden_dim)
):
    """
    For each (b,i,j) tuple this kernel loads a block of the hidden
    dimension (size = BLOCK_H) from the left/right projections and their
    corresponding gates, multiplies them together and also applies the
    binary mask (broadcast over the hidden axis).  The result is written
    back in‑place to the left/right tensors.
    """
    pid = tl.program_id(0)                     # one program per (b,i,j)
    total_ij = N * N
    b = pid // total_ij
    ij = pid % total_ij
    i = ij // N
    j = ij % N

    # offset vectors for the hidden dimension
    offs_h = tl.arange(0, BLOCK_H)

    # ---- load mask value (scalar) ----
    mask_off = mask_ptr + b * stride_m_b + i * stride_m_i + j * stride_m_j
    mask_val = tl.load(mask_off)               # fp16 scalar (0.0 or 1.0)

    # ---- compute base pointers for the four tensors ----
    left_base = left_ptr + b * stride_l_b + i * stride_l_i + j * stride_l_j
    right_base = right_ptr + b * stride_l_b + i * stride_l_i + j * stride_l_j
    left_gate_base = left_gate_ptr + b * stride_g_b + i * stride_g_i + j * stride_g_j
    right_gate_base = right_gate_ptr + b * stride_g_b + i * stride_g_i + j * stride_g_j

    # mask for the hidden dimension (handles H < BLOCK_H)
    mask_h = offs_h < H

    # ---- load tensors (masked loads to avoid OOB) ----
    left = tl.load(left_base + offs_h * stride_l_h,
                   mask=mask_h, other=0.0)
    right = tl.load(right_base + offs_h * stride_l_h,
                    mask=mask_h, other=0.0)

    left_gate = tl.load(left_gate_base + offs_h * stride_g_h,
                        mask=mask_h, other=0.0)
    right_gate = tl.load(right_gate_base + offs_h * stride_g_h,
                         mask=mask_h, other=0.0)

    # ---- fused operation ----
    left_out = left * left_gate * mask_val
    right_out = right * right_gate * mask_val

    # ---- store results back in‑place ----
    tl.store(left_base + offs_h * stride_l_h, left_out, mask=mask_h)
    tl.store(right_base + offs_h * stride_l_h, right_out, mask=mask_h)


# ---------------------------------------------------------------------------
# Helper: PyTorch LayerNorm (kept in PyTorch for simplicity)
# ---------------------------------------------------------------------------
def layer_norm_torch(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm over the last dimension using PyTorch (highly optimized)."""
    return torch.nn.functional.layer_norm(
        x, x.shape[-1:], weight=weight, bias=bias, eps=eps
    )


# ---------------------------------------------------------------------------
# Entry point required by the evaluation harness
# ---------------------------------------------------------------------------
def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]) -> torch.Tensor:
    """
    Triton‑accelerated “outgoing” TriMul implementation.

    Steps
    -------
    1. Layer‑norm over the channel dimension (dim) – PyTorch.
    2. Linear projections (left/right) – fp16 matmul via torch.nn.functional.linear.
    3. Compute three gating tensors (left_gate, right_gate, out_gate) – sigmoid(linear).
    4. Fuse mask‑application and gate‑multiplication for left/right using a Triton kernel.
       (If `nomask` is True the kernel is skipped and only the gates are applied.)
    5. Perform the N³ contraction with a batched GEMM:
           out[b,i,j,h] = Σₖ left[b,i,k,h] * right[b,j,k,h]
       realized as a batch of (B·H) matrix‑multiply (N×N) @ (N×N)ᵀ.
    6. Layer‑norm over the hidden dimension (hidden_dim) – PyTorch.
    7. Apply the output gate and map back to the original `dim` with a final linear.
    8. Return a float32 tensor of shape [B, N, N, dim].

    The only custom‑CUDA work is the mask‑gate fusion kernel; all large
    matrix multiplies profit from cuBLAS (Tensor‑Core) acceleration.
    """
    # --------------------------------------------------------------
    # unpack inputs
    # --------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dtype = input_tensor.dtype

    # --------------------------------------------------------------
    # configuration
    # --------------------------------------------------------------
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-5
    use_mask = not config.get("nomask", True)

    # --------------------------------------------------------------
    # 1) LayerNorm over the input channel dimension
    # --------------------------------------------------------------
    norm_w = weights["norm.weight"].to(device, dtype)
    norm_b = weights["norm.bias"].to(device, dtype)
    x = layer_norm_torch(input_tensor, norm_w, norm_b, eps=eps)   # [B,N,N,dim]

    # --------------------------------------------------------------
    # Cast to fp16 for the heavy part (benefits from Tensor Cores)
    # --------------------------------------------------------------
    x = x.to(torch.float16)

    # --------------------------------------------------------------
    # 2) Linear projections (no bias)
    # --------------------------------------------------------------
    left_proj_w = weights["left_proj.weight"].to(device, torch.float16)
    right_proj_w = weights["right_proj.weight"].to(device, torch.float16)

    left = torch.nn.functional.linear(x, left_proj_w)   # [B,N,N,hidden]
    right = torch.nn.functional.linear(x, right_proj_w)

    # --------------------------------------------------------------
    # 3) Gating tensors (sigmoid(linear(x)))
    # --------------------------------------------------------------
    left_gate_w = weights["left_gate.weight"].to(device, torch.float16)
    right_gate_w = weights["right_gate.weight"].to(device, torch.float16)
    out_gate_w = weights["out_gate.weight"].to(device, torch.float16)

    left_gate = torch.sigmoid(torch.nn.functional.linear(x, left_gate_w))
    right_gate = torch.sigmoid(torch.nn.functional.linear(x, right_gate_w))
    out_gate = torch.sigmoid(torch.nn.functional.linear(x, out_gate_w))

    # --------------------------------------------------------------
    # 4) Fuse mask + gates (Triton kernel) – in‑place on left/right
    # --------------------------------------------------------------
    if use_mask:
        # ensure mask is fp16 and broadcastable
        mask_fp = mask.to(torch.float16).contiguous()
        # make sure all tensors are contiguous (required for Triton pointer arithmetic)
        left = left.contiguous()
        right = right.contiguous()
        left_gate = left_gate.contiguous()
        right_gate = right_gate.contiguous()

        B, N, _, H = left.shape
        total_programs = B * N * N            # one program per (b,i,j)
        apply_mask_gate_kernel[total_programs](
            left.data_ptr(),
            right.data_ptr(),
            left_gate.data_ptr(),
            right_gate.data_ptr(),
            mask_fp.data_ptr(),
            B, N, H,
            left.stride(0), left.stride(1), left.stride(2), left.stride(3),
            left_gate.stride(0), left_gate.stride(1), left_gate.stride(2), left_gate.stride(3),
            mask_fp.stride(0), mask_fp.stride(1), mask_fp.stride(2),
            BLOCK_H=128,                     # hidden_dim ≤ 128 per the spec
        )
        # `left` and `right` now contain the masked‑gated tensors.
    else:
        # No mask: just element‑wise gate multiplication
        left = left * left_gate
        right = right * right_gate

    # --------------------------------------------------------------
    # 5) N³ contraction via batched GEMM (B·H independent matmuls)
    # --------------------------------------------------------------
    B, N, _, H = left.shape
    # bring hidden dimension to the batch axis
    left_perm = left.permute(0, 3, 1, 2).contiguous()   # (B, H, N, N)
    right_perm = right.permute(0, 3, 1, 2).contiguous()

    left_flat = left_perm.view(B * H, N, N)            # (B·H, N, N)
    right_flat = right_perm.view(B * H, N, N)

    # Batched matmul: left @ rightᵀ  →  (B·H, N, N)
    out_flat = torch.bmm(left_flat, right_flat.transpose(1, 2))

    # reshape back to (B,N,N,H)
    out = out_flat.view(B, H, N, N).permute(0, 2, 3, 1).contiguous()

    # --------------------------------------------------------------
    # 6) LayerNorm over hidden dimension
    # --------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(device, torch.float16)
    to_out_norm_b = weights["to_out_norm.bias"].to(device, torch.float16)

    out = layer_norm_torch(out, to_out_norm_w, to_out_norm_b, eps=eps)

    # --------------------------------------------------------------
    # 7) Apply the output gate and final linear projection back to `dim`
    # --------------------------------------------------------------
    out = out * out_gate

    to_out_w = weights["to_out.weight"].to(device, torch.float16)
    out = torch.nn.functional.linear(out, to_out_w)   # [B,N,N,dim]

    # --------------------------------------------------------------
    # 8) Cast back to float32 (the public API expects float32)
    # --------------------------------------------------------------
    return out.to(torch.float32)