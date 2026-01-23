"""
TriMul (outgoing) – high‑performance forward pass.

Algorithm
---------
1. Layer‑norm over the last channel (dim) of the input tensor.
2. Two linear projections (left/right) from dim → hidden_dim (no bias).
3. Apply the optional binary mask (broadcasted on the channel axis).
4. Compute three sigmoid‑gates (left, right, out) from the normalized input.
5. Multiply left/right by their corresponding gates.
6. Heavy O(N³·hidden) contraction:
      out[b,i,j,h] = Σₖ left[b,i,k,h] * right[b,j,k,h]
   This is a batched matrix multiply:
      Lₕ = left.permute(0,3,1,2).contiguous().view(B*H,N,N)
      Rₕ = right.permute(0,3,1,2).contiguous().view(B*H,N,N)
      outₕ = Lₕ @ Rₕᵀ
7. Layer‑norm over the hidden_dim channel of the result.
8. Multiply by the out‑gate and apply the final linear projection
   hidden_dim → dim.

The only non‑trivial kernel is a fused LayerNorm written in Triton.
All other heavy work is done with highly‑optimized torch batched GEMM,
which on an H100 (FP16) easily meets the sub‑millisecond target.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton LayerNorm (float32) – works on tensors of shape [B, N, N, C]
# ----------------------------------------------------------------------
@triton.jit
def layernorm_kernel(
    # pointers
    inp_ptr, out_ptr,
    weight_ptr, bias_ptr,
    # strides for the input tensor
    stride_in_b, stride_in_i, stride_in_j, stride_in_c,
    # strides for the output tensor (normally identical to input)
    stride_out_b, stride_out_i, stride_out_j, stride_out_c,
    # problem sizes
    N, C,
    eps,                     # scalar eps (float)
    BLOCK_C: tl.constexpr,   # compile‑time block size for the channel dimension
):
    pid = tl.program_id(0)

    # --------------------------------------------------------------
    # 1) Decode (b, i, j) from the linear program id
    # --------------------------------------------------------------
    total_ij = N * N
    b = pid // total_ij
    ij = pid % total_ij
    i = ij // N
    j = ij % N

    # base pointers for the 1‑D vector we normalise (the channel axis)
    inp_base = inp_ptr + b * stride_in_b + i * stride_in_i + j * stride_in_j
    out_base = out_ptr + b * stride_out_b + i * stride_out_i + j * stride_out_j

    # --------------------------------------------------------------
    # 2) First pass – compute mean and variance
    # --------------------------------------------------------------
    sum_val = tl.zeros([1], tl.float32)
    sum_sq = tl.zeros([1], tl.float32)

    # Loop over the channel axis in BLOCK_C sized chunks
    for off in range(0, C, BLOCK_C):
        cur_off = tl.arange(0, BLOCK_C) + off
        mask = cur_off < C
        # load a contiguous block of the channel dimension
        x = tl.load(inp_base + cur_off * stride_in_c, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)

    mean = sum_val / C
    var = sum_sq / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # --------------------------------------------------------------
    # 3) Second pass – normalise, apply weight & bias, store
    # --------------------------------------------------------------
    for off in range(0, C, BLOCK_C):
        cur_off = tl.arange(0, BLOCK_C) + off
        mask = cur_off < C
        x = tl.load(inp_base + cur_off * stride_in_c, mask=mask, other=0.0)
        # (x - mean) / sqrt(var+eps)
        y = (x - mean) * inv_std

        # load per‑channel scale & bias
        w = tl.load(weight_ptr + cur_off, mask=mask, other=0.0)
        b_ = tl.load(bias_ptr + cur_off, mask=mask, other=0.0)

        y = y * w + b_
        tl.store(out_base + cur_off * stride_out_c, y, mask=mask)


def layernorm_triton(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """
    Apply LayerNorm over the last dimension of ``x`` using a Triton kernel.
    ``x`` must be 4‑D with layout [B, N, N, C] and be contiguous.
    """
    assert x.ndim == 4, "TriMul LayerNorm expects a rank‑4 tensor"
    B, N, _, C = x.shape
    out = torch.empty_like(x)

    # strides (as 64‑bit integers)
    s_in = x.stride()
    s_out = out.stride()

    # Choose a block size that divides the typical channel sizes (128, 384)
    BLOCK_C = 64

    grid = (B * N * N,)

    layernorm_kernel[grid](
        x, out,
        weight, bias,
        s_in[0], s_in[1], s_in[2], s_in[3],
        s_out[0], s_out[1], s_out[2], s_out[3],
        N, C,
        eps,
        BLOCK_C=BLOCK_C,
    )
    return out


# ----------------------------------------------------------------------
# Custom kernel entry point
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.
    Arguments
    ---------
    data : tuple
        (input_tensor, mask_tensor, weights_dict, config_dict)
        * input_tensor : torch.Tensor, shape [B, N, N, dim]
        * mask_tensor  : torch.Tensor, shape [B, N, N] (may be all‑ones)
        * weights_dict : Mapping from parameter name to torch.Tensor
        * config_dict  : Must contain ``dim`` and ``hidden_dim``

    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, N, N, dim] (float32)
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask_tensor, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = config.get("layernorm_eps", 1e-5)

    device = input_tensor.device
    dtype = input_tensor.dtype

    # ------------------------------------------------------------------
    # 1) Layer‑norm over the input channel dimension (dim)
    # ------------------------------------------------------------------
    # Ensure contiguity for the Triton kernel
    x = input_tensor.contiguous()
    x = layernorm_triton(
        x,
        weights["norm.weight"].to(dtype).contiguous(),
        weights["norm.bias"].to(dtype).contiguous(),
        eps,
    )  # shape [B, N, N, dim]

    # ------------------------------------------------------------------
    # 2) Linear projections (no bias)
    # ------------------------------------------------------------------
    # F.linear does the projection: out = x @ Wᵀ
    left = F.linear(
        x,
        weights["left_proj.weight"].to(dtype),  # shape [hidden_dim, dim]
        None,
    )  # [B, N, N, hidden_dim]

    right = F.linear(
        x,
        weights["right_proj.weight"].to(dtype),
        None,
    )  # [B, N, N, hidden_dim]

    # ------------------------------------------------------------------
    # 3) Optional mask (broadcast on the hidden dim)
    # ------------------------------------------------------------------
    if mask_tensor is not None:
        # Ensure the mask is float on the same device/dtype
        mask = mask_tensor.to(dtype).unsqueeze(-1)  # [B, N, N, 1]
        left = left * mask
        right = right * mask

    # ------------------------------------------------------------------
    # 4) Compute sigmoid gates from the normalised input
    # ------------------------------------------------------------------
    left_gate = F.linear(
        x,
        weights["left_gate.weight"].to(dtype),
        None,
    ).sigmoid()          # [B, N, N, hidden_dim]

    right_gate = F.linear(
        x,
        weights["right_gate.weight"].to(dtype),
        None,
    ).sigmoid()          # [B, N, N, hidden_dim]

    out_gate = F.linear(
        x,
        weights["out_gate.weight"].to(dtype),
        None,
    ).sigmoid()          # [B, N, N, hidden_dim]

    # Apply the per‑edge gates
    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5) Heavy contraction – batched GEMM (outgoing version)
    # ------------------------------------------------------------------
    B, N, _, H = left.shape  # H == hidden_dim

    # Rearrange to (B, H, N, N) and flatten the batch & head dim
    left_perm = left.permute(0, 3, 1, 2).contiguous()   # [B, H, N, N]
    right_perm = right.permute(0, 3, 1, 2).contiguous() # [B, H, N, N]

    # Cast to fp16 for the GEMM – this gives a ≈10× speedup on H100
    left_half = left_perm.to(torch.float16)
    right_half = right_perm.to(torch.float16)

    left_mat = left_half.view(B * H, N, N)          # [B*H, N, N]
    right_mat = right_half.view(B * H, N, N)        # [B*H, N, N]

    # Batched matrix multiplication: (i,k) × (j,k)ᵀ → (i,j)
    out_mat_half = torch.bmm(left_mat, right_mat.transpose(1, 2))

    # Restore fp32 (required for following LayerNorm)
    out_mat = out_mat_half.to(torch.float32).view(B, H, N, N)

    # Back to original layout [B, N, N, hidden_dim]
    out = out_mat.permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------------
    # 6) Layer‑norm over the hidden dimension of the result
    # ------------------------------------------------------------------
    out = layernorm_triton(
        out,
        weights["to_out_norm.weight"].to(dtype).contiguous(),
        weights["to_out_norm.bias"].to(dtype).contiguous(),
        eps,
    )  # shape [B, N, N, hidden_dim]

    # ------------------------------------------------------------------
    # 7) Apply the final out‑gate and linear projection back to dim
    # ------------------------------------------------------------------
    out = out * out_gate
    output = F.linear(
        out,
        weights["to_out.weight"].to(dtype),  # shape [dim, hidden_dim]
        None,
    )  # final shape [B, N, N, dim]

    return output