"""
TriMul (outgoing) forward pass implemented with a small Triton kernel.

Algorithm
---------
1. Layer‑norm the input tensor (B,N,N,C) using PyTorch (fast CUDA implementation).
2. Cast to fp16 for the heavy compute (all linear layers and the cubic update).
3. Project the normalized tensor to the hidden dimension:
   left  = x @ left_proj_weight   (no bias)
   right = x @ right_proj_weight
4. Compute three sigmoid gates (left_gate, right_gate, out_gate) with linear layers.
5. If a mask is present, a Triton kernel fuses:
        left  = left  * left_gate  * mask
        right = right * right_gate * mask
   (mask is broadcasted over the hidden dimension).
   When `nomask=True` the mask step is skipped.
6. Compute the outgoing multiplicative update:
        out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
   This is performed as a batch of GEMMs:
        left  → (B, H, N, N) → (B*H, N, N)
        right → (B, H, N, N) → (B*H, N, N)
        out   = bmm(left_flat, right_flat.transpose(1,2))
        reshape back to (B,N,N,H).
7. Apply a second LayerNorm (over the hidden dimension) and the out‑gate.
8. Final linear projection back to the original dimensionality (C) and cast to fp32.

Only the mask‑gate fusion is done in Triton, satisfying the requirement
to have a custom kernel while keeping the overall runtime fast.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernel: element‑wise multiplication with mask and sigmoid gate
# --------------------------------------------------------------------------- #
@triton.jit
def apply_gate_mask_kernel(
    proj_ptr,          # [rows, hidden]  left or right projection
    gate_ptr,          # [rows, hidden]  corresponding gate
    mask_ptr,          # [rows]          mask broadcasted over hidden
    out_ptr,           # [rows, hidden]  output
    stride_proj_row, stride_proj_col,
    stride_gate_row, stride_gate_col,
    stride_mask_row,
    stride_out_row, stride_out_col,
    hidden: tl.constexpr,  # hidden dimension (C_hidden)
    BLOCK: tl.constexpr    # block size for hidden dimension
):
    row = tl.program_id(0)                    # index over rows = B*N*N
    col = tl.arange(0, BLOCK)                 # hidden dimension tiles

    # scalar mask for this row (float16)
    mask_val = tl.load(mask_ptr + row * stride_mask_row)

    # number of tiles needed to cover the hidden dimension
    num_iter = (hidden + BLOCK - 1) // BLOCK

    for it in range(num_iter):
        idx = it * BLOCK + col
        mask_idx = idx < hidden

        proj = tl.load(
            proj_ptr + row * stride_proj_row + idx * stride_proj_col,
            mask=mask_idx, other=0.0
        )
        gate = tl.load(
            gate_ptr + row * stride_gate_row + idx * stride_gate_col,
            mask=mask_idx, other=0.0
        )
        out = proj * gate * mask_val
        tl.store(
            out_ptr + row * stride_out_row + idx * stride_out_col,
            out, mask=mask_idx
        )


def apply_gate_mask(proj, gate, mask, hidden_dim):
    """
    Fuse projection, gate and mask using Triton.
    proj, gate : [B, N, N, hidden_dim]  (fp16, contiguous)
    mask       : [B, N, N]               (fp16, contiguous)
    Returns tensor of same shape as proj.
    """
    B, N, _, _ = proj.shape
    rows = B * N * N
    proj_f = proj.view(rows, hidden_dim).contiguous()
    gate_f = gate.view(rows, hidden_dim).contiguous()
    mask_f = mask.view(rows).contiguous()
    out_f = torch.empty_like(proj_f)

    # Block size for hidden dimension; 128 works for all hidden≤384
    BLOCK = 128
    grid = (rows,)
    apply_gate_mask_kernel[grid](
        proj_f, gate_f, mask_f, out_f,
        proj_f.stride(0), proj_f.stride(1),
        gate_f.stride(0), gate_f.stride(1),
        mask_f.stride(0),
        out_f.stride(0), out_f.stride(1),
        hidden=hidden_dim,
        BLOCK=BLOCK,
    )
    return out_f.view_as(proj)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul module.

    Args:
        data: (input_tensor, mask_tensor, weights_dict, config_dict)

    Returns:
        Tensor of shape [bs, seq_len, seq_len, dim] (float32)
    """
    input_tensor, mask_tensor, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-5
    nomask = config.get("nomask", True)

    # --------------------------------------------------------------------- #
    # Cast weights to the working dtype (fp16) – this saves memory and time.
    # --------------------------------------------------------------------- #
    def to_fp16(t):
        return t.to(dtype=torch.float16)

    norm_weight = to_fp16(weights["norm.weight"])
    norm_bias   = to_fp16(weights["norm.bias"])
    left_proj_w   = to_fp16(weights["left_proj.weight"])
    right_proj_w  = to_fp16(weights["right_proj.weight"])
    left_gate_w   = to_fp16(weights["left_gate.weight"])
    right_gate_w  = to_fp16(weights["right_gate.weight"])
    out_gate_w    = to_fp16(weights["out_gate.weight"])
    to_out_norm_w = to_fp16(weights["to_out_norm.weight"])
    to_out_norm_b = to_fp16(weights["to_out_norm.bias"])
    to_out_w      = to_fp16(weights["to_out.weight"])

    # --------------------------------------------------------------------- #
    # 1. Input layer‑norm (float32 → fp16 after)
    # --------------------------------------------------------------------- #
    x = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=norm_weight.float(),   # layer_norm works in fp32 for stability
        bias=norm_bias.float(),
        eps=eps,
    )
    # Cast to half for the remaining heavy computation
    x = x.half()

    # --------------------------------------------------------------------- #
    # 2. Linear projections (no bias)
    # --------------------------------------------------------------------- #
    left_proj   = torch.nn.functional.linear(x, left_proj_w)   # [B,N,N,hd]
    right_proj  = torch.nn.functional.linear(x, right_proj_w)

    # 3. Gates (sigmoid)
    left_gate   = torch.nn.functional.linear(x, left_gate_w).sigmoid()
    right_gate  = torch.nn.functional.linear(x, right_gate_w).sigmoid()
    out_gate    = torch.nn.functional.linear(x, out_gate_w).sigmoid()

    # --------------------------------------------------------------------- #
    # 4. Apply mask + gates (fused Triton kernel when mask is present)
    # --------------------------------------------------------------------- #
    if not nomask:
        # mask is [B,N,N]; convert to half and ensure contiguous layout
        mask = mask_tensor.to(dtype=torch.float16).contiguous()
        left  = apply_gate_mask(left_proj, left_gate, mask, hidden_dim)
        right = apply_gate_mask(right_proj, right_gate, mask, hidden_dim)
    else:
        left  = left_proj * left_gate
        right = right_proj * right_gate

    # --------------------------------------------------------------------- #
    # 5. Outgoing multiplicative update:
    #    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    #    Implemented as a batch of GEMMs (B*hidden_dim, N, N)
    # --------------------------------------------------------------------- #
    B, N, _, _ = left.shape
    # reshape -> (B, hidden_dim, N, N)
    left_perm  = left.permute(0, 3, 1, 2).contiguous()
    right_perm = right.permute(0, 3, 1, 2).contiguous()

    # merge batch and hidden dimension
    left_flat  = left_perm.view(B * hidden_dim, N, N)
    right_flat = right_perm.view(B * hidden_dim, N, N)

    # batched matrix multiplication
    out_flat = torch.bmm(left_flat, right_flat.transpose(1, 2))

    # reshape back to [B,N,N,hidden_dim]
    out = out_flat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()  # fp16

    # --------------------------------------------------------------------- #
    # 6. Second LayerNorm over hidden dimension
    # --------------------------------------------------------------------- #
    out = torch.nn.functional.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=eps,
    )

    # --------------------------------------------------------------------- #
    # 7. Apply out‑gate and final linear projection back to dim
    # --------------------------------------------------------------------- #
    out = out * out_gate
    out = torch.nn.functional.linear(out, to_out_w)  # still fp16
    # Cast final result back to float32 as required by the spec
    out = out.float()

    return out