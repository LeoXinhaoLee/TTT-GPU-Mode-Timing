"""
TriMul (outgoing) forward pass implemented with a small Triton kernel.

Algorithm
---------
1. Layer‑norm the input tensor (last dimension = dim).
2. Project the normalized tensor to a hidden dimension with four linear layers:
   * left_proj / right_proj  – main values
   * left_gate / right_gate  – sigmoid gates
   * out_gate                – post‑norm gate
3. Fuse mask, left/right gates in a Triton kernel
   (left = left * left_gate * mask, right = right * right_gate * mask).
4. Compute the pairwise outer‑product sum:
        out[i,j,:] = Σ_k left[i,k,:] * right[j,k,:]
   This is a batched matrix multiply over the hidden dimension
   (B·hidden) × N × N.
5. Layer‑norm the result over the hidden dimension, multiply by out_gate,
   and linearly project back to the original dimension.
6. Return a tensor of shape [B, N, N, dim] (same dtype as the input).

The heavy element‑wise mask + gate fusion is executed in Triton; all
other linear algebra uses PyTorch/TorchScript which already maps to
high‑performance cuBLAS kernels on the H100.
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mask_gate_fusion_kernel(
    left_ptr,          # [B,N,N,H] flattened
    left_gate_ptr,     # same shape as left
    right_ptr,         # same shape as left
    right_gate_ptr,    # same shape as left
    mask_ptr,          # [B,N,N] flattened (broadcast over H)
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse mask and sigmoid gates into left/right tensors."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)          # flat index in left/right
    total = B * N * N * H                               # total number of elements
    mask_total = B * N * N

    # validity masks
    valid = offs < total
    mask_offs = offs // H                               # index into the mask (broadcast)
    mask_valid = mask_offs < mask_total

    # load tensors (masked loads return 0 for out‑of‑bounds elements)
    left = tl.load(left_ptr + offs, mask=valid, other=0.0)
    left_gate = tl.load(left_gate_ptr + offs, mask=valid, other=0.0)
    right = tl.load(right_ptr + offs, mask=valid, other=0.0)
    right_gate = tl.load(right_gate_ptr + offs, mask=valid, other=0.0)
    mask_val = tl.load(mask_ptr + mask_offs, mask=mask_valid, other=0.0)

    # apply mask + gates
    left_out = left * left_gate * mask_val
    right_out = right * right_gate * mask_val

    # store results back in‑place
    tl.store(left_ptr + offs, left_out, mask=valid)
    tl.store(right_ptr + offs, right_out, mask=valid)


def custom_kernel(data):
    """
    Triton‑accelerated forward pass for the outgoing TriMul module.

    Parameters
    ----------
    data : tuple
        (input_tensor, mask, weights, config) where
        - input_tensor : torch.Tensor  [B, N, N, dim]
        - mask         : torch.Tensor  [B, N, N]    (may be ignored if config["nomask"] is True)
        - weights      : dict of torch.Tensors containing all learned parameters
        - config       : dict with keys "dim", "hidden_dim", optional "nomask"

    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, N, N, dim] (same dtype/device as input_tensor)
    """
    # ------------------------------------------------------------------ #
    # unpack arguments
    # ------------------------------------------------------------------ #
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dtype = input_tensor.dtype

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    # ------------------------------------------------------------------ #
    # 1️⃣ Layer‑norm over the last dimension (dim)
    # ------------------------------------------------------------------ #
    x = F.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
        eps=1e-5,
    )  # [B, N, N, dim]

    # ------------------------------------------------------------------ #
    # 2️⃣ Linear projections and sigmoid gates (no bias)
    # ------------------------------------------------------------------ #
    left = F.linear(x, weights["left_proj.weight"])          # [B, N, N, H]
    right = F.linear(x, weights["right_proj.weight"])

    left_gate = torch.sigmoid(F.linear(x, weights["left_gate.weight"]))
    right_gate = torch.sigmoid(F.linear(x, weights["right_gate.weight"]))
    out_gate = torch.sigmoid(F.linear(x, weights["out_gate.weight"]))

    # ------------------------------------------------------------------ #
    # 3️⃣ Fuse mask and gates with a Triton kernel
    # ------------------------------------------------------------------ #
    B, N, _, _ = left.shape

    if nomask:
        # mask of all ones – no effect
        mask_tensor = torch.ones((B, N, N), dtype=dtype, device=device)
    else:
        # ensure mask is float (0/1) and same dtype as the tensors
        mask_tensor = mask.to(dtype)

    # Flatten everything to 1‑D for the kernel (contiguous layout)
    left_f = left.contiguous().view(-1)
    right_f = right.contiguous().view(-1)
    left_gate_f = left_gate.contiguous().view(-1)
    right_gate_f = right_gate.contiguous().view(-1)
    mask_f = mask_tensor.contiguous().view(-1)

    total_elements = B * N * N * hidden_dim
    BLOCK = 1024  # can be tuned; 1024 gives a good balance on H100
    grid = (math.ceil(total_elements / BLOCK),)

    # launch kernel (all tensors are 1‑D pointers with stride‑1 layout)
    _mask_gate_fusion_kernel[grid](
        left_f,
        left_gate_f,
        right_f,
        right_gate_f,
        mask_f,
        B,
        N,
        hidden_dim,
        BLOCK=BLOCK,
    )

    # reshape back to 4‑D tensors
    left = left_f.view(B, N, N, hidden_dim)
    right = right_f.view(B, N, N, hidden_dim)

    # ------------------------------------------------------------------ #
    # 4️⃣ Pairwise outer‑product sum via batched GEMM
    #     out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]
    # ------------------------------------------------------------------ #
    # rearrange to (B*H, N, N) for torch.bmm
    left_mat = left.permute(0, 3, 1, 2).contiguous().view(B * hidden_dim, N, N)
    right_mat = right.permute(0, 3, 1, 2).contiguous().view(B * hidden_dim, N, N)

    # batched matrix multiplication: (B*H, N, N) x (B*H, N, N)^T
    out_mat = torch.bmm(left_mat, right_mat.transpose(1, 2))

    # reshape back to [B, N, N, H]
    out = out_mat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------------ #
    # 5️⃣ Second Layer‑norm over hidden dimension, apply out_gate
    # ------------------------------------------------------------------ #
    out = F.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
        eps=1e-5,
    )
    out = out * out_gate  # broadcast over hidden dim

    # ------------------------------------------------------------------ #
    # 6️⃣ Final linear projection back to original dim
    # ------------------------------------------------------------------ #
    out = F.linear(out, weights["to_out.weight"])

    # Ensure output dtype matches the original input dtype
    if out.dtype != dtype:
        out = out.to(dtype)

    return out