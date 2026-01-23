"""
TriMul (outgoing) implementation for AlphaFold3.

The operator projects the input tensor X ∈ ℝ^{B×N×N×C} to a hidden
dimension H, masks and gates the projections, and finally computes

    OUT_{b,i,j,:} = Σ_k LEFT_{b,i,k,:} * RIGHT_{b,j,k,:}

which can be seen as a batch of H independent matrix‑multiplications:
    LEFT_{b,:, :, h} @ RIGHT_{b,:, :, h}.T.

The cubic O(B·H·N³) reduction is performed by a custom Triton kernel
that tiles the i‑, j‑ and k‑dimensions (BLOCK_I, BLOCK_J, BLOCK_K)
and accumulates the dot‑product in FP32.  All other operations
(LayerNorm, Linear projections, gating, final linear) are done with
PyTorch for simplicity.

The kernel receives stride information so it works with any contiguous
layout of the [B,N,N,H] tensors.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def trimul_outgoing_kernel(
    left_ptr,            # *float32
    right_ptr,           # *float32
    out_ptr,             # *float32
    B,                   # int
    H,                   # int
    stride_l_b, stride_l_i, stride_l_k, stride_l_h,
    stride_r_b, stride_r_j, stride_r_k, stride_r_h,
    stride_o_b, stride_o_i, stride_o_j, stride_o_h,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N: tl.constexpr,    # seq length (compile‑time constant)
):
    """Compute OUT = LEFT @ RIGHTᵀ for every batch and hidden channel."""
    pid_i = tl.program_id(axis=0)          # i‑tile
    pid_j = tl.program_id(axis=1)          # j‑tile
    pid_bh = tl.program_id(axis=2)         # combined batch * hidden index

    # decode batch and hidden channel from pid_bh
    b = pid_bh // H
    h = pid_bh % H

    # tile start indices
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J

    # coordinate vectors within the tile
    i = i_start + tl.arange(0, BLOCK_I)
    j = j_start + tl.arange(0, BLOCK_J)

    # boundary masks
    mask_i = i < N
    mask_j = j < N

    # accumulator
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

    # number of k‑tiles (compile‑time)
    NUM_K = tl.cdiv(N, BLOCK_K)

    # -----------------------------------------------------------------
    # Loop over k‑dimension
    # -----------------------------------------------------------------
    for k_tile in range(NUM_K):
        k_start = k_tile * BLOCK_K
        k = k_start + tl.arange(0, BLOCK_K)
        mask_k = k < N

        # -------------------------------------------------------------
        # Load LEFT[i, k]   -> shape (BLOCK_I, BLOCK_K)
        # -------------------------------------------------------------
        left_ptrs = (
            left_ptr
            + b * stride_l_b
            + h * stride_l_h
            + i[:, None] * stride_l_i
            + k[None, :] * stride_l_k
        )
        left = tl.load(
            left_ptrs,
            mask=(mask_i[:, None] & mask_k[None, :]),
            other=0.0,
        )  # fp32

        # -------------------------------------------------------------
        # Load RIGHTᵀ[k, j] -> shape (BLOCK_K, BLOCK_J)
        #   (original RIGHT is indexed as RIGHT[b, j, k, h])
        # -------------------------------------------------------------
        right_ptrs_T = (
            right_ptr
            + b * stride_r_b
            + h * stride_r_h
            + k[:, None] * stride_r_k
            + j[None, :] * stride_r_j
        )
        right_T = tl.load(
            right_ptrs_T,
            mask=(mask_k[:, None] & mask_j[None, :]),
            other=0.0,
        )  # fp32

        # -------------------------------------------------------------
        # Accumulate the block‑wise matrix product
        # -------------------------------------------------------------
        acc += tl.dot(left, right_T)   # (BLOCK_I, BLOCK_J)

    # -----------------------------------------------------------------
    # Store the result block
    # -----------------------------------------------------------------
    out_ptrs = (
        out_ptr
        + b * stride_o_b
        + h * stride_o_h
        + i[:, None] * stride_o_i
        + j[None, :] * stride_o_j
    )
    tl.store(
        out_ptrs,
        acc,
        mask=(mask_i[:, None] & mask_j[None, :]),
    )


def custom_kernel(data):
    """
    Forward pass of the TriMul (outgoing) module.
    Arguments:
        data: tuple (input, mask, weights, config)
            input  : torch.Tensor of shape [B, N, N, C]  (C = dim)
            mask   : torch.Tensor of shape [B, N, N]   (or None)
            weights: dict of model parameters (see code)
            config : dict containing "dim", "hidden_dim", "nomask" (bool)
    Returns:
        torch.Tensor of shape [B, N, N, dim]
    """
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dtype = input_tensor.dtype

    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    eps = 1e-5

    # ------------------------------------------------------------
    # 1. Input LayerNorm
    # ------------------------------------------------------------
    x = F.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
        eps=eps,
    )  # [B, N, N, dim]

    # ------------------------------------------------------------
    # 2. Linear projections (no bias)
    # ------------------------------------------------------------
    left = F.linear(x, weights["left_proj.weight"])   # [B, N, N, hidden_dim]
    right = F.linear(x, weights["right_proj.weight"])

    # ------------------------------------------------------------
    # 3. Optional mask
    # ------------------------------------------------------------
    if not nomask:
        mask_unsq = mask.unsqueeze(-1).to(dtype)   # [B, N, N, 1]
        left = left * mask_unsq
        right = right * mask_unsq

    # ------------------------------------------------------------
    # 4. Gating (sigmoid)
    # ------------------------------------------------------------
    left_gate = F.linear(x, weights["left_gate.weight"]).sigmoid()
    right_gate = F.linear(x, weights["right_gate.weight"]).sigmoid()
    out_gate = F.linear(x, weights["out_gate.weight"]).sigmoid()

    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------
    # 5. Triton kernel: out = Σ_k left[i,k] * right[j,k]
    # ------------------------------------------------------------
    left = left.contiguous()
    right = right.contiguous()
    B, N, _, _ = left.shape     # N is the sequence length
    out = torch.empty((B, N, N, hidden_dim), dtype=dtype, device=device)

    # Strides (in element units, not bytes)
    stride_l_b, stride_l_i, stride_l_k, stride_l_h = left.stride()
    stride_r_b, stride_r_j, stride_r_k, stride_r_h = right.stride()
    stride_o_b, stride_o_i, stride_o_j, stride_o_h = out.stride()

    # Tile sizes – chosen to balance register pressure and occupancy.
    BLOCK_I = 64
    BLOCK_J = 64
    BLOCK_K = 32

    # Grid: (i‑tiles, j‑tiles, batch*hidden)
    grid = (
        triton.cdiv(N, BLOCK_I),
        triton.cdiv(N, BLOCK_J),
        B * hidden_dim,
    )

    trimul_outgoing_kernel[grid](
        left_ptr=left,
        right_ptr=right,
        out_ptr=out,
        B=B,
        H=hidden_dim,
        stride_l_b=stride_l_b,
        stride_l_i=stride_l_i,
        stride_l_k=stride_l_k,
        stride_l_h=stride_l_h,
        stride_r_b=stride_r_b,
        stride_r_j=stride_r_j,
        stride_r_k=stride_r_k,
        stride_r_h=stride_r_h,
        stride_o_b=stride_o_b,
        stride_o_i=stride_o_i,
        stride_o_j=stride_o_j,
        stride_o_h=stride_o_h,
        BLOCK_I=BLOCK_I,
        BLOCK_J=BLOCK_J,
        BLOCK_K=BLOCK_K,
        N=N,                     # compile‑time constant for the kernel
    )

    # ------------------------------------------------------------
    # 6. Hidden‑dim LayerNorm
    # ------------------------------------------------------------
    out_norm = F.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
        eps=eps,
    )

    # ------------------------------------------------------------
    # 7. Output gating and final linear projection
    # ------------------------------------------------------------
    out_gated = out_norm * out_gate
    out_final = F.linear(out_gated, weights["to_out.weight"])

    return out_final