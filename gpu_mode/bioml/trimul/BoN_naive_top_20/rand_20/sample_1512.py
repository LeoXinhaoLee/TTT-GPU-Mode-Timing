"""
TriMul (outgoing) forward implementation with a custom Triton kernel.

The heavy part of the operation is the Einstein‑summation
    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
which is equivalent to a batched matrix multiplication
    out_d = left_d @ right_dᵀ  for every hidden channel d.
We fuse this N×N matrix multiplication for all (batch·hidden) “heads”
using a Triton kernel, while the surrounding linear / layer‑norm / gating
operations are carried out with vanilla PyTorch for simplicity.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------------------------------------------------------------
# Triton kernel:  out_head = left_head @ right_headᵀ
#   left  : [head, N, N]   (row i, column k)
#   right : [head, N, N]   (row j, column k) → we load its transpose on‑the‑fly
#   out   : [head, N, N]   (row i, column j)
# -------------------------------------------------------------------------
@triton.jit
def tri_mul_kernel(
    # Pointers
    left_ptr, right_ptr, out_ptr,
    # Strides (in elements, not bytes)
    stride_left_head, stride_left_i, stride_left_k,
    stride_right_head, stride_right_i, stride_right_k,
    stride_out_head, stride_out_i, stride_out_j,
    # Problem size
    N,
    # Compile‑time block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # -----------------------------------------------------------------
    # 1) program IDs – each block works on a (i,j) tile of one head.
    # -----------------------------------------------------------------
    pid_h = tl.program_id(2)          # head = batch * hidden_dim
    pid_m = tl.program_id(0)          # tile along output rows (i)
    pid_n = tl.program_id(1)          # tile along output cols (j)

    # -----------------------------------------------------------------
    # 2) Offsets for the current tile.
    # -----------------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < N
    mask_n = offs_n < N

    # Pointers to the beginning of the current head.
    left_head  = left_ptr  + pid_h * stride_left_head
    right_head = right_ptr + pid_h * stride_right_head
    out_head   = out_ptr   + pid_h * stride_out_head

    # Accumulator for the tile.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -----------------------------------------------------------------
    # 3) Loop over the reduction dimension k.
    # -----------------------------------------------------------------
    for k in range(0, tl.cdiv(N, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < N

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = left_head + (offs_m[:, None] * stride_left_i
                              + offs_k[None, :] * stride_left_k)
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0)

        # Load Bᵀ tile: shape (BLOCK_K, BLOCK_N)
        # Bᵀ[k, j] = right[j, k]
        b_ptrs = right_head + (offs_k[:, None] * stride_right_k
                               + offs_n[None, :] * stride_right_i)
        b = tl.load(b_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0)

        # GEMM on the tile.
        acc += tl.dot(a, b)

    # -----------------------------------------------------------------
    # 4) Write the result back.
    # -----------------------------------------------------------------
    c_ptrs = out_head + (offs_m[:, None] * stride_out_i
                         + offs_n[None, :] * stride_out_j)
    tl.store(c_ptrs,
             acc,
             mask=mask_m[:, None] & mask_n[None, :])


# -------------------------------------------------------------------------
# Entry point expected by the evaluation harness
# -------------------------------------------------------------------------
def custom_kernel(data):
    """
    Performs the forward pass of the outgoing TriMul module.

    Arguments
    ---------
    data : tuple
        (input_tensor, mask_tensor, weights_dict, config_dict)

    Returns
    -------
    torch.Tensor
        The processed tensor of shape [bs, seq_len, seq_len, dim].
    """
    # -----------------------------------------------------------------
    # Unpack inputs
    # -----------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dtype = input_tensor.dtype

    dim = config["dim"]          # input feature dimension
    hidden_dim = config["hidden_dim"]
    eps = 1e-5

    # -----------------------------------------------------------------
    # 0) Layer‑norm over the last dimension (dim)
    # -----------------------------------------------------------------
    x_norm = F.layer_norm(
        input_tensor,
        (dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
        eps=eps,
    )

    # -----------------------------------------------------------------
    # 1) Linear projections (no bias)
    # -----------------------------------------------------------------
    left_proj  = F.linear(x_norm, weights["left_proj.weight"])   # [B,N,N,hidden]
    right_proj = F.linear(x_norm, weights["right_proj.weight"])

    # -----------------------------------------------------------------
    # 2) Apply mask if supplied (mask shape [B,N,N])
    # -----------------------------------------------------------------
    if mask is not None:
        # broadcast mask over the hidden dimension
        mask_f = mask.unsqueeze(-1).type_as(left_proj)   # [B,N,N,1]
        left_proj  = left_proj  * mask_f
        right_proj = right_proj * mask_f

    # -----------------------------------------------------------------
    # 3) Gating vectors (sigmoid)
    # -----------------------------------------------------------------
    left_gate  = torch.sigmoid(F.linear(x_norm, weights["left_gate.weight"]))
    right_gate = torch.sigmoid(F.linear(x_norm, weights["right_gate.weight"]))
    out_gate   = torch.sigmoid(F.linear(x_norm, weights["out_gate.weight"]))

    left  = left_proj  * left_gate
    right = right_proj * right_gate

    # -----------------------------------------------------------------
    # 4) Prepare tensors for the Triton kernel
    #    We bring the hidden dimension to the leading axis so that each
    #    (batch * hidden) becomes an independent “head”.
    # -----------------------------------------------------------------
    B, N, _, _ = left.shape                           # B = batch size
    head_cnt = B * hidden_dim

    # [B, hidden, N, N] → (contiguous)
    left_perm  = left.permute(0, 3, 1, 2).contiguous()
    right_perm = right.permute(0, 3, 1, 2).contiguous()

    # flatten heads: shape [head_cnt, N, N]
    left_flat  = left_perm.view(head_cnt, N, N)
    right_flat = right_perm.view(head_cnt, N, N)

    # Output buffer
    out_flat = torch.empty_like(left_flat)

    # -----------------------------------------------------------------
    # 5) Strides (in elements) – required by the kernel.
    # -----------------------------------------------------------------
    # left
    stride_l_head = left_flat.stride(0)
    stride_l_i    = left_flat.stride(1)   # row (i)
    stride_l_k    = left_flat.stride(2)   # column (k)

    # right
    stride_r_head = right_flat.stride(0)
    stride_r_i    = right_flat.stride(1)  # row (j)
    stride_r_k    = right_flat.stride(2)  # column (k)

    # out
    stride_o_head = out_flat.stride(0)
    stride_o_i    = out_flat.stride(1)    # row (i)
    stride_o_j    = out_flat.stride(2)    # column (j)

    # -----------------------------------------------------------------
    # 6) Triton launch configuration
    # -----------------------------------------------------------------
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        ( (N + BLOCK_M - 1) // BLOCK_M,
          (N + BLOCK_N - 1) // BLOCK_N,
          head_cnt )
    )

    # -----------------------------------------------------------------
    # 7) Run the kernel
    # -----------------------------------------------------------------
    tri_mul_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        stride_l_head, stride_l_i, stride_l_k,
        stride_r_head, stride_r_i, stride_r_k,
        stride_o_head, stride_o_i, stride_o_j,
        N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # -----------------------------------------------------------------
    # 8) Reshape back to the original layout: [B, N, N, hidden]
    # -----------------------------------------------------------------
    out = out_flat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    # -----------------------------------------------------------------
    # 9) Output layer‑norm (hidden_dim) + gating + final linear
    # -----------------------------------------------------------------
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
        eps=eps,
    )

    out = out * out_gate
    out = F.linear(out, weights["to_out.weight"])
    return out