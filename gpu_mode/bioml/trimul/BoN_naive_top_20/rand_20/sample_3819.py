"""
TriMul (outgoing) forward pass.

The heavy part of the TriMul operator is the
    out[i, j, d] = Σ_k left[i, k, d] * right[j, k, d]
which is a batched matrix multiplication of shape
    (B * hidden_dim, N, N) @ (B * hidden_dim, N, N)^T .
We fuse the batch and hidden‑dim dimensions and implement the
matrix multiplication with a custom Triton kernel.  All other
operations (layer‑norms, linear layers and sigmoids) are performed
with PyTorch for simplicity.

The kernel computes C = A @ B^T where
    A : (B*H, M, K)   (M = N, K = N)
    B : (B*H, K, N)   (N = N)
The output C has shape (B*H, M, N) which is later reshaped back to
(B, N, N, H).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: batched GEMM (C = A @ Bᵀ)
# ----------------------------------------------------------------------
@triton.jit
def _tri_mul_kernel(
    a_ptr, b_ptr, c_ptr,               # pointers
    batch, M, N, K,                     # dimensions (batch = B*H)
    stride_ah, stride_ak, stride_a_batch,   # A strides (batch, M, K)
    stride_bk, stride_bn, stride_b_batch,   # B strides (batch, K, N)
    stride_ch, stride_cn, stride_c_batch,   # C strides (batch, M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program instance computes a (BLOCK_M x BLOCK_N) tile of C for a
    given batch index (pid_batch). The inner reduction over K is tiled.
    """
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Starting offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension tiles
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Pointers for the current tile of A and B
        a_ptrs = (
            a_ptr
            + pid_batch * stride_a_batch
            + offs_m[:, None] * stride_ah
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + pid_batch * stride_b_batch
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )

        # Load tiles with boundary masks
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate (dot does fp16 * fp16 -> fp32)
        acc += tl.dot(a, b)

    # Write out the result tile
    c_ptrs = (
        c_ptr
        + pid_batch * stride_c_batch
        + offs_m[:, None] * stride_ch
        + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


# ----------------------------------------------------------------------
# Main entry point called by the evaluation harness
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator using a custom Triton kernel.

    Args:
        data: tuple (input_tensor, mask, weights, config)
            - input_tensor: torch.Tensor of shape [B, N, N, C] (float32)
            - mask: torch.Tensor of shape [B, N, N] (float32) – may be all‑ones
            - weights: dict with tensors for every linear / layer‑norm weight/bias
            - config: dict with keys "dim", "hidden_dim", "nomask" (bool)

    Returns:
        torch.Tensor of shape [B, N, N, C] (float32)
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dtype = input_tensor.dtype

    bs, seq_len, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    # ------------------------------------------------------------------
    # LayerNorm over the channel dimension (dim)
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    x = F.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=1e-5,
    )  # [B, N, N, dim]

    # ------------------------------------------------------------------
    # Linear projections (no bias)
    # ------------------------------------------------------------------
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left = F.linear(x, left_proj_weight)   # [B, N, N, hidden_dim]
    right = F.linear(x, right_proj_weight)  # [B, N, N, hidden_dim]

    # ------------------------------------------------------------------
    # Optional mask (mask shape: [B, N, N])
    # ------------------------------------------------------------------
    if not nomask:
        mask = mask.to(dtype)  # ensure same dtype
        mask = mask.unsqueeze(-1)  # [B, N, N, 1]
        left = left * mask
        right = right * mask

    # ------------------------------------------------------------------
    # Gating (sigmoid) and apply
    # ------------------------------------------------------------------
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]

    left_gate = torch.sigmoid(F.linear(x, left_gate_weight))   # [B,N,N,hidden]
    right_gate = torch.sigmoid(F.linear(x, right_gate_weight))
    out_gate = torch.sigmoid(F.linear(x, out_gate_weight))

    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # Cast to FP16 for the Triton GEMM
    # ------------------------------------------------------------------
    left = left.to(torch.float16)
    right = right.to(torch.float16)

    # ------------------------------------------------------------------
    # Prepare tensors for the kernel: fuse batch and hidden dimensions
    # ------------------------------------------------------------------
    # left:  [B, N, N, H] -> [B, H, N, N] -> (B*H, N, N)
    left_perm = left.permute(0, 3, 1, 2).contiguous()
    right_perm = right.permute(0, 3, 1, 2).contiguous()
    # right needs to be transposed on the last two dimensions for Bᵀ
    right_t = right_perm.transpose(-2, -1).contiguous()

    batch_h = bs * hidden_dim
    M = N = seq_len
    K = seq_len

    left_flat = left_perm.view(batch_h, M, K)          # (B*H, M, K)
    right_flat = right_t.view(batch_h, K, N)           # (B*H, K, N)

    # Output allocation (FP16, will be cast back later)
    out_flat = torch.empty(batch_h, M, N, dtype=torch.float16, device=device)

    # ------------------------------------------------------------------
    # Extract strides for Triton (must be Python ints)
    # ------------------------------------------------------------------
    stride_a_batch = left_flat.stride(0)
    stride_ah = left_flat.stride(1)
    stride_ak = left_flat.stride(2)

    stride_b_batch = right_flat.stride(0)
    stride_bk = right_flat.stride(1)
    stride_bn = right_flat.stride(2)

    stride_c_batch = out_flat.stride(0)
    stride_ch = out_flat.stride(1)
    stride_cn = out_flat.stride(2)

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (
        batch_h,
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    _tri_mul_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        batch_h,
        M,
        N,
        K,
        stride_ah,
        stride_ak,
        stride_a_batch,
        stride_bk,
        stride_bn,
        stride_b_batch,
        stride_ch,
        stride_cn,
        stride_c_batch,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # ------------------------------------------------------------------
    # Reshape output back to [(B, N, N, hidden_dim)]
    # ------------------------------------------------------------------
    out = out_flat.view(bs, hidden_dim, seq_len, seq_len)
    out = out.permute(0, 2, 3, 1).contiguous()   # [B, N, N, hidden_dim]
    out = out.to(torch.float32)                  # back to fp32 for the remaining ops

    # ------------------------------------------------------------------
    # Final LayerNorm, gating and linear projection
    # ------------------------------------------------------------------
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    out = F.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=to_out_norm_weight,
        bias=to_out_norm_bias,
        eps=1e-5,
    )  # [B, N, N, hidden_dim]

    out = out * out_gate  # broadcasted multiplication

    # Linear projection back to original dimension
    out = F.linear(out, to_out_weight)  # [B, N, N, dim]

    return out