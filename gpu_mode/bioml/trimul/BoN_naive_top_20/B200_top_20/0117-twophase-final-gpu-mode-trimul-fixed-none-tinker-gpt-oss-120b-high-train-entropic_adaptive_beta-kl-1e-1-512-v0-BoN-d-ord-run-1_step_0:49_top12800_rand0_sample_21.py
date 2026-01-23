"""
TriMul (outgoing) forward pass implemented with a fused Triton kernel for the
core N³ matrix‑multiplication. The surrounding operations (LayerNorm, Linear
projections, gating, masking) are performed with PyTorch for simplicity.
Only the heavy “einsum” (left @ rightᵀ summed over the third axis) is
accelerated using a batched GEMM kernel written in Triton.
"""

import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: batched matrix multiplication (a @ b) for many batches.
# Input a: [B, M, K]   (dtype=fp16)
# Input b: [B, K, N]   (dtype=fp16)
# Output c: [B, M, N]  (dtype=fp16)
# The batch dimension B is the flattened (seq_batch * hidden_dim) dimension.
# ----------------------------------------------------------------------
@triton.jit
def batch_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Triton GEMM kernel where each program_id(0) indexes a batch (B*H),
    program_id(1) indexes a tile of the output rows (M dimension) and
    program_id(2) a tile of the output columns (N dimension).
    """
    batch_id = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # -------------- offsets inside the tile --------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # -------------- pointers for the current batch --------------
    a_tile_ptr = a_ptr + batch_id * stride_a_batch + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
    b_tile_ptr = b_ptr + batch_id * stride_b_batch + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n

    # -------------- accumulator (FP32 for precision) --------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------- loop over K dimension --------------
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for _ in range(num_k_tiles):
        # Load a tile (with out‑of‑bounds protection)
        a = tl.load(a_tile_ptr,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_tile_ptr,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        # Multiply‑accumulate (TL automatically up‑casts to FP32)
        acc += tl.dot(a, b)

        # Advance K‑tiles
        a_tile_ptr += BLOCK_K * stride_a_k
        b_tile_ptr += BLOCK_K * stride_b_k

    # -------------- write the result --------------
    c_tile_ptr = c_ptr + batch_id * stride_c_batch + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    tl.store(c_tile_ptr,
             acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.

    Arguments
    ----------
    data : tuple
        (input_tensor, mask_tensor, weights_dict, config_dict)

    Returns
    -------
    torch.Tensor
        Output tensor of shape [bs, seq_len, seq_len, dim] (float32)
    """
    # ------------------------------------------------------------------
    # unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dtype_fp16 = torch.float16

    # model configuration
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-5
    nomask = config.get("nomask", True)

    # ------------------------------------------------------------------
    # fetch weight tensors and cast where appropriate
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]            # [dim]
    norm_bias   = weights["norm.bias"]              # [dim]

    left_proj_weight  = weights["left_proj.weight"].to(dtype_fp16)   # [hidden_dim, dim]
    right_proj_weight = weights["right_proj.weight"].to(dtype_fp16)  # [hidden_dim, dim]

    left_gate_weight  = weights["left_gate.weight"].to(dtype_fp16)   # [hidden_dim, dim]
    right_gate_weight = weights["right_gate.weight"].to(dtype_fp16)  # [hidden_dim, dim]
    out_gate_weight   = weights["out_gate.weight"].to(dtype_fp16)   # [hidden_dim, dim]

    to_out_norm_weight = weights["to_out_norm.weight"].to(dtype_fp16)  # [hidden_dim]
    to_out_norm_bias   = weights["to_out_norm.bias"].to(dtype_fp16)    # [hidden_dim]

    to_out_weight = weights["to_out.weight"].to(dtype_fp16)   # [dim, hidden_dim]

    # ------------------------------------------------------------------
    # LayerNorm on the input (float32 for numeric stability, cast to fp16)
    # ------------------------------------------------------------------
    x_norm = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
    ).to(dtype_fp16)   # [B, N, N, dim]  fp16

    # ------------------------------------------------------------------
    # Linear projections (fp16)
    # ------------------------------------------------------------------
    left = torch.nn.functional.linear(x_norm, left_proj_weight)   # [B,N,N,hidden_dim]
    right = torch.nn.functional.linear(x_norm, right_proj_weight) # [B,N,N,hidden_dim]

    # ------------------------------------------------------------------
    # Optional mask (broadcast over hidden_dim)
    # ------------------------------------------------------------------
    if not nomask and mask is not None:
        mask_f = mask.to(dtype_fp16).unsqueeze(-1)   # [B,N,N,1]
        left = left * mask_f
        right = right * mask_f

    # ------------------------------------------------------------------
    # Gating vectors (sigmoid)
    # ------------------------------------------------------------------
    left_gate = torch.sigmoid(torch.nn.functional.linear(x_norm, left_gate_weight))
    right_gate = torch.sigmoid(torch.nn.functional.linear(x_norm, right_gate_weight))
    out_gate = torch.sigmoid(torch.nn.functional.linear(x_norm, out_gate_weight))

    # Apply gates
    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # Prepare tensors for the batched GEMM kernel
    #   left:  [B, N, N, hidden_dim]   -> permute -> [B, hidden_dim, N, N]
    #   right: [B, N, N, hidden_dim]   -> permute & transpose K dimension -> [B, hidden_dim, N, N]
    # Both are then flattened across (B*hidden_dim) => shape [B*hidden_dim, N, N]
    # ------------------------------------------------------------------
    B, N, _, _ = left.shape
    left_perm = left.permute(0, 3, 1, 2).contiguous()    # [B, hidden_dim, N, N]
    right_perm = right.permute(0, 3, 2, 1).contiguous()  # transpose K <-> J

    # Flatten the batch+channel dimension
    batch_h = B * hidden_dim
    left_flat = left_perm.view(batch_h, N, N)
    right_flat = right_perm.view(batch_h, N, N)

    # Output placeholder (fp16)
    out_flat = torch.empty_like(left_flat, dtype=dtype_fp16)

    # ------------------------------------------------------------------
    # Triton kernel launch parameters
    # ------------------------------------------------------------------
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        batch_h,
        triton.cdiv(N, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    # Strides (in elements) – convert to Python ints
    stride_a_batch, stride_a_m, stride_a_k = left_flat.stride()
    stride_b_batch, stride_b_k, stride_b_n = right_flat.stride()
    stride_c_batch, stride_c_m, stride_c_n = out_flat.stride()

    batch_matmul_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        M=N,
        N=N,
        K=N,
        stride_a_batch=stride_a_batch,
        stride_a_m=stride_a_m,
        stride_a_k=stride_a_k,
        stride_b_batch=stride_b_batch,
        stride_b_k=stride_b_k,
        stride_b_n=stride_b_n,
        stride_c_batch=stride_c_batch,
        stride_c_m=stride_c_m,
        stride_c_n=stride_c_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )

    # ------------------------------------------------------------------
    # Reshape back to [B, N, N, hidden_dim]
    # ------------------------------------------------------------------
    out = out_flat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()  # [B, N, N, hidden_dim]

    # ------------------------------------------------------------------
    # Output LayerNorm + gating
    # ------------------------------------------------------------------
    out = torch.nn.functional.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=to_out_norm_weight,
        bias=to_out_norm_bias,
        eps=eps,
    )
    out = out * out_gate  # Element‑wise gating (fp16)

    # ------------------------------------------------------------------
    # Final linear projection back to original dimensionality
    # ------------------------------------------------------------------
    out = torch.nn.functional.linear(out, to_out_weight)   # [B, N, N, dim]   fp16

    # Cast back to float32 for the final output
    return out.to(torch.float32)