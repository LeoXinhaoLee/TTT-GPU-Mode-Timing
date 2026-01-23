"""
TriMul "outgoing" kernel (AlphaFold3 style)

The forward pass consists of:
1. Layer‑norm on the input tensor.
2. Two linear projections (left/right) and three gating linear layers.
3. Optional element‑wise masking.
4. The heavy O(N³·C) “pairwise” multiplication:
       out[b,i,j,c] = Σₖ left[b,i,k,c] * right[b,j,k,c]
   This is implemented as a batched matrix‑multiply with the right operand
   transposed.  The batch dimension is flattened into B·C to obtain many
   independent NxN matrix‐multiplications.  A custom Triton kernel tiles
   the matrices (BLOCK_M×BLOCK_N) and accumulation tiles the K dimension
   (BLOCK_K) with fp32 accumulation for numerical stability.
5. Layer‑norm on the hidden dimension, gated by an “out‑gate”.
6. Final linear projection back to the original channel dimension.

Only the expensive O(N³) multiplication is off‑loaded to Triton;
all other ops use regular PyTorch which keeps the implementation simple
and compatible with any input dtype.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _trimul_kernel(
    left_ptr,                 # *[batch*C, N, N]   left matrix (i,k)
    right_ptr,                # *[batch*C, N, N]   right matrix (j,k)   (will be read as (k,j))
    out_ptr,                  # *[batch*C, N, N]   output matrix (i,j)
    N,                       # scalar: sequence length
    stride_l_batch, stride_l_i, stride_l_k,
    stride_r_batch, stride_r_j, stride_r_k,
    stride_o_batch, stride_o_i, stride_o_j,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute out = left @ rightᵀ  for many independent batches.
    All strides are expressed in *elements* (not bytes) and refer to
    contiguous tensors of shape [batch*C, N, N] with layout (batch, row, col).
    """
    pid_b = tl.program_id(2)             # flattened batch * hidden index
    pid_m = tl.program_id(0)             # block row
    pid_n = tl.program_id(1)             # block column

    # ------------------------------------------------------------------
    # Offsets for the output tile (i, j)
    # ------------------------------------------------------------------
    offs_i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_i = offs_i < N
    mask_j = offs_j < N

    # Accumulator in fp32 for accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Loop over the reduction dimension k
    # ------------------------------------------------------------------
    num_k_blocks = tl.cdiv(N, BLOCK_K)
    for k in range(0, num_k_blocks):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < N

        # left[i,k]  -> shape (BLOCK_M, BLOCK_K)
        left_ptrs = (left_ptr
                     + pid_b * stride_l_batch
                     + offs_i[:, None] * stride_l_i
                     + offs_k[None, :] * stride_l_k)
        left = tl.load(left_ptrs,
                       mask=mask_i[:, None] & mask_k[None, :],
                       other=0.0)

        # right[j,k] needs to be transposed to (k,j) for the dot
        # right is stored as (j,k) → we index it as (k,j)
        right_ptrs = (right_ptr
                      + pid_b * stride_r_batch
                      + offs_k[:, None] * stride_r_k
                      + offs_j[None, :] * stride_r_j)
        right = tl.load(right_ptrs,
                        mask=mask_k[:, None] & mask_j[None, :],
                        other=0.0)

        # Cast to fp32 before the dot
        left  = left.to(tl.float32)
        right = right.to(tl.float32)

        # Accumulate the product
        acc += tl.dot(left, right)

    # ------------------------------------------------------------------
    # Store the result
    # ------------------------------------------------------------------
    out_ptrs = (out_ptr
                + pid_b * stride_o_batch
                + offs_i[:, None] * stride_o_i
                + offs_j[None, :] * stride_o_j)
    tl.store(out_ptrs,
             acc,
             mask=mask_i[:, None] & mask_j[None, :])


def _launch_trimul(left: torch.Tensor,
                   right: torch.Tensor,
                   block_m: int = 64,
                   block_n: int = 64,
                   block_k: int = 64) -> torch.Tensor:
    """
    left/right: [B, C, N, N] contiguous (C == hidden_dim)
    Returns: out tensor of same shape.
    """
    B, C, N, _ = left.shape
    # flatten (B*C) into one batch dimension
    left_flat  = left.view(-1, N, N)
    right_flat = right.view(-1, N, N)
    out_flat   = torch.empty_like(left_flat)

    # strides expressed in *elements*
    s_l_batch, s_l_i, s_l_k = left_flat.stride()
    s_r_batch, s_r_j, s_r_k = right_flat.stride()
    s_o_batch, s_o_i, s_o_j = out_flat.stride()

    grid = (
        (N + block_m - 1) // block_m,   # blocks in M dimension
        (N + block_n - 1) // block_n,   # blocks in N dimension
        left_flat.shape[0]              # flattened batch*C dimension
    )

    _trimul_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        N,
        s_l_batch, s_l_i, s_l_k,
        s_r_batch, s_r_j, s_r_k,
        s_o_batch, s_o_i, s_o_j,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,          # reasonable for the H100
    )
    # reshape back to [B, C, N, N]
    return out_flat.view(B, C, N, N)


def custom_kernel(data):
    """
    Triton‑accelerated forward pass of the TriMul “outgoing” operator.
    Arguments:
        data = (input_tensor, mask, weights, config)
    Returns:
        Tensor of shape [bs, seq_len, seq_len, dim] (float32)
    """
    input_tensor, mask, weights, config = data

    dim         = config["dim"]
    hidden_dim  = config["hidden_dim"]
    eps         = 1e-5

    # ------------------------------------------------------------------
    # 1️⃣ Layer‑norm on the input (dim‑wise)
    # ------------------------------------------------------------------
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
        eps=eps,
    )   # [B, N, N, dim]

    # ------------------------------------------------------------------
    # 2️⃣ Linear projections (no bias)
    # ------------------------------------------------------------------
    left  = F.linear(x, weights["left_proj.weight"])   # [B, N, N, hidden]
    right = F.linear(x, weights["right_proj.weight"])  # [B, N, N, hidden]

    # ------------------------------------------------------------------
    # 3️⃣ Optional mask (broadcast over hidden dim)
    # ------------------------------------------------------------------
    if mask is not None:
        # ensure mask dtype matches the tensors
        mask_ = mask.to(x.dtype).unsqueeze(-1)               # [B, N, N, 1]
        left  = left * mask_
        right = right * mask_

    # ------------------------------------------------------------------
    # 4️⃣ Gating (sigmoid of linear layers)
    # ------------------------------------------------------------------
    left_gate  = torch.sigmoid(F.linear(x, weights["left_gate.weight"]))
    right_gate = torch.sigmoid(F.linear(x, weights["right_gate.weight"]))
    out_gate   = torch.sigmoid(F.linear(x, weights["out_gate.weight"]))

    left  = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5️⃣ Heavy O(N³·C) multiplication – Triton kernel
    #    Work in (B, C, N, N) layout for easy flattening.
    # ------------------------------------------------------------------
    # Rearrange so that the hidden dimension is the second axis.
    # This yields a tensor of shape [B, hidden, N, N] which we then view
    # as [B*hidden, N, N] to launch the kernel.
    left_perm  = left.permute(0, 3, 1, 2).contiguous()
    right_perm = right.permute(0, 3, 1, 2).contiguous()

    # Run the batched mat‑mul (left @ rightᵀ) in fp32
    out_perm = _launch_trimul(left_perm, right_perm,
                              block_m=64, block_n=64, block_k=64)

    # Restore original layout: [B, N, N, hidden]
    out = out_perm.permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------------
    # 6️⃣ Hidden‑dim layer‑norm + out‑gate
    # ------------------------------------------------------------------
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
        eps=eps,
    )
    out = out * out_gate

    # ------------------------------------------------------------------
    # 7️⃣ Final linear projection back to `dim`
    # ------------------------------------------------------------------
    out = F.linear(out, weights["to_out.weight"])

    return out