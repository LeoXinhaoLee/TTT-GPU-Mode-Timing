"""
TriMul forward (outgoing) – Triton‑accelerated implementation.

Algorithm
---------
1. Layer‑normalize the input tensor over the last dimension.
2. Linear projections (left/right) to the hidden dimension and compute three
   gating vectors (left, right, out).  All linear layers are bias‑free.
3. (Optional) mask the projected tensors.
4. Apply the left/right gates element‑wise.
5. Compute the core TriMul operation:
       out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]
   This is exactly a batched matrix multiplication:
       out_h = left_h @ right_hᵀ   for every hidden slice h (and batch).
   The heavy GEMM is performed in FP16 with FP32 accumulation inside a
   Triton kernel `batched_gemm_kernel`.
6. Layer‑normalize the intermediate result over the hidden dimension,
   apply the out‑gate, and a final linear layer back to the original `dim`.
7. Return the result in float32.

Only the matrix‑multiplication (step 5) is written in Triton; the surrounding
operations are handled in PyTorch (FP16 where possible) for simplicity.
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Triton kernel: batched GEMM (A @ Bᵀ) for many independent batches.
# ----------------------------------------------------------------------
@triton.jit
def batched_gemm_kernel(
    A, B, C,               # pointers to tensors
    M, N, K,               # matrix sizes (M×K)·(K×N) → M×N
    stride_Ab, stride_Am, stride_Ak,   # strides for A (batch, M, K)
    stride_Bb, stride_Bk, stride_Bn,   # strides for B (batch, K, N)
    stride_Cb, stride_Cm, stride_Cn,   # strides for C (batch, M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)          # batch / hidden slice index
    pid_m = tl.program_id(0)          # block row
    pid_n = tl.program_id(1)          # block column

    # Compute offsets within the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)
    offs_k = tl.arange(0, BLOCK_K)                    # (BLOCK_K,)

    # Pointers to the start of the tile in A and B
    a_ptrs = A + pid_b * stride_Ab + offs_m[:, None] * stride_Am + offs_k[None, :] * stride_Ak
    b_ptrs = B + pid_b * stride_Bb + offs_k[:, None] * stride_Bk + offs_n[None, :] * stride_Bn

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K sized chunks
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles (masked for out‑of‑bounds)
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0).to(tl.float32)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0).to(tl.float32)

        acc += tl.dot(a, b)          # (BLOCK_M, BLOCK_K)·(BLOCK_K, BLOCK_N) → (BLOCK_M, BLOCK_N)

        # Advance K pointers
        a_ptrs += BLOCK_K * stride_Ak
        b_ptrs += BLOCK_K * stride_Bk

    # Write back results (cast to FP16 automatically by Triton)
    c_ptrs = C + pid_b * stride_Cb + offs_m[:, None] * stride_Cm + offs_n[None, :] * stride_Cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


# ----------------------------------------------------------------------
# Entry point used by the evaluation harness
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul module using a Triton GEMM kernel.
    Arguments
    ---------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor: torch.Tensor [B, N, N, dim]
        - mask       : torch.Tensor [B, N, N] (may be ignored when nomask=True)
        - weights    : dict of pre‑loaded model parameters
        - config     : dict with keys "dim", "hidden_dim", "nomask" (bool)

    Returns
    -------
    torch.Tensor
        Output tensor [B, N, N, dim] (float32)
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    bs, seq_len, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)

    # ------------------------------------------------------------------
    # 1. Input LayerNorm
    # ------------------------------------------------------------------
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=weights["norm.weight"],
        bias=weights["norm.bias"],
    )   # [B, N, N, dim]  (float32)

    # Cast to FP16 for the heavy compute
    x_h = x.to(torch.float16)

    # ------------------------------------------------------------------
    # 2. Linear projections + gates (all without bias)
    # ------------------------------------------------------------------
    def linear_h(inp, w):
        return F.linear(inp, w)  # no bias

    left  = linear_h(x_h, weights["left_proj.weight"].to(torch.float16))
    right = linear_h(x_h, weights["right_proj.weight"].to(torch.float16))

    left_gate  = torch.sigmoid(linear_h(x_h, weights["left_gate.weight"].to(torch.float16)))
    right_gate = torch.sigmoid(linear_h(x_h, weights["right_gate.weight"].to(torch.float16)))
    out_gate   = torch.sigmoid(linear_h(x_h, weights["out_gate.weight"].to(torch.float16)))

    # ------------------------------------------------------------------
    # 3. Optional mask (broadcast over hidden_dim)
    # ------------------------------------------------------------------
    if not nomask:
        mask_h = mask.to(torch.float16).unsqueeze(-1)   # [B, N, N, 1]
        left = left * mask_h
        right = right * mask_h

    # ------------------------------------------------------------------
    # 4. Apply gating
    # ------------------------------------------------------------------
    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5. Core TriMul via batched GEMM (left @ rightᵀ)
    # ------------------------------------------------------------------
    # Rearrange tensors so that hidden_dim becomes part of the batch dimension:
    # left  : [B, N, N, H] -> [B*H, N, N]   (row=i, col=k)
    # right : [B, N, N, H] -> [B*H, N, N] then transpose the last two dims (k, j)
    left_reshape = left.permute(0, 3, 1, 2).contiguous().view(bs * hidden_dim, seq_len, seq_len)
    right_t      = right.permute(0, 3, 2, 1).contiguous().view(bs * hidden_dim, seq_len, seq_len)

    out_reshape = torch.empty_like(left_reshape)   # fp16 output placeholder

    # Block sizes – tuned for H100 (FP16 matmul)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        triton.cdiv(seq_len, BLOCK_M),          # rows
        triton.cdiv(seq_len, BLOCK_N),          # cols
        bs * hidden_dim                         # batch*hidden slices
    )

    batched_gemm_kernel[grid](
        left_reshape,
        right_t,
        out_reshape,
        seq_len,               # M
        seq_len,               # N
        seq_len,               # K
        left_reshape.stride(0), left_reshape.stride(1), left_reshape.stride(2),
        right_t.stride(0),     right_t.stride(1),     right_t.stride(2),
        out_reshape.stride(0), out_reshape.stride(1), out_reshape.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Restore original shape: [B, N, N, H]
    out = out_reshape.view(bs, hidden_dim, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()

    # ------------------------------------------------------------------
    # 6. Post‑process: LayerNorm + out‑gate + final projection
    # ------------------------------------------------------------------
    # Layer‑norm over the hidden dimension
    out = F.layer_norm(
        out.to(torch.float32),               # convert to FP32 for stability
        (hidden_dim,),
        weight=weights["to_out_norm.weight"],
        bias=weights["to_out_norm.bias"],
    )

    # Apply out‑gate (already in FP16 – cast to FP32)
    out = out * out_gate.to(torch.float32)

    # Final linear projection back to `dim`
    out = F.linear(out, weights["to_out.weight"].to(torch.float32))

    return out  # [B, N, N, dim] (float32)