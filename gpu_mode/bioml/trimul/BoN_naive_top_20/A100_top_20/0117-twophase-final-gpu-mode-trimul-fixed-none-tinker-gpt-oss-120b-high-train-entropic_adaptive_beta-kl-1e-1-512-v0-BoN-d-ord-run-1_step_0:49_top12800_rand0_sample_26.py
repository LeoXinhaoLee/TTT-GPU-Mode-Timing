# -*- coding: utf-8 -*-
"""
TriMul (outgoing) forward pass with Triton‑accelerated linear maps.
--------------------------------------------------------------------------------
The heavy work in AlphaFold3's TriMul consists of three 1‑D linear
projections (left, right, gate) followed by a pairwise multiplication
over the sequence dimension:

    out[i, j, d] = Σ_k left[i, k, d] * right[j, k, d]

The per‑position 1‑D linear layers (including the three gates) are
implemented with a custom GEMM written in Triton, giving us a fast
fp16‑only path that fuses the weight transpose and the matrix multiply.
The subsequent N×N batched matmul across heads is performed with
torch.bmm, which already maps to Tensor‑Core GEMMs on H100.

All LayerNorms and element‑wise sigmoids are kept in PyTorch for
correctness, and the final projection back to `dim` also uses the
standard PyTorch linear.  The function returns a float32 tensor identical
to the reference implementation.
"""

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton GEMM (A: M×K, B: K×N → C: M×N)
# ----------------------------------------------------------------------
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)                     # (BM×BK)·(BK×BN)

    c = acc.to(tl.float16)
    tl.store(
        C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        c,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _gemm_triton(A: torch.Tensor, B: torch.Tensor,
                 BLOCK_M=128, BLOCK_N=128, BLOCK_K=32) -> torch.Tensor:
    """A @ B where A is (M, K) and B is (K, N).  Returns (M, N)."""
    assert A.is_contiguous() and B.is_contiguous()
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) *
                        triton.cdiv(N, META["BLOCK_N"]),)

    _matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


def _linear_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Linear layer without bias:  y = x @ weight.T
    - x : (M, in_dim)   FP16
    - weight : (out_dim, in_dim)  FP16
    Returns (M, out_dim) FP16.
    """
    w_t = weight.t().contiguous()
    return _gemm_triton(x, w_t)


# ----------------------------------------------------------------------
# Custom kernel entry point
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward of the outgoing TriMul module.
    Parameters
    ----------
    data : tuple
        (input_tensor, mask_tensor, weights_dict, config_dict)

    Returns
    -------
    torch.Tensor
        Tensor of shape [bs, seq_len, seq_len, dim] (float32)
    """
    # --------------------------------------------------------------
    # unpack inputs
    # --------------------------------------------------------------
    input_tensor, mask_tensor, weights, config = data
    device = input_tensor.device

    bs, N, N2, dim = input_tensor.shape
    assert N == N2
    hidden_dim = config["hidden_dim"]
    eps = config.get("eps", 1e-5)
    nomask = config.get("nomask", False)

    # --------------------------------------------------------------
    # 1) LayerNorm on the input pair representation
    # --------------------------------------------------------------
    norm_w = weights["norm.weight"].to(input_tensor.dtype).to(device)
    norm_b = weights["norm.bias"].to(input_tensor.dtype).to(device)
    x = torch.nn.functional.layer_norm(
        input_tensor,
        (dim,),
        weight=norm_w,
        bias=norm_b,
        eps=eps,
    )
    # Use half‑precision for everything that follows
    x = x.to(torch.float16)

    # --------------------------------------------------------------
    # 2) Flatten to (M, dim)  where M = B * N * N
    # --------------------------------------------------------------
    M = bs * N * N
    x_flat = x.reshape(M, dim).contiguous()

    # --------------------------------------------------------------
    # 3) Load all Linear weights as fp16 and make them contiguous
    # --------------------------------------------------------------
    left_proj_w = weights["left_proj.weight"].to(torch.float16).to(device).contiguous()
    right_proj_w = weights["right_proj.weight"].to(torch.float16).to(device).contiguous()
    left_gate_w = weights["left_gate.weight"].to(torch.float16).to(device).contiguous()
    right_gate_w = weights["right_gate.weight"].to(torch.float16).to(device).contiguous()
    out_gate_w = weights["out_gate.weight"].to(torch.float16).to(device).contiguous()

    # --------------------------------------------------------------
    # 4) Linear maps via Triton (fp16)
    # --------------------------------------------------------------
    left_proj = _linear_triton(x_flat, left_proj_w)          # (M, hidden_dim)
    right_proj = _linear_triton(x_flat, right_proj_w)        # (M, hidden_dim)

    left_gate = torch.sigmoid(_linear_triton(x_flat, left_gate_w))
    right_gate = torch.sigmoid(_linear_triton(x_flat, right_gate_w))
    out_gate = torch.sigmoid(_linear_triton(x_flat, out_gate_w))

    # --------------------------------------------------------------
    # 5) Apply mask (if present) and gates
    # --------------------------------------------------------------
    if nomask:
        mask_flat = torch.ones((M, 1), dtype=torch.float16, device=device)
    else:
        # mask shape : (bs, N, N)  →  (M, 1)
        mask_flat = mask_tensor.reshape(M, 1).to(torch.float16)

    left = left_proj * left_gate * mask_flat      # (M, hidden_dim)
    right = right_proj * right_gate * mask_flat

    # Release temporaries that are no longer needed
    del left_proj, right_proj, left_gate, right_gate, x_flat

    # --------------------------------------------------------------
    # 6) Reshape to [B, N, N, hidden_dim]
    # --------------------------------------------------------------
    left = left.view(bs, N, N, hidden_dim)
    right = right.view(bs, N, N, hidden_dim)
    out_gate = out_gate.view(bs, N, N, hidden_dim)

    # --------------------------------------------------------------
    # 7) Pairwise multiplication across the "k" dimension:
    #    out[i,j,d] = Σ_k left[i,k,d] * right[j,k,d]
    #    Implemented as batched GEMM:  left @ rightᵀ per head.
    # --------------------------------------------------------------
    # → (B, hidden_dim, N, N) layout for a single GEMM per head
    left_perm = left.permute(0, 3, 1, 2).contiguous()   # (B, H, N, N)
    right_perm = right.permute(0, 3, 1, 2).contiguous()

    BH = bs * hidden_dim
    left_mat = left_perm.reshape(BH, N, N)              # (B*H, N, N)
    right_mat = right_perm.reshape(BH, N, N)            # (B*H, N, N)

    # C = A @ Bᵀ  →  (B*H, N, N)
    out_mat = torch.bmm(left_mat, right_mat.transpose(1, 2))

    # back to [B, N, N, hidden_dim]
    out = out_mat.reshape(bs, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()

    # --------------------------------------------------------------
    # 8) LayerNorm over the hidden dimension
    # --------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(torch.float16).to(device)
    to_out_norm_b = weights["to_out_norm.bias"].to(torch.float16).to(device)

    out = torch.nn.functional.layer_norm(
        out,
        (hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=eps,
    )

    # --------------------------------------------------------------
    # 9) Apply the output gate
    # --------------------------------------------------------------
    out = out * out_gate   # (B,N,N,hidden_dim)

    # --------------------------------------------------------------
    # 10) Final linear projection back to `dim`
    # --------------------------------------------------------------
    to_out_w = weights["to_out.weight"].to(torch.float16).to(device)
    out = torch.nn.functional.linear(out, to_out_w)

    # Return in FP32 for consistency with the reference module
    return out.to(torch.float32)