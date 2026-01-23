"""
TriMul “outgoing” implementation (AlphaFold‑3 style).

The forward pass consists of:
1. Layer‑Norm over the channel dimension (dim).
2. Two linear projections (left/right) from dim → hidden_dim.
3. Optional binary mask applied to the projected tensors.
4. Gated versions of the projections (sigmoid‑gates).
5. Core bilinear update:
       out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
   This is a batched matrix‑multiply (A @ Bᵀ) for each hidden channel.
   The heavy N³ work is performed by a fused Triton kernel
   (block‑wise GEMM, FP16 compute, FP32 accumulation).
6. Layer‑Norm + gating on the hidden tensor.
7. Final linear projection hidden_dim → dim.

All heavy arithmetic is carried out in FP16 (torch.float16) to exploit
tensor‑cores on H100.  The Triton kernel receives the flattened
(batch * hidden_dim, N, N) tensors and computes C = A @ Bᵀ with
BLOCK_M/N = 128 and BLOCK_K = 32.  The resulting tensor is cast back
to FP32 before returning.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_outgoing_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,                          # matrix sizes (M = N = K = seq_len)
    stride_ab, stride_am, stride_ak,   # A: (batch, M, K) strides
    stride_bb, stride_bn, stride_bk,   # B: (batch, N, K) strides
    stride_cb, stride_cm, stride_cn,   # C: (batch, M, N) strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched A @ Bᵀ where A,B ∈ [batch, M, K] and C ∈ [batch, M, N].

    This kernel is launched with a 3‑D grid:
        program_id(0) → batch index
        program_id(1) → block row (M)
        program_id(2) → block column (N)
    """
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator in FP32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension in tiles
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_tile * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # -------- load A tile (M × K) --------
        a_ptrs = (
            a_ptr
            + pid_batch * stride_ab
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # -------- load B tile transposed (K × N) --------
        # we need Bᵀ → indices (k, n)
        b_ptrs = (
            b_ptr
            + pid_batch * stride_bb
            + offs_k[:, None] * stride_bk   # k * stride_bk
            + offs_n[None, :] * stride_bn   # n * stride_bn
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # FP16 → FP32 for accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # dot: (M×K) @ (K×N) → (M×N)
        acc += tl.dot(a, b)

    # -------- store C tile --------
    c_ptrs = (
        c_ptr
        + pid_batch * stride_cb
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    # Cast back to FP16 for downstream ops
    c = acc.to(tl.float16)
    tl.store(
        c_ptrs,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def custom_kernel(data):
    """
    Triton‑accelerated forward of the TriMul “outgoing” module.

    Args:
        data: tuple (input_tensor, mask_tensor, weights_dict, config_dict)
            input_tensor – torch.Tensor [B, N, N, dim] (float32)
            mask_tensor   – torch.Tensor [B, N, N] (bool/float) – may be ignored if config["nomask"]=True
            weights_dict – mapping of parameter names to torch.Tensor (already on the target device)
            config_dict  – contains "dim", "hidden_dim", "nomask" (bool)

    Returns:
        torch.Tensor of shape [B, N, N, dim] (float32)
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dtype = input_tensor.dtype  # expected float32

    # configuration
    dim = config["dim"]                     # input channel dimension
    hidden_dim = config["hidden_dim"]       # inner dimension (c_z)
    nomask = config.get("nomask", True)    # True → ignore mask

    # ------------------------------------------------------------------
    # 1️⃣ LayerNorm over last dim (float16 for speed)
    # ------------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    x = torch.nn.functional.layer_norm(
        input_tensor,
        (dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=1e-5,
    ).half()  # FP16 compute from here onward

    # ------------------------------------------------------------------
    # 2️⃣ Linear projections (no bias)
    # ------------------------------------------------------------------
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left = torch.nn.functional.linear(x, left_proj_weight.half())   # [B,N,N,hidden]
    right = torch.nn.functional.linear(x, right_proj_weight.half())

    # ------------------------------------------------------------------
    # 3️⃣ Optional binary mask
    # ------------------------------------------------------------------
    if not nomask:
        # mask is bool/float, broadcast to hidden dim and convert to FP16
        m = mask.to(x.dtype).unsqueeze(-1)   # [B,N,N,1]
        left = left * m
        right = right * m

    # ------------------------------------------------------------------
    # 4️⃣ Gating (sigmoid)
    # ------------------------------------------------------------------
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]

    left_gate = torch.sigmoid(
        torch.nn.functional.linear(x, left_gate_weight.half())
    )
    right_gate = torch.sigmoid(
        torch.nn.functional.linear(x, right_gate_weight.half())
    )
    out_gate = torch.sigmoid(
        torch.nn.functional.linear(x, out_gate_weight.half())
    )

    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5️⃣ Core bilinear update via Triton (batched matmul)
    #    Compute C = left @ rightᵀ  for every hidden channel.
    # ------------------------------------------------------------------
    B, N, _, _ = left.shape  # B = batch size, N = seq_len
    # 5a – permute to (B, hidden, N, N) and flatten batch*hidden for GEMM
    left_perm = left.permute(0, 3, 1, 2).contiguous()   # [B, hidden, N, N]
    right_perm = right.permute(0, 3, 1, 2).contiguous()

    batch_hidden = B * hidden_dim
    left_flat = left_perm.view(batch_hidden, N, N)   # [BH, M, K]
    right_flat = right_perm.view(batch_hidden, N, N) # [BH, N, K]

    # Output buffer
    out_flat = torch.empty_like(left_flat, dtype=torch.float16)

    # Kernel launch configuration
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (
        batch_hidden,
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    # Triton call
    batched_matmul_outgoing_kernel[grid](
        left_flat,
        right_flat,
        out_flat,
        N,                     # M
        N,                     # N
        N,                     # K
        left_flat.stride(0),   # batch stride for A
        left_flat.stride(1),   # row stride (M) for A
        left_flat.stride(2),   # col stride (K) for A
        right_flat.stride(0),  # batch stride for B
        right_flat.stride(1),  # row stride (N) for B (used as “n” after transpose)
        right_flat.stride(2),  # col stride (K) for B
        out_flat.stride(0),    # batch stride for C
        out_flat.stride(1),    # row stride (M) for C
        out_flat.stride(2),    # col stride (N) for C
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    # ------------------------------------------------------------------
    # 5b – reshape back to [B,N,N,hidden]
    # ------------------------------------------------------------------
    out = (
        out_flat.view(B, hidden_dim, N, N)
        .permute(0, 2, 3, 1)
        .contiguous()
    )  # [B, N, N, hidden] (fp16)

    # ------------------------------------------------------------------
    # 6️⃣ Post‑projection LayerNorm (hidden) + out‑gate
    # ------------------------------------------------------------------
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    out = torch.nn.functional.layer_norm(
        out,
        (hidden_dim,),
        weight=to_out_norm_weight.half(),
        bias=to_out_norm_bias.half(),
        eps=1e-5,
    )
    out = out * out_gate  # element‑wise gating (still fp16)

    # ------------------------------------------------------------------
    # 7️⃣ Final linear projection back to `dim`
    # ------------------------------------------------------------------
    to_out_weight = weights["to_out.weight"]          # shape (dim, hidden)
    out = torch.nn.functional.linear(out, to_out_weight.half())  # [B,N,N,dim] fp16

    # Cast result back to FP32 as required by the interface
    return out.float()