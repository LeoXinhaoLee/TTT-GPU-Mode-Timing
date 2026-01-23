import torch
import triton
import triton.language as tl


@triton.jit
def fused_mul_inplace_kernel(
    a_ptr,          # left or right tensor (output overwritten in‑place)
    g_ptr,          # gate tensor (sigmoid output)
    m_ptr,          # mask expanded to [B,N,N,hidden_dim]
    total_elems,    # total number of elements (= B*N*N*hidden_dim)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple Triton kernel that fuses element‑wise multiplication:
        a = a * g * m
    The result is written back to ``a`` (in‑place) to save memory.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elems

    a = tl.load(a_ptr + offs, mask=mask)          # dtype inferred from a_ptr (half)
    g = tl.load(g_ptr + offs, mask=mask)
    m = tl.load(m_ptr + offs, mask=mask)

    out = a * g * m
    tl.store(a_ptr + offs, out, mask=mask)


def custom_kernel(data):
    """
    Triton‑accelerated implementation of the “outgoing” TriMul operator
    (AlphaFold‑3 style).  The heavy N³ reduction is performed with
    batched matrix multiplication (cuBLAS) while the mask‑gate fusion is
    executed in a small Triton kernel.
    Input/outputs are float32, but the bulk of the computation runs in
    float16 for speed on H100.
    """
    # unpack arguments
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)

    # -----------------------------------------------------------------
    # 1. Layer‑norm over the last dimension (float32 -> keep accuracy)
    # -----------------------------------------------------------------
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    x = torch.nn.functional.layer_norm(
        input_tensor,
        normalized_shape=(dim,),
        weight=norm_weight,
        bias=norm_bias,
        eps=1e-5,
    )
    # switch to half for the expensive part
    x = x.to(torch.float16)

    # -----------------------------------------------------------------
    # 2. Linear projections (no bias) and gating networks
    # -----------------------------------------------------------------
    left_proj_w = weights["left_proj.weight"].to(torch.float16)
    right_proj_w = weights["right_proj.weight"].to(torch.float16)

    left = torch.nn.functional.linear(x, left_proj_w)          # [B,N,N,hidden_dim]
    right = torch.nn.functional.linear(x, right_proj_w)

    left_gate_w = weights["left_gate.weight"].to(torch.float16)
    right_gate_w = weights["right_gate.weight"].to(torch.float16)
    out_gate_w = weights["out_gate.weight"].to(torch.float16)

    left_gate = torch.nn.functional.linear(x, left_gate_w).sigmoid()
    right_gate = torch.nn.functional.linear(x, right_gate_w).sigmoid()
    out_gate = torch.nn.functional.linear(x, out_gate_w).sigmoid()

    # -----------------------------------------------------------------
    # 3. Fuse mask (if present) and gate into a single kernel
    # -----------------------------------------------------------------
    if not nomask:
        # mask: [B,N,N] -> expand to [B,N,N,hidden_dim]
        mask_f = mask.to(torch.float16).unsqueeze(-1)          # [B,N,N,1]
        mask_f = mask_f.expand(-1, -1, -1, hidden_dim).contiguous()

        total = left.numel()
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)

        # left = left * left_gate * mask
        fused_mul_inplace_kernel[grid](
            left, left_gate, mask_f, total, BLOCK_SIZE=BLOCK
        )
        # right = right * right_gate * mask
        fused_mul_inplace_kernel[grid](
            right, right_gate, mask_f, total, BLOCK_SIZE=BLOCK
        )
    else:
        # Simple element‑wise multiplication without mask
        left = left * left_gate
        right = right * right_gate

    # -----------------------------------------------------------------
    # 4. Core N³ reduction via batched matrix multiplication
    #    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    #    This is equivalent to: out_d = left_d @ right_dᵀ  for each d.
    # -----------------------------------------------------------------
    B, N, _, _ = left.shape
    # shape -> (B, hidden_dim, N, N) for cuBLAS
    left_m = left.permute(0, 3, 1, 2).contiguous()
    right_m = right.permute(0, 3, 1, 2).contiguous()
    # batched gemm
    out_m = torch.matmul(left_m, right_m.transpose(-2, -1))
    # back to original layout
    out = out_m.permute(0, 2, 3, 1).contiguous()   # [B,N,N,hidden_dim]

    # -----------------------------------------------------------------
    # 5. Output layer‑norm, gate and final linear projection
    # -----------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(torch.float16)
    to_out_norm_b = weights["to_out_norm.bias"].to(torch.float16)

    out = torch.nn.functional.layer_norm(
        out,
        normalized_shape=(hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=1e-5,
    )
    out = out * out_gate

    to_out_w = weights["to_out.weight"].to(torch.float16)   # (dim, hidden_dim)
    out = torch.nn.functional.linear(out, to_out_w)        # [B,N,N,dim]

    # Cast back to float32 as required by the interface
    return out.to(torch.float32)