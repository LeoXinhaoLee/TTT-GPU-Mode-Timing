"""
TriMul (outgoing) forward pass with Triton fusion.

Algorithm
---------
1. Layer‑norm the input tensor (B,N,N,dim).
2. FP16 linear maps:
   * left_proj, right_proj  -> hidden_dim
   * left_gate, right_gate, out_gate (followed by sigmoid) -> hidden_dim
3. Apply mask (or a mask of ones) and the per‑position gates with a small
   Triton kernel that computes `out = proj * mask * gate` element‑wise.
4. Batched matrix multiplication for each hidden channel:
   left  : (B,N,N,H) → (B,H,N,N)  (i,k)
   right : (B,N,N,H) → (B,H,N,N)  (k,j)   (transpose k‑j)
   out   = left @ right  (B*H, N, N) → (B,N,N,H)
5. Layer‑norm on the hidden dimension, multiply by `out_gate`,
   and a final linear projection back to `dim`.
All heavy arithmetic is performed in fp16; the final result is cast to fp32.
A Triton kernel is used for the fused mask‑gate‑proj multiplication.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel: elementwise (proj * mask * gate)
# ----------------------------------------------------------------------
@triton.jit
def fused_mul_kernel(proj_ptr, mask_ptr, gate_ptr, out_ptr,
                     N_ELEMS,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)

    mask = offs < N_ELEMS
    proj = tl.load(proj_ptr + offs, mask=mask, other=tl.constexpr(0.0))
    m    = tl.load(mask_ptr + offs, mask=mask, other=tl.constexpr(1.0))
    gate = tl.load(gate_ptr + offs, mask=mask, other=tl.constexpr(0.0))
    out = proj * m * gate
    tl.store(out_ptr + offs, out, mask=mask)


def fused_mul(proj: torch.Tensor, mask: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    proj, mask, gate must have the same dtype, be on the same device
    and be contiguous. Returns proj * mask * gate computed by Triton.
    """
    assert proj.shape == mask.shape == gate.shape
    total = proj.numel()
    out = torch.empty_like(proj)

    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    fused_mul_kernel[grid](
        proj, mask, gate, out,
        total,
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------------
# Entry point called from the test harness
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul update.

    Args:
        data: tuple (input_tensor, mask_tensor, weights_dict, config_dict)
              - input_tensor: (B, N, N, dim)  torch.float32
              - mask_tensor : (B, N, N)      torch.float32/torch.bool (ignored if config["nomask"] is True)
              - weights_dict: keys exactly as used in the reference implementation
              - config_dict : must contain "hidden_dim" and optionally "nomask"

    Returns:
        Tensor of shape (B, N, N, dim) (float32)
    """
    # unpack
    input_tensor, mask_tensor, weights, config = data
    device = input_tensor.device
    B, N, _, dim = input_tensor.shape
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", False)

    # ------------------------------------------------------------------
    # 1) Input layer‑norm (float32)
    # ------------------------------------------------------------------
    norm_w = weights["norm.weight"]
    norm_b = weights["norm.bias"]
    x = F.layer_norm(
        input_tensor,
        (dim,),
        weight=norm_w,
        bias=norm_b,
        eps=1e-5,
    )  # (B,N,N,dim) float32

    # ------------------------------------------------------------------
    # 2) Linear projections + gate pre‑activations (cast to fp16)
    # ------------------------------------------------------------------
    # cast weights once
    left_proj_w  = weights["left_proj.weight"].to(torch.float16, non_blocking=True)
    right_proj_w = weights["right_proj.weight"].to(torch.float16, non_blocking=True)

    left_gate_w  = weights["left_gate.weight"].to(torch.float16, non_blocking=True)
    right_gate_w = weights["right_gate.weight"].to(torch.float16, non_blocking=True)
    out_gate_w   = weights["out_gate.weight"].to(torch.float16, non_blocking=True)

    to_out_norm_w = weights["to_out_norm.weight"].to(torch.float16, non_blocking=True)
    to_out_norm_b = weights["to_out_norm.bias"].to(torch.float16, non_blocking=True)
    to_out_w      = weights["to_out.weight"].to(torch.float16, non_blocking=True)

    # use half‑precision for all GEMM work
    x_h = x.to(torch.float16, non_blocking=True)

    # projections
    left_proj  = F.linear(x_h, left_proj_w)   # (B,N,N,hidden_dim)
    right_proj = F.linear(x_h, right_proj_w) # (B,N,N,hidden_dim)

    # gate pre‑activations
    left_gate_pre  = F.linear(x_h, left_gate_w)
    right_gate_pre = F.linear(x_h, right_gate_w)
    out_gate_pre   = F.linear(x_h, out_gate_w)

    left_gate  = torch.sigmoid(left_gate_pre)   # (B,N,N,hidden_dim)
    right_gate = torch.sigmoid(right_gate_pre) # (B,N,N,hidden_dim)
    out_gate   = torch.sigmoid(out_gate_pre)   # (B,N,N,hidden_dim)

    # ------------------------------------------------------------------
    # 3) Mask + gate fusion (Triton)
    # ------------------------------------------------------------------
    if nomask:
        # mask of ones – allocate directly to avoid extra copy
        mask_exp = torch.ones(
            (B, N, N, hidden_dim),
            dtype=torch.float16,
            device=device,
        )
    else:
        # mask_tensor may be bool or float, bring to half and expand
        mask_exp = mask_tensor.to(torch.float16, non_blocking=True).unsqueeze(-1)
        mask_exp = mask_exp.expand(-1, -1, -1, hidden_dim).contiguous()

    left  = fused_mul(left_proj, mask_exp, left_gate)   # (B,N,N,H) half
    right = fused_mul(right_proj, mask_exp, right_gate) # (B,N,N,H) half

    # ------------------------------------------------------------------
    # 4) Batched matrix multiplication across the shared sequence dim
    # ------------------------------------------------------------------
    # shape transformations:
    # left : (B,N,N,H) -> (B,H,N,N)  (i,k)
    # right: (B,N,N,H) -> (B,H,N,N)  (k,j)  (transpose k<->j)
    left_t  = left.permute(0, 3, 1, 2).contiguous()   # (B, H, i, k)
    right_t = right.permute(0, 3, 2, 1).contiguous()  # (B, H, k, j)

    # collapse batch and head dimensions for torch.bmm
    left_flat  = left_t.view(B * hidden_dim, N, N)
    right_flat = right_t.view(B * hidden_dim, N, N)

    out_flat = torch.bmm(left_flat, right_flat)       # (B*H, N, N) half
    out = out_flat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()  # (B,N,N,H)

    # ------------------------------------------------------------------
    # 5) Post‑norm, out‑gate and final projection
    # ------------------------------------------------------------------
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=1e-5,
    )
    out = out * out_gate
    out = F.linear(out, to_out_w)   # (B,N,N,dim) half

    # Convert back to float32 for the external API
    return out.float()