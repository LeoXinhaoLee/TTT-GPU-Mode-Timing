"""
TriMul (outgoing) forward pass implemented with a fused Triton kernel.

Algorithm
---------
1. Layer‑normalize the input tensor `x` with the provided weight/bias.
2. Linear projections:
   * left  = x @ left_proj.weight.T
   * right = x @ right_proj.weight.T
3. (Optional) mask the projected tensors.
4. Compute element‑wise gates (sigmoid):
   left_gate  = σ(x @ left_gate.weight .T )
   right_gate = σ(x @ right_gate.weight.T)
   out_gate   = σ(x @ out_gate.weight   .T )
5. Apply the gates to the projected tensors.
6. Compute the pair‑wise multiplicative update:
   out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d] .
   This is a batched matrix multiplication over the `k` axis and
   is performed with torch.bmm (highly‑optimised cuBLAS).
7. Fuse the output‑side LayerNorm, the gating by `out_gate`,
   and the scaling/bias of `to_out_norm` in a Triton kernel.
   Each program processes one (b,i,j) position and a full hidden‑dim
   vector (H threads, H == hidden_dim).  The kernel:
      * loads the hidden vector,
      * computes mean/variance → normalises,
      * applies (to_out_norm.weight, to_out_norm.bias),
      * multiplies by `out_gate`,
      * stores the result back in‑place.
8. Final linear projection: out @ to_out.weight.T  → shape (B,N,N,dim).

Only the LayerNorm + gate fusion (step 7) is written in Triton; the
remaining heavy computation uses the best available PyTorch/cuBLAS
implementation.  All tensors stay on the GPU and the final result is
float‑32.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layernorm_gate_kernel(
    out_ptr,               # ptr to the tensor [B, N, N, H]
    gate_ptr,              # ptr to out_gate [B, N, N, H]
    ln_weight_ptr,         # ptr to to_out_norm.weight   [H]
    ln_bias_ptr,           # ptr to to_out_norm.bias     [H]
    stride_out_b, stride_out_i, stride_out_j, stride_out_h,   # strides for out
    stride_gate_b, stride_gate_i, stride_gate_j, stride_gate_h,  # strides for gate
    B, N, H,               # problem dimensions (runtime)
    EPS: tl.constexpr = 1e-5,
    BLOCK_H: tl.constexpr = 128,   # hidden dimension (compile‑time)
):
    """Fused LayerNorm + out_gate multiplication for a single (b,i,j) element."""
    pid = tl.program_id(0)
    total = B * N * N
    if pid >= total:
        return

    # decode linear pid -> (b,i,j)
    b = pid // (N * N)
    ij = pid % (N * N)
    i = ij // N
    j = ij % N

    # base pointers
    out_base = out_ptr + b * stride_out_b + i * stride_out_i + j * stride_out_j
    gate_base = gate_ptr + b * stride_gate_b + i * stride_gate_i + j * stride_gate_j

    # hidden‑dim offsets (one thread per hidden element)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # load hidden vectors
    v = tl.load(out_base + offs_h * stride_out_h, mask=mask_h, other=0.0).to(tl.float32)
    g = tl.load(gate_base + offs_h * stride_gate_h, mask=mask_h, other=0.0).to(tl.float32)

    # ---------- LayerNorm ----------
    sum_v = tl.sum(v)               # Σ_h v_h
    mean = sum_v * (1.0 / H)        # μ = (1/H) Σ v
    diff = v - mean
    var = tl.sum(diff * diff) * (1.0 / H)   # σ²
    inv_std = 1.0 / tl.sqrt(var + EPS)

    v_norm = diff * inv_std

    # scale & bias of to_out_norm
    ln_w = tl.load(ln_weight_ptr + offs_h, mask=mask_h, other=1.0)
    ln_b = tl.load(ln_bias_ptr   + offs_h, mask=mask_h, other=0.0)
    v_norm = v_norm * ln_w + ln_b

    # ---------- out_gate ----------
    v_out = v_norm * g

    # store the result back to out (in‑place)
    tl.store(out_base + offs_h * stride_out_h, v_out, mask=mask_h)


def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul operator.

    Parameters
    ----------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor: torch.Tensor of shape [B, N, N, dim]
        - mask        : torch.Tensor of shape [B, N, N]   (may be unused)
        - weights     : dict of model parameters
        - config      : dict with keys "dim", "hidden_dim", "nomask" (optional)

    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, N, N, dim]
    """
    # unpack arguments
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = 1e-5

    device = input_tensor.device
    B, N, _, _ = input_tensor.shape

    # ---------------------------------------------------------
    # 1. Input LayerNorm
    # ---------------------------------------------------------
    norm_w = weights["norm.weight"]
    norm_b = weights["norm.bias"]
    # mean & variance over the last dim
    mean = input_tensor.mean(dim=-1, keepdim=True)
    var = ((input_tensor - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (input_tensor - mean) * torch.rsqrt(var + eps)
    x = x * norm_w + norm_b                 # [B, N, N, dim]

    # ---------------------------------------------------------
    # 2. Linear projections (no bias)
    # ---------------------------------------------------------
    left = F.linear(x, weights["left_proj.weight"])
    right = F.linear(x, weights["right_proj.weight"])

    # ---------------------------------------------------------
    # 3. Optional mask
    # ---------------------------------------------------------
    if not config.get("nomask", False):
        # mask is [B, N, N]; broadcast to hidden dim
        mask_exp = mask.to(dtype=left.dtype).unsqueeze(-1)   # [B,N,N,1]
        left = left * mask_exp
        right = right * mask_exp

    # ---------------------------------------------------------
    # 4. Gates (sigmoid)
    # ---------------------------------------------------------
    left_gate = torch.sigmoid(F.linear(x, weights["left_gate.weight"]))
    right_gate = torch.sigmoid(F.linear(x, weights["right_gate.weight"]))
    out_gate = torch.sigmoid(F.linear(x, weights["out_gate.weight"]))

    # apply gates
    left = left * left_gate
    right = right * right_gate

    # ---------------------------------------------------------
    # 5. Pairwise multiplicative update (einsum)
    #    out[b,i,j,d] = Σ_k left[b,i,k,d] * right[b,j,k,d]
    # ---------------------------------------------------------
    # reshape to (B*H, N, N) and use batched matmul
    left_flat = left.permute(0, 3, 1, 2).reshape(B * hidden_dim, N, N)   # (B*H, I, K)
    right_flat = right.permute(0, 3, 2, 1).reshape(B * hidden_dim, N, N)  # (B*H, K, J)

    out = torch.bmm(left_flat, right_flat)                               # (B*H, I, J)
    out = out.view(B, hidden_dim, N, N).permute(0, 2, 3, 1)             # (B, N, N, H)

    # ---------------------------------------------------------
    # 6. Fuse to_out_norm + out_gate using Triton
    # ---------------------------------------------------------
    # LayerNorm weights for the output side
    to_out_norm_w = weights["to_out_norm.weight"]
    to_out_norm_b = weights["to_out_norm.bias"]

    # Launch Triton kernel: one program per (b,i,j) position
    total_positions = B * N * N
    grid = (total_positions,)

    _layernorm_gate_kernel[grid](
        out_ptr=out,
        gate_ptr=out_gate,
        ln_weight_ptr=to_out_norm_w,
        ln_bias_ptr=to_out_norm_b,
        stride_out_b=out.stride(0),
        stride_out_i=out.stride(1),
        stride_out_j=out.stride(2),
        stride_out_h=out.stride(3),
        stride_gate_b=out_gate.stride(0),
        stride_gate_i=out_gate.stride(1),
        stride_gate_j=out_gate.stride(2),
        stride_gate_h=out_gate.stride(3),
        B=B,
        N=N,
        H=hidden_dim,
        EPS=eps,
        BLOCK_H=hidden_dim,      # hidden_dim is a compile‑time constant (≤128)
    )

    # ---------------------------------------------------------
    # 7. Final linear projection (to_out)
    # ---------------------------------------------------------
    to_out_w = weights["to_out.weight"]
    output = F.linear(out, to_out_w)   # shape (B, N, N, dim)

    return output