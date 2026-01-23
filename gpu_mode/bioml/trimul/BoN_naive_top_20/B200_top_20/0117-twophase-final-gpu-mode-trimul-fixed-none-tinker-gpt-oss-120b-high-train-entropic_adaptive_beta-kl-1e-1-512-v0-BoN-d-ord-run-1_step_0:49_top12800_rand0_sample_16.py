"""
TrionMul (outgoing) forward pass – highly‑optimized Triton + PyTorch implementation.

Algorithm
---------
1️⃣  Layer‑norm on the input tensor `x` (shape [B,N,N,C]) is performed
    by a custom Triton kernel (mean/var reduction over the channel dim C).
    The kernel works on a flattened view (M = B·N·N rows × C columns) and
    writes the normalized tensor back in FP32.

2️⃣  The normalized tensor is cast to FP16 and projected with five linear layers:
    * left_proj, right_proj            → hidden_dim
    * left_gate, right_gate, out_gate  → hidden_dim (followed by sigmoid)

3️⃣  Optional binary mask (shape [B,N,N]) is applied (broadcasted on the
    hidden dimension) to *left* and *right*.

4️⃣  Gating: left *= left_gate , right *= right_gate.

5️⃣  Multiplicative update:
        out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]
    This is exactly a batched matrix multiplication:
        (B·H, N, N) = (B·H, N, N) @ (B·H, N, N)^T
    where H = hidden_dim.  The operation uses `torch.bmm` (cublas).

6️⃣  A second Layer‑norm (hidden_dim) is applied with a Torch kernel,
    followed by element‑wise multiplication with `out_gate`.

7️⃣  Final linear projection (hidden_dim → C) produces the output
    tensor of shape [B,N,N,C] (cast back to FP32).

Only the first Layer‑Norm is Triton‑implemented; the rest uses
high‑performance PyTorch primitives (FP16 matmuls, bmm, and layer‑norm).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: per‑row (flattened) layer‑norm over the channel dimension
# ----------------------------------------------------------------------
@triton.jit
def layernorm_kernel(
    inp_ptr,                # *float32  input  (M, C)
    out_ptr,                # *float32  output (M, C)
    weight_ptr,             # *float32  gamma  (C,)
    bias_ptr,               # *float32  beta   (C,)
    M,                      # int32  number of rows  (= B*N*N)
    C,                      # int32  channels (dim)
    stride_inm, stride_inc,  # int32 strides of input  (row, channel)
    stride_outm, stride_outc, # int32 strides of output (row, channel)
    eps,                    # float32 epsilon
    BLOCK_C: tl.constexpr   # compile‑time block size for channel dim
):
    pid = tl.program_id(0)                # one program per row
    if pid >= M:
        return

    # -----------------------------------------------------------------
    # 1) Compute mean and variance over C for the row `pid`
    # -----------------------------------------------------------------
    sum = tl.zeros([1], dtype=tl.float32)
    sum_sq = tl.zeros([1], dtype=tl.float32)

    offs_c = tl.arange(0, BLOCK_C)
    num_c_blocks = tl.cdiv(C, BLOCK_C)

    for blk in range(0, num_c_blocks):
        c = blk * BLOCK_C + offs_c
        mask = c < C
        # load a BLOCK_C‑wide slice of the row
        x = tl.load(inp_ptr + pid * stride_inm + c * stride_inc, mask=mask, other=0.0)
        sum += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum / C
    var = sum_sq / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # -----------------------------------------------------------------
    # 2) Write normalized values with affine transform
    # -----------------------------------------------------------------
    for blk in range(0, num_c_blocks):
        c = blk * BLOCK_C + offs_c
        mask = c < C
        x = tl.load(inp_ptr + pid * stride_inm + c * stride_inc, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        w = tl.load(weight_ptr + c, mask=mask, other=0.0)
        b = tl.load(bias_ptr   + c, mask=mask, other=0.0)
        y = y * w + b
        tl.store(out_ptr + pid * stride_outm + c * stride_outc, y, mask=mask)


def layernorm_triton(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """
    Apply LayerNorm over the last dimension using the Triton kernel above.
    Input `x` must be a contiguous tensor of shape [B, N, N, C] (C = dim).
    Returns a tensor of the same shape and dtype (float32).
    """
    B, N, N2, C = x.shape
    assert N == N2, "Input must be square in the two middle dimensions."
    M = B * N * N
    x_flat = x.reshape(M, C).contiguous()
    out_flat = torch.empty_like(x_flat)

    # Choose a channel block size; 128 works for all supported C (≤384)
    BLOCK_C = 128 if C >= 128 else 32

    # Launch one program per row
    grid = (M,)

    layernorm_kernel[grid](
        x_flat,
        out_flat,
        weight,
        bias,
        M,
        C,
        x_flat.stride(0), x_flat.stride(1),   # input strides (row, channel)
        out_flat.stride(0), out_flat.stride(1),# output strides
        eps,
        BLOCK_C=BLOCK_C,
    )
    return out_flat.view(B, N, N, C)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul module.

    Parameters
    ----------
    data : tuple
        (input_tensor, mask, weights, config)
        - input_tensor : torch.Tensor   shape [B, N, N, C]  (float32)
        - mask         : torch.Tensor   shape [B, N, N]    (bool/float) or None
        - weights      : dict of torch.Tensors with keys:
            "norm.weight", "norm.bias",
            "left_proj.weight", "right_proj.weight",
            "left_gate.weight", "right_gate.weight", "out_gate.weight",
            "to_out_norm.weight", "to_out_norm.bias",
            "to_out.weight"
        - config       : dict containing at least "dim" and "hidden_dim"

    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, N, N, C] (float32)
    """
    # ------------------------------------------------------------------
    # Unpack inputs
    # ------------------------------------------------------------------
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    eps = config.get("eps", 1e-5)

    # ------------------------------------------------------------------
    # 1) First LayerNorm (Triton)
    # ------------------------------------------------------------------
    x_norm = layernorm_triton(
        input_tensor,
        weights["norm.weight"],
        weights["norm.bias"],
        eps=eps,
    )  # still float32

    # ------------------------------------------------------------------
    # Cast to FP16 for the heavy linear / matmul work
    # ------------------------------------------------------------------
    x_norm_fp16 = x_norm.to(torch.float16)

    # ------------------------------------------------------------------
    # 2) Linear projections – all weight tensors are cast to FP16 once
    # ------------------------------------------------------------------
    left_proj_w = weights["left_proj.weight"].to(torch.float16)
    right_proj_w = weights["right_proj.weight"].to(torch.float16)
    left_gate_w = weights["left_gate.weight"].to(torch.float16)
    right_gate_w = weights["right_gate.weight"].to(torch.float16)
    out_gate_w = weights["out_gate.weight"].to(torch.float16)

    left = F.linear(x_norm_fp16, left_proj_w)      # [B,N,N,hidden]
    right = F.linear(x_norm_fp16, right_proj_w)

    # ------------------------------------------------------------------
    # 3) Optional mask (broadcast on hidden dimension)
    # ------------------------------------------------------------------
    if mask is None:
        # all‑ones mask
        mask_tensor = torch.ones_like(input_tensor[..., 0], dtype=x_norm_fp16.dtype,
                                     device=x_norm_fp16.device)
    else:
        mask_tensor = mask.to(dtype=x_norm_fp16.dtype, device=x_norm_fp16.device)
    mask_tensor = mask_tensor.unsqueeze(-1)  # [B,N,N,1]

    left = left * mask_tensor
    right = right * mask_tensor

    # ------------------------------------------------------------------
    # 4) Gating (sigmoid after linear)
    # ------------------------------------------------------------------
    left_gate = torch.sigmoid(F.linear(x_norm_fp16, left_gate_w))
    right_gate = torch.sigmoid(F.linear(x_norm_fp16, right_gate_w))
    out_gate = torch.sigmoid(F.linear(x_norm_fp16, out_gate_w))

    left = left * left_gate
    right = right * right_gate

    # ------------------------------------------------------------------
    # 5) Multiplicative update: batched matmul over the sequence dimension
    #    out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]
    # ------------------------------------------------------------------
    B, N, _, H = left.shape  # H == hidden_dim
    # reshape to (B*H, N, N) for batched matmul
    left_flat = left.permute(0, 3, 1, 2).reshape(B * H, N, N)          # (B*H, N, N)
    right_flat = right.permute(0, 3, 1, 2).reshape(B * H, N, N)       # (B*H, N, N)

    # right transposed on the last two dimensions
    out_flat = torch.bmm(left_flat, right_flat.transpose(1, 2))       # (B*H, N, N)

    # reshape back to [B, N, N, hidden]
    out = out_flat.view(B, H, N, N).permute(0, 2, 3, 1)               # (B,N,N,hidden)

    # ------------------------------------------------------------------
    # 6) Second LayerNorm over hidden dimension
    # ------------------------------------------------------------------
    to_out_norm_w = weights["to_out_norm.weight"].to(torch.float16)
    to_out_norm_b = weights["to_out_norm.bias"].to(torch.float16)

    out_norm = torch.nn.functional.layer_norm(
        out,
        (hidden_dim,),
        weight=to_out_norm_w,
        bias=to_out_norm_b,
        eps=eps,
    )
    out_norm = out_norm * out_gate  # element‑wise gating

    # ------------------------------------------------------------------
    # 7) Final linear projection back to original channel size
    # ------------------------------------------------------------------
    to_out_w = weights["to_out.weight"].to(torch.float16)
    output_fp16 = F.linear(out_norm, to_out_w)   # [B,N,N,dim] in FP16

    # Cast back to FP32 as required by the specification
    return output_fp16.to(torch.float32)