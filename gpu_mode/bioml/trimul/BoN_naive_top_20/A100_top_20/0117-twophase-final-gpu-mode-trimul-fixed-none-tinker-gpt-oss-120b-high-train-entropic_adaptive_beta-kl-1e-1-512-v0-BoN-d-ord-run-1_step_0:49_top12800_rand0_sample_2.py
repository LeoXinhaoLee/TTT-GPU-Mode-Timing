"""
TriMul (outgoing) forward pass implemented with a small Triton kernel.

The kernel fuses three element‑wise operations that appear after the linear
projections:
    left = (x @ W_left)   * left_gate   * mask
    right = (x @ W_right) * right_gate  * mask
Both are computed in FP16 for speed.  The heavy pairwise reduction
∑_k left[...,i,k,d] * right[...,j,k,d] is performed with a
batched GEMM (torch.bmm) on the (B·hidden_dim) independent
(N×N) matrix multiplications.  The rest of the module (LayerNorms,
gates, final projection) is done with torch ops on FP16 tensors and
the final result is returned in FP32.

This implementation respects the required signature:
    custom_kernel(data) -> Tensor
where `data` is a tuple (input, mask, weights, config).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that multiplies a projection with its gate and (optionally)
# a binary mask.  It works on a flattened view of the tensor
#   X : [B, N, N, H]   (float16)
#   G : [B, N, N, H]   (float16)
#   M : [B, N, N]      (float16)  (optional)
# and writes the result to Y.
# ----------------------------------------------------------------------
@triton.jit
def _gate_mask_kernel(
    X_ptr, G_ptr, Y_ptr,   # pointers to X, gate, output
    M_ptr,                 # pointer to mask (ignored when HAS_MASK=False)
    total_elems,           # total number of elements in X/G/Y (B*N*N*H)
    hidden_dim,            # size of channel dimension H (runtime constant)
    HAS_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)

    # guard out‑of‑bounds loads
    mask = offs < total_elems

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0)

    if HAS_MASK:
        # each group of `hidden_dim` elements shares the same mask entry
        mask_idx = offs // hidden_dim
        m = tl.load(M_ptr + mask_idx, mask=mask, other=1.0)
        y = x * g * m
    else:
        y = x * g

    tl.store(Y_ptr + offs, y, mask=mask)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def custom_kernel(data):
    """
    Forward pass of the outgoing TriMul module.

    Arguments
    ----------
    data : tuple
        (input_tensor, mask_tensor, weights_dict, config_dict)

    Returns
    -------
    torch.Tensor
        Tensor of shape [bs, seq_len, seq_len, dim] (float32)
    """
    input_tensor, mask_tensor, weights, config = data

    # ------------------------------------------------------------------
    # unpack configuration
    # ------------------------------------------------------------------
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    nomask = config.get("nomask", True)   # True => mask should be ignored

    # ------------------------------------------------------------------
    # unpack weights (all are provided as float32 tensors)
    # ------------------------------------------------------------------
    norm_w = weights["norm.weight"]
    norm_b = weights["norm.bias"]

    left_proj_w  = weights["left_proj.weight"]      # (hidden_dim, dim)
    right_proj_w = weights["right_proj.weight"]     # (hidden_dim, dim)

    left_gate_w  = weights["left_gate.weight"]      # (hidden_dim, dim)
    right_gate_w = weights["right_gate.weight"]     # (hidden_dim, dim)
    out_gate_w   = weights["out_gate.weight"]       # (hidden_dim, dim)

    to_out_norm_w = weights["to_out_norm.weight"]   # (hidden_dim,)
    to_out_norm_b = weights["to_out_norm.bias"]     # (hidden_dim,)

    to_out_w = weights["to_out.weight"]            # (dim, hidden_dim)

    # ------------------------------------------------------------------
    # ---- 1. LayerNorm over the last dimension (dim) -----------------
    # ------------------------------------------------------------------
    # keep the data in FP32 for the norm, then cast to FP16 for the rest
    x_norm = F.layer_norm(input_tensor,
                          normalized_shape=(dim,),
                          weight=norm_w,
                          bias=norm_b)          # [B, N, N, dim]  FP32
    x = x_norm.to(torch.float16)                    # FP16 for speed

    # ------------------------------------------------------------------
    # ---- 2. Linear projections (no bias) ---------------------------
    # ------------------------------------------------------------------
    # cast weights to FP16 once
    left_proj_w_h  = left_proj_w.to(torch.float16)
    right_proj_w_h = right_proj_w.to(torch.float16)

    left = F.linear(x, left_proj_w_h)   # [B, N, N, hidden_dim]  FP16
    right = F.linear(x, right_proj_w_h)

    # ------------------------------------------------------------------
    # ---- 3. Gates (sigmoid of linear) -------------------------------
    # ------------------------------------------------------------------
    left_gate_w_h  = left_gate_w.to(torch.float16)
    right_gate_w_h = right_gate_w.to(torch.float16)
    out_gate_w_h   = out_gate_w.to(torch.float16)

    left_gate  = torch.sigmoid(F.linear(x, left_gate_w_h))   # [B,N,N,hidden_dim]
    right_gate = torch.sigmoid(F.linear(x, right_gate_w_h))
    out_gate   = torch.sigmoid(F.linear(x, out_gate_w_h))

    # ------------------------------------------------------------------
    # ---- 4. Fuse mask (if any) and gate multiplication using Triton-
    # ------------------------------------------------------------------
    B, N, _, _ = left.shape
    total_elems = left.numel()                     # B * N * N * hidden_dim

    # allocate output tensors for gated & masked projections
    left_gated  = torch.empty_like(left)
    right_gated = torch.empty_like(right)

    # flatten tensors for the kernel
    left_ptr   = left.contiguous().view(-1)
    left_gate_ptr = left_gate.contiguous().view(-1)
    left_out_ptr = left_gated.view(-1)

    right_ptr   = right.contiguous().view(-1)
    right_gate_ptr = right_gate.contiguous().view(-1)
    right_out_ptr = right_gated.view(-1)

    # ------------------------------------------------------------------
    # Prepare mask pointer (FP16) – it can be empty when not used
    # ------------------------------------------------------------------
    if not nomask and mask_tensor is not None:
        # mask: [B, N, N] -> FP16 and flatten
        mask_fp16 = mask_tensor.to(torch.float16).contiguous()
        mask_flat = mask_fp16.view(-1)      # length B * N * N
        mask_ptr = mask_flat
        has_mask = True
    else:
        # dummy tensor (never read because HAS_MASK=False)
        mask_ptr = torch.empty(0, dtype=torch.float16, device=left.device)
        has_mask = False

    # ------------------------------------------------------------------
    # Kernel launch configuration
    # ------------------------------------------------------------------
    BLOCK_SIZE = 8192   # ~64KB per block (tuned for H100)

    grid = lambda meta: (triton.cdiv(total_elems, meta['BLOCK_SIZE']),)

    # left branch
    _gate_mask_kernel[grid](
        left_ptr,
        left_gate_ptr,
        left_out_ptr,
        mask_ptr,
        total_elems,
        hidden_dim,
        HAS_MASK=has_mask,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # right branch
    _gate_mask_kernel[grid](
        right_ptr,
        right_gate_ptr,
        right_out_ptr,
        mask_ptr,
        total_elems,
        hidden_dim,
        HAS_MASK=has_mask,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # ------------------------------------------------------------------
    # ---- 5. Pairwise reduction (matrix multiplication) -------------
    # ------------------------------------------------------------------
    # reshape to (B*hidden_dim, N, N) for batched GEMM
    left_mat  = left_gated.permute(0, 3, 1, 2).reshape(B * hidden_dim, N, N)   # (B*H, N, N)
    right_mat = right_gated.permute(0, 3, 1, 2).reshape(B * hidden_dim, N, N)  # (B*H, N, N)

    # out_mat[b*h, i, j] = sum_k left[i,k] * right[j,k]
    out_mat = torch.bmm(left_mat, right_mat.transpose(1, 2))   # (B*H, N, N)

    # reshape back to [B, N, N, hidden_dim]
    out = out_mat.view(B, hidden_dim, N, N).permute(0, 2, 3, 1).contiguous()  # (B,N,N,hidden_dim)

    # ------------------------------------------------------------------
    # ---- 6. Output layer norm and final gate ------------------------
    # ------------------------------------------------------------------
    to_out_norm_w_h = to_out_norm_w.to(torch.float16)
    to_out_norm_b_h = to_out_norm_b.to(torch.float16)

    out = F.layer_norm(out,
                       normalized_shape=(hidden_dim,),
                       weight=to_out_norm_w_h,
                       bias=to_out_norm_b_h)          # FP16

    out = out * out_gate                              # apply out_gate (FP16)

    # ------------------------------------------------------------------
    # ---- 7. Final linear projection back to dim --------------------
    # ------------------------------------------------------------------
    to_out_w_h = to_out_w.to(torch.float16)           # (dim, hidden_dim)
    out = F.linear(out, to_out_w_h)                   # [B,N,N,dim]  FP16

    # ------------------------------------------------------------------
    # Cast back to FP32 for the final output (as in the reference) ---
    # ------------------------------------------------------------------
    return out.to(torch.float32)