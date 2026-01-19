# Triton‑accelerated outgoing TriMul (AlphaFold‑3) – forward only
# ----------------------------------------------------------------------
# Public entry point: `custom_kernel`
# ----------------------------------------------------------------------
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Dict, Any

# ----------------------------------------------------------------------
# 1️⃣  Triton kernel – fuse mask, left‑gate and right‑gate (in‑place)
# ----------------------------------------------------------------------
@triton.jit
def _mask_gate_fused_kernel(
    left_ptr,          # half * [B,N,N,H] – will be overwritten with left*gate*mask
    right_ptr,         # half * [B,N,N,H] – will be overwritten with right*gate*mask
    mask_ptr,          # half * [B,N,N]   (0/1)
    left_gate_ptr,     # half * [B,N,N,H]
    right_gate_ptr,    # half * [B,N,N,H]
    B, N, H,           # runtime sizes (int64)
    BLOCK_SIZE: tl.constexpr,
):
    """
    For each element (b,i,j,h) compute:
        left  = left  * left_gate  * mask
        right = right * right_gate * mask
    All tensors are contiguous in row‑major order.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)          # linear element ids
    total = B * N * N * H
    mask = offs < total                                          # guard for tail

    # -------------- decode linear index --------------
    idx = offs
    b = idx // (N * N * H)
    idx = idx % (N * N * H)
    i = idx // (N * H)
    idx = idx % (N * H)
    j = idx // H
    h = idx % H

    # offsets in the flattened tensors
    off      = ((b * N + i) * N + j) * H + h            # for left/right/gates (with H)
    mask_off = (b * N + i) * N + j                     # for mask (no H)

    # ------------------- load -------------------------
    l  = tl.load(left_ptr  + off, mask=mask, other=tl.float16(0.0))
    r  = tl.load(right_ptr + off, mask=mask, other=tl.float16(0.0))
    lg = tl.load(left_gate_ptr  + off, mask=mask, other=tl.float16(0.0))
    rg = tl.load(right_gate_ptr + off, mask=mask, other=tl.float16(0.0))
    m  = tl.load(mask_ptr + mask_off, mask=mask, other=tl.float16(0.0))

    # ------------------- write back -------------------
    tl.store(left_ptr  + off,  l * lg * m, mask=mask)
    tl.store(right_ptr + off,  r * rg * m, mask=mask)


# ----------------------------------------------------------------------
# 2️⃣  Core TriMul implementation (torch + Triton)
# ----------------------------------------------------------------------
def _tri_mul_impl(
    data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]
) -> torch.Tensor:
    """
    Forward pass of the outgoing TriMul.
    Expected ``data`` tuple:
        (input_tensor, mask_tensor, weights_dict, config_dict)

    Returns:
        Tensor of shape [B, N, N, dim] (float32)
    """
    inp, mask, weights, cfg = data           # unpack
    B, N, _, dim = inp.shape                 # inp : [B,N,N,dim]
    H = cfg["hidden_dim"]
    nomask = cfg.get("nomask", True)

    device = inp.device

    # --------------------------------------------------------------
    # 2️⃣  Input LayerNorm – keep everything in fp16
    # --------------------------------------------------------------
    norm_w = weights["norm.weight"].to(device).to(torch.float16)
    norm_b = weights["norm.bias"].to(device).to(torch.float16)
    x = F.layer_norm(inp.to(torch.float16), (dim,), weight=norm_w, bias=norm_b)  # [B,N,N,dim]

    # --------------------------------------------------------------
    # 3️⃣  Fuse five linear layers (proj / gates / out_gate) – bias‑free
    # --------------------------------------------------------------
    fused_weight = torch.cat(
        [
            weights["left_proj.weight"],
            weights["left_gate.weight"],
            weights["right_proj.weight"],
            weights["right_gate.weight"],
            weights["out_gate.weight"],
        ],
        dim=0,
    ).to(device).to(torch.float16)               # (5*H, dim)

    x_flat = x.view(-1, dim)                     # [(B·N·N), dim]
    fused = F.linear(x_flat, fused_weight)       # [(B·N·N), 5*H]

    # --------------------------------------------------------------
    # 4️⃣  Slice & reshape to per‑tensor components
    # --------------------------------------------------------------
    left_proj_f   = fused[:, 0:H]
    left_gate_f   = fused[:, H:2*H]
    right_proj_f  = fused[:, 2*H:3*H]
    right_gate_f  = fused[:, 3*H:4*H]
    out_gate_f    = fused[:, 4*H:5*H]

    left_proj = left_proj_f.view(B, N, N, H)                      # fp16
    left_gate = torch.sigmoid(left_gate_f.view(B, N, N, H))       # fp16
    right_proj = right_proj_f.view(B, N, N, H)                    # fp16
    right_gate = torch.sigmoid(right_gate_f.view(B, N, N, H))    # fp16
    out_gate = torch.sigmoid(out_gate_f.view(B, N, N, H))        # fp16

    # --------------------------------------------------------------
    # 5️⃣  Apply mask × gate (fused Triton when a mask is present)
    # --------------------------------------------------------------
    if (not nomask) and (mask is not None):
        # ensure contiguous layout for the kernel
        left_proj = left_proj.contiguous()
        right_proj = right_proj.contiguous()
        left_gate = left_gate.contiguous()
        right_gate = right_gate.contiguous()
        mask_h = mask.to(device).to(torch.float16).contiguous()

        total = B * N * N * H
        BLOCK = 2048                                 # larger block → fewer program ids
        grid = (triton.cdiv(total, BLOCK),)
        _mask_gate_fused_kernel[grid](
            left_proj,
            right_proj,
            mask_h,
            left_gate,
            right_gate,
            B, N, H,
            BLOCK_SIZE=BLOCK,
        )
        left = left_proj
        right = right_proj
    else:
        left = left_proj * left_gate
        right = right_proj * right_gate

    # --------------------------------------------------------------
    # 6️⃣  Core O(N³) bilinear product – batched GEMM (fp16, Tensor‑cores)
    # --------------------------------------------------------------
    #   out[b,i,j,h] = Σ_k left[b,i,k,h] * right[b,j,k,h]
    left_mat = left.permute(0, 3, 1, 2).contiguous().view(B * H, N, N)   # (B*H,N,N)
    right_mat = right.permute(0, 3, 1, 2).contiguous().view(B * H, N, N) # (B*H,N,N)

    out_mat = torch.bmm(left_mat, right_mat.transpose(1, 2))              # (B*H,N,N)

    out = out_mat.view(B, H, N, N).permute(0, 2, 3, 1).contiguous()      # [B,N,N,H]

    # --------------------------------------------------------------
    # 7️⃣  Output LayerNorm (still fp16)
    # --------------------------------------------------------------
    out_norm_w = weights["to_out_norm.weight"].to(device).to(torch.float16)
    out_norm_b = weights["to_out_norm.bias"].to(device).to(torch.float16)
    out = F.layer_norm(out, (H,), weight=out_norm_w, bias=out_norm_b)

    # --------------------------------------------------------------
    # 8️⃣  Multiply by the final gate
    # --------------------------------------------------------------
    out = out * out_gate

    # --------------------------------------------------------------
    # 9️⃣  Final linear projection back to `dim`
    # --------------------------------------------------------------
    to_out_w = weights["to_out.weight"].to(device).to(torch.float16)
    out = F.linear(out, to_out_w)                     # [B,N,N,dim] (fp16)

    # --------------------------------------------------------------
    # 🎯  Cast back to FP32 for the public API
    # --------------------------------------------------------------
    return out.to(torch.float32)


# ----------------------------------------------------------------------
# Public entry point – optionally torch.compile for extra speed
# ----------------------------------------------------------------------
try:
    # `fullgraph=True` forces a single compiled graph which removes Python
    # overhead on the hot path.
    custom_kernel = torch.compile(_tri_mul_impl, fullgraph=True, dynamic=False)
except Exception:   # pragma: no cover
    custom_kernel = _tri_mul_impl