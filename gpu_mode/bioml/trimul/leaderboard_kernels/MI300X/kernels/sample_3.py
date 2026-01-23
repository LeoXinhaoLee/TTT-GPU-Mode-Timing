#!POPCORN leaderboard trimul

from task import input_t, output_t

import torch
from torch.nn import init
import torch.nn.functional as F

def custom_kernel(data: input_t) -> output_t:
    """
    Inline implementation of TriMul.forward.
    Expects:
      input_tensor: Tensor shape (batch, seq_len, seq_len, dim)
      mask:         Tensor shape (batch, seq_len, seq_len)  (or broadcastable to that)
      weights:      dict of tensors (optional). Keys used:
                    'norm.weight', 'norm.bias',
                    'left_proj.weight', 'right_proj.weight',
                    'left_gate.weight', 'right_gate.weight', 'out_gate.weight',
                    'to_out_norm.weight', 'to_out_norm.bias',
                    'to_out.weight'
      config:       dict with "dim" and "hidden_dim"
    Returns:
      Tensor shape (batch, seq_len, seq_len, dim)
    """
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dtype = input_tensor.dtype

    d = config["dim"]
    h = config["hidden_dim"]

    # helper to fetch or initialize a weight tensor
    def get_weight(key):
        try:
            w = weights[key]
            # Only move/cast if device or dtype mismatch
            if w.device != device or w.dtype != dtype:
                w = w.to(device=device, dtype=dtype)
            return w
        except Exception as e:
            raise Exception(f"Error fetching weight '{key}': {e}")

    # LayerNorm params for input x (norm over last dim d)
    norm_w = get_weight('norm.weight')
    norm_b = get_weight('norm.bias')

    # projection & gate weights (Linear(in=d, out=h) -> weight shape (h, d))
    left_proj_w  = get_weight('left_proj.weight')
    right_proj_w = get_weight('right_proj.weight')
    left_gate_w  = get_weight('left_gate.weight')
    right_gate_w = get_weight('right_gate.weight')
    out_gate_w   = get_weight('out_gate.weight')

    # output layernorm (over hidden dim h)
    to_out_norm_w = get_weight('to_out_norm.weight')
    to_out_norm_b = get_weight('to_out_norm.bias')

    # to_out linear weight: Linear(in=h, out=d) -> weight shape (d, h)
    to_out_w = get_weight('to_out.weight')

    # --- forward ---
    with torch.amp.autocast('cuda'):
        x = input_tensor  # (batch, seq_len, seq_len, d)

        d = norm_w.shape[0]
        h = left_proj_w.shape[0]

        # Fuse LayerNorm and first set of linear projections
        x_norm = F.layer_norm(x, (d,), weight=norm_w, bias=norm_b, eps=1e-5)

        stacked_w = torch.cat(
            [left_proj_w, right_proj_w, left_gate_w, right_gate_w, out_gate_w],
            dim=0
        )
        projected = F.linear(x_norm, stacked_w)  # (B, L, L, 5*h)

        lr = projected[..., :2*h]
        lr_gate = projected[..., 2*h:4*h]
        out_gate = projected[..., 4*h:]

        if mask is not None:
            lr_gated = lr * torch.sigmoid(lr_gate) * mask.unsqueeze(-1)
        else:
            lr_gated = lr * torch.sigmoid(lr_gate)

        left_gated = lr_gated[..., :h]
        right_gated = lr_gated[..., h:]

        # This works in basically the same time as hand-written code
        out = torch.einsum("bikh,bjkh->bijh", left_gated, right_gated)

        out_norm = F.layer_norm(out, (h,), weight=to_out_norm_w, bias=to_out_norm_b, eps=1e-5)
        out_gated = out_norm * torch.sigmoid(out_gate)
        final_out = F.linear(out_gated, to_out_w)

        return final_out