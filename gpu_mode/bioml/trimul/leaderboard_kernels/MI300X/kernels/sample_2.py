#!POPCORN leaderboard trimul

import torch
from torch import nn
import triton
import triton.language as tl
from task import input_t, output_t

torch.set_float32_matmul_precision('high')

@triton.jit
def trimul_fused_projection(
    x_ptr,
    mask_ptr, 
    l_proj_ptr,
    l_gate_ptr,
    r_proj_ptr,
    r_gate_ptr,
    o_gate_ptr,
    l_output_ptr,
    r_output_ptr,
    g_output_ptr,
    x_stride_b,
    x_stride_s,
    x_stride_m,
    x_stride_k,
    mask_stride_b,
    mask_stride_s,
    mask_stride_m,
    mask_stride_k,
    w_stride_k,
    w_stride_n,
    o_stride_b,
    o_stride_s,
    o_stride_m,
    o_stride_n,
    g_stride_b,
    g_stride_s,
    g_stride_m,
    g_stride_n,
    S: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr
):

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N

    b = pid_b // S
    s = pid_b % S

    batch_offset_x = b * x_stride_b + s * x_stride_s
    batch_offset_mask = b * mask_stride_b + s * mask_stride_s
    batch_offset_out = b * o_stride_b + s * o_stride_s
    batch_offset_gate = b * g_stride_b + s * g_stride_s

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr + batch_offset_x,
        shape=(M, K),
        strides=(x_stride_m, x_stride_k),
        offsets=(m_offset, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    mask_block_ptr = tl.make_block_ptr(
        base=mask_ptr + batch_offset_mask,
        shape=(M, 1),
        strides=(mask_stride_m, mask_stride_k),
        offsets=(m_offset, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )

    l_proj_block_ptr = tl.make_block_ptr(
        base=l_proj_ptr,
        shape=(K, N),
        strides=(w_stride_k, w_stride_n),
        offsets=(0, n_offset),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    r_proj_block_ptr = tl.make_block_ptr(
        base=r_proj_ptr,
        shape=(K, N),
        strides=(w_stride_k, w_stride_n),
        offsets=(0, n_offset),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    l_gate_block_ptr = tl.make_block_ptr(
        base=l_gate_ptr,
        shape=(K, N),
        strides=(w_stride_k, w_stride_n),
        offsets=(0, n_offset),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    r_gate_block_ptr = tl.make_block_ptr(
        base=r_gate_ptr,
        shape=(K, N),
        strides=(w_stride_k, w_stride_n),
        offsets=(0, n_offset),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    o_gate_block_ptr = tl.make_block_ptr(
        base=o_gate_ptr,
        shape=(K, N),
        strides=(w_stride_k, w_stride_n),
        offsets=(0, n_offset),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    l_output_block_ptr = tl.make_block_ptr(
        base=l_output_ptr + batch_offset_out,
        shape=(M, N),
        strides=(o_stride_m, o_stride_n),
        offsets=(m_offset, n_offset),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    r_output_block_ptr = tl.make_block_ptr(
        base=r_output_ptr + batch_offset_out,
        shape=(M, N),
        strides=(o_stride_m, o_stride_n),
        offsets=(m_offset, n_offset),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    g_output_block_ptr = tl.make_block_ptr(
        base=g_output_ptr + batch_offset_gate,
        shape=(M, N),
        strides=(g_stride_m, g_stride_n),
        offsets=(m_offset, n_offset),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    l_proj_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    r_proj_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    l_gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    r_gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    o_gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):

        x = tl.load(x_block_ptr, boundary_check=(0, 1))
        l_proj = tl.load(l_proj_block_ptr, boundary_check=(0, 1))
        r_proj = tl.load(r_proj_block_ptr, boundary_check=(0, 1))
        l_gate = tl.load(l_gate_block_ptr, boundary_check=(0, 1))
        r_gate = tl.load(r_gate_block_ptr, boundary_check=(0, 1))
        out_gate = tl.load(o_gate_block_ptr, boundary_check=(0, 1))

        l_proj_acc += tl.dot(x, l_proj)
        r_proj_acc += tl.dot(x, r_proj)
        l_gate_acc += tl.dot(x, l_gate)
        r_gate_acc += tl.dot(x, r_gate)
        o_gate_acc += tl.dot(x, out_gate)

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
        l_proj_block_ptr = tl.advance(l_proj_block_ptr, (BLOCK_K, 0))
        r_proj_block_ptr = tl.advance(r_proj_block_ptr, (BLOCK_K, 0))
        l_gate_block_ptr = tl.advance(l_gate_block_ptr, (BLOCK_K, 0))
        r_gate_block_ptr = tl.advance(r_gate_block_ptr, (BLOCK_K, 0))
        o_gate_block_ptr = tl.advance(o_gate_block_ptr, (BLOCK_K, 0))

    mask = tl.load(mask_block_ptr, boundary_check=(0, 1))
    l_proj_acc = l_proj_acc * mask
    r_proj_acc = r_proj_acc * mask

    l_gate_acc = tl.sigmoid(l_gate_acc)
    r_gate_acc = tl.sigmoid(r_gate_acc)
    o_gate_acc = tl.sigmoid(o_gate_acc)

    tl.store(l_output_block_ptr, l_proj_acc * l_gate_acc)
    tl.store(r_output_block_ptr, r_proj_acc * r_gate_acc)
    tl.store(g_output_block_ptr, o_gate_acc)


def launch_trimul_fused_projection(x, mask, l_proj, l_gate, r_proj, r_gate, o_gate):
    B, S, M, K = x.shape
    N = l_proj.shape[1]

    BLOCK_M = min(M, 64)
    BLOCK_N = min(N, 64)
    BLOCK_K = min(K, 32)

    grid_x = triton.cdiv(M, BLOCK_M)
    grid_y = triton.cdiv(N, BLOCK_N)
    grid_z = B * S

    l_output = torch.empty((B, N, S, M), device=x.device, dtype=x.dtype)
    r_output = torch.empty((B, N, S, M), device=x.device, dtype=x.dtype)
    g_output = torch.empty((B, S, M, N), device=x.device, dtype=x.dtype)

    trimul_fused_projection[(grid_x, grid_y, grid_z)](
        x,
        mask,
        l_proj,
        l_gate,
        r_proj,
        r_gate,
        o_gate,
        l_output,
        r_output,
        g_output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        mask.stride(3),
        l_proj.stride(0),
        l_proj.stride(1),
        l_output.stride(0),
        l_output.stride(2),
        l_output.stride(3),
        l_output.stride(1),
        g_output.stride(0),
        g_output.stride(1),
        g_output.stride(2),
        g_output.stride(3),
        S,
        M,
        K,
        N,
        BLOCK_M,
        BLOCK_K,
        BLOCK_N,
    )
    
    return l_output, r_output, g_output


class TriMulTriton(torch.nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.norm = torch.nn.LayerNorm(dim)
        self.to_out_norm = torch.nn.LayerNorm(hidden_dim)
        self.to_out = torch.nn.Linear(hidden_dim, dim, bias=False)

        self.left_proj_weight = None
        self.left_gate_weight = None
        self.right_proj_weight = None
        self.right_gate_weight = None
        self.out_gate_weight = None

    def forward(self, x, mask):
        x = self.norm(x)

        l_out, r_out, g_out = launch_trimul_fused_projection(
            x,
            mask.unsqueeze(-1),
            self.left_proj_weight,
            self.left_gate_weight,
            self.right_proj_weight,
            self.right_gate_weight,
            self.out_gate_weight
        )

        out = torch.einsum('... d i k, ... d j k -> ... i j d', l_out, r_out)

        out = self.to_out_norm(out)
        out = out * g_out
        return self.to_out(out)

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of TriMul using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, mask: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, seq_len, dim]
            - mask: Mask tensor of shape [batch_size, seq_len, seq_len]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
    """
    input_tensor, mask, weights, config = data
    trimul = TriMulTriton(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    # Fill in the given weights of the model
    trimul.left_proj_weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32).T)
    trimul.right_proj_weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32).T)
    trimul.left_gate_weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32).T)
    trimul.right_gate_weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32).T)
    trimul.out_gate_weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32).T)
    
    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)
    return output
