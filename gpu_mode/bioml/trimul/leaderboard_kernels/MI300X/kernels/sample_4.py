#!POPCORN leaderboard trimul

# This is a submission template for popcorn leaderboard 'trimul'.
# Your task is as follows:
# > For a more complete description, see: https://tinyurl.com/gpumode-trimul
# > You will be implementing a Triangle Multiplicative Update (TriMul) module that is a core operation
# > for AlphaFold3, Chai, Protenix, and other protein structure prediction models in BioML.
# > 
# > The TriMul operator operates over a 4D tensor of shape [B, N, N, C]. 
# > 
# > Your task:
# > - Implement the "outgoing" version of the TriMul operator from the AlphaFold3 paper.
# > - You will not have to compute or store gradients for this version. You will only need to implement the forward pass.
# > 
# > Input:
# > - `data`: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
# >   - input: Input tensor of shape [bs, seq_len, seq_len, dim]
# >   - mask: Mask tensor of shape [bs, seq_len, seq_len]
# >   - weights: Dictionary containing model weights
# >   - config: Dictionary containing model configuration parameters
# > 
# > Output:
# > - Tuple containing:
# >   - output: Processed tensor [bs, seq_len, seq_len, dim]
# The deadline for this leaderboard is 2025-09-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

import torch
from torch import nn, einsum
from task import input_t, output_t

import triton
import triton.language as tl

class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_right_proj = nn.Linear(dim, hidden_dim*2, bias=False, dtype=torch.bfloat16)
        self.left_right_gate = nn.Linear(dim, hidden_dim*2, bias=False, dtype=torch.bfloat16)
        
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]

        Returns:
            output: [bs, seq_len, seq_len, dim]
        """
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x.to(torch.float32)).to(torch.bfloat16)

        left_right = self.left_right_proj(x)

        mask = mask.to(torch.bfloat16).unsqueeze(-1)
        left_right *= mask

        left_right_gate = self.left_right_gate(x).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left_right *= left_right_gate # B,L,L,2D

        out = (left_right[...,:self.hidden_dim].permute(0,3,1,2)@left_right[...,self.hidden_dim:].permute(0,3,2,1)).permute(0,2,3,1)
        # This above operation is the same as the following:
        # out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)
        # out: B,L,L,C
        # left,right: B,L,C
        # # Compute using nested loops
        # for b in range(batch_size):
        #     for i in range(seq_len):
        #         for j in range(seq_len):
        #             # Compute each output element
        #             for k in range(seq_len):
        #                 out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

        out = self.to_out_norm(out.to(torch.float32))
        out *= out_gate
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
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    # Fill in the given weights of the model
    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    # trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.bfloat16))
    # trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.bfloat16))
    trimul.left_right_proj.weight = nn.Parameter(torch.vstack([weights['left_proj.weight'],weights['right_proj.weight']]).to(torch.bfloat16))
    # trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.bfloat16))
    # trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.bfloat16))
    trimul.left_right_gate.weight = nn.Parameter(torch.vstack([weights['left_gate.weight'],weights['right_gate.weight']]).to(torch.bfloat16))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    with torch.no_grad():
        output = trimul(input_tensor, mask).to(torch.float32)

    return output