import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from task import input_t, output_t
from typing import Tuple
import time
import triton
import triton.language as tl
# import aiter
@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int 
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: torch.Tensor
    Q_proj_up_weight: torch.Tensor
    KV_proj_down_weight: torch.Tensor
    KV_proj_up_weight: torch.Tensor
    wo_weight: torch.Tensor

class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16))
        self.seq_len = 0
        self.zero()

    def zero(self) -> None:
        self.data.zero_()
    
    def get_data(self) -> torch.Tensor:
        return self.data

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(c_kv.dtype)
        self.data[
            :, self.seq_len : self.seq_len + c_kv.size(1), :
        ] = c_kv
        self.seq_len += c_kv.size(1)

        return self.data[:, :self.seq_len], self.seq_len

def torch_rope(
    theta: torch.Tensor,
    x: torch.Tensor,
    start_pos: int = 0,
):
    # x: (-1, seq_len, d_model)
    seq_len = x.size(-2)
    d_model = x.size(-1)
    old_shape = x.shape
    x = x.reshape(-1, seq_len, d_model)
    seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device) # (seq_len,)
    idx_theta = torch.einsum('s,d->sd', seq_idx, theta) # (seq_len, d_model//2)
    
    cos = idx_theta.cos()
    sin = idx_theta.sin()
    x0 = x[:, :, :d_model//2] # (-1, seq_len, d_model//2)
    x1 = x[:, :, d_model//2:] # (-1, seq_len, d_model//2)
    new_x0 = x0 * cos - x1 * sin # (-1, seq_len, d_model//2)
    new_x1 = x0 * sin + x1 * cos # (-1, seq_len, d_model//2)
    new_x = torch.cat([new_x0, new_x1], dim=-1) # (-1, seq_len, d_model)
    new_x = new_x.view(old_shape) # (batch_size, seq_len, d_model)
    return new_x
@triton.jit
def q_rope_kernel(
    x,
    y,
    theta,
    start_position,
    batchsize,
    seq_len,
    n_heads,
    rope_head_dim: tl.constexpr,
    x_batch_stride,
    x_seq_stride,
    x_head_stride,
    y_batch_stride,
    y_seq_stride,
    y_head_stride,
    head_block_size: tl.constexpr,
):
    # x:  (batchsize, seq_len, n_heads, rope_head_dim) 
    # y:  (batchsize, seq_len, n_heads, rope_head_dim)
    # theta: (rope_head_dim // 2, )
    # grid: (cdiv(n_heads, head_block_size), seq_len, batchsize)
    
    batch_idx = tl.program_id(2)
    seq_idx = tl.program_id(1)
    head_begin = tl.program_id(0) * head_block_size
    
    x0_block_ptr = tl.make_block_ptr(
        x + batch_idx * x_batch_stride + seq_idx * x_seq_stride + head_begin * x_head_stride,
        shape=(tl.minimum(head_block_size, n_heads - head_begin), rope_head_dim//2),
        strides=(x_head_stride, 1),
        offsets=(0, 0),
        block_shape=(head_block_size, rope_head_dim//2),
        order=(1, 0),
    )
    x1_block_ptr = tl.make_block_ptr(
        x + batch_idx * x_batch_stride + seq_idx * x_seq_stride + head_begin * x_head_stride + rope_head_dim//2,
        shape=(tl.minimum(head_block_size, n_heads - head_begin), rope_head_dim//2),
        strides=(x_head_stride, 1),
        offsets=(0, 0),
        block_shape=(head_block_size, rope_head_dim//2),
        order=(1, 0),
    )
    y0_block_ptr = tl.make_block_ptr(
        y + batch_idx * y_batch_stride + seq_idx * y_seq_stride + head_begin * y_head_stride,
        shape=(tl.minimum(head_block_size, n_heads - head_begin), rope_head_dim//2),
        strides=(y_head_stride, 1),
        offsets=(0, 0),
        block_shape=(head_block_size, rope_head_dim//2),
        order=(1, 0),
    )
    y1_block_ptr = tl.make_block_ptr(
        y + batch_idx * y_batch_stride + seq_idx * y_seq_stride + head_begin * y_head_stride + rope_head_dim//2,
        shape=(tl.minimum(head_block_size, n_heads - head_begin), rope_head_dim//2),
        strides=(y_head_stride, 1),
        offsets=(0, 0),
        block_shape=(head_block_size, rope_head_dim//2),
        order=(1, 0),
    )
    theta_block_ptr = tl.make_block_ptr(
        theta,
        shape=(1, rope_head_dim//2, ),
        strides=(0, 1),
        offsets=(0, 0),
        block_shape=(1, rope_head_dim//2),
        order=(1, 0),
    )
    x0 = tl.load(x0_block_ptr, boundary_check=(0, 1)) # (head_block_size, rope_head_dim//2)
    x1 = tl.load(x1_block_ptr, boundary_check=(0, 1)) # (head_block_size, rope_head_dim//2)
    theta = tl.load(theta_block_ptr) # (1, rope_head_dim//2)
    position = (start_position + seq_idx) # (1, )
    # precision!!!!
    theta = (theta * position) # (1, rope_head_dim//2)
    cos = tl.cos(theta).to(tl.bfloat16) # (1, rope_head_dim//2)
    sin = tl.sin(theta).to(tl.bfloat16) # (1, rope_head_dim//2)
    
    y0 = (x0 * cos) - (x1 * sin) # (head_block_size, rope_head_dim//2)
    y1 = (x0 * sin) + (x1 * cos) # (head_block_size, rope_head_dim//2)
    # convert to the type of y
    y0 = y0.to(y.dtype.element_ty)
    y1 = y1.to(y.dtype.element_ty)
    tl.store(y0_block_ptr, y0, boundary_check=(0, 1)) # (head_block_size, rope_head_dim//2)
    tl.store(y1_block_ptr, y1, boundary_check=(0, 1)) # (head_block_size, rope_head_dim//2)


@triton.jit
def q_rope_cos_sin_kernel(
    x,
    y,
    cos,
    sin,
    start_position,
    batchsize,
    seq_len,
    n_heads,
    rope_head_dim: tl.constexpr,
    x_batch_stride,
    x_seq_stride,
    x_head_stride,
    y_batch_stride,
    y_seq_stride,
    y_head_stride,
    head_block_size: tl.constexpr,
):
    # x:  (batchsize, seq_len, n_heads, rope_head_dim) 
    # y:  (batchsize, seq_len, n_heads, rope_head_dim)
    # cos: (max_seq_len, rope_head_dim // 2)
    # sin: (max_seq_len, rope_head_dim // 2)
    # start_position: (1, )
    # grid: (cdiv(n_heads, head_block_size), seq_len, batchsize)
    
    batch_idx = tl.program_id(2)
    seq_idx = tl.program_id(1)
    head_begin = tl.program_id(0) * head_block_size
    
    x0_block_ptr = tl.make_block_ptr(
        x + batch_idx * x_batch_stride + seq_idx * x_seq_stride + head_begin * x_head_stride,
        shape=(tl.minimum(head_block_size, n_heads - head_begin), rope_head_dim//2),
        strides=(x_head_stride, 1),
        offsets=(0, 0),
        block_shape=(head_block_size, rope_head_dim//2),
        order=(1, 0),
    )
    x1_block_ptr = tl.make_block_ptr(
        x + batch_idx * x_batch_stride + seq_idx * x_seq_stride + head_begin * x_head_stride + rope_head_dim//2,
        shape=(tl.minimum(head_block_size, n_heads - head_begin), rope_head_dim//2),
        strides=(x_head_stride, 1),
        offsets=(0, 0),
        block_shape=(head_block_size, rope_head_dim//2),
        order=(1, 0),
    )
    y0_block_ptr = tl.make_block_ptr(
        y + batch_idx * y_batch_stride + seq_idx * y_seq_stride + head_begin * y_head_stride,
        shape=(tl.minimum(head_block_size, n_heads - head_begin), rope_head_dim//2),
        strides=(y_head_stride, 1),
        offsets=(0, 0),
        block_shape=(head_block_size, rope_head_dim//2),
        order=(1, 0),
    )
    y1_block_ptr = tl.make_block_ptr(
        y + batch_idx * y_batch_stride + seq_idx * y_seq_stride + head_begin * y_head_stride + rope_head_dim//2,
        shape=(tl.minimum(head_block_size, n_heads - head_begin), rope_head_dim//2),
        strides=(y_head_stride, 1),
        offsets=(0, 0),
        block_shape=(head_block_size, rope_head_dim//2),
        order=(1, 0),
    )
    cos_block_ptr = tl.make_block_ptr(
        cos + (start_position + seq_idx) * (rope_head_dim//2),
        shape=(1, rope_head_dim//2),
        strides=(0, 1),
        offsets=(0, 0),
        block_shape=(1, rope_head_dim//2),
        order=(1, 0),
    )
    sin_block_ptr = tl.make_block_ptr(
        sin + (start_position + seq_idx) * (rope_head_dim//2),
        shape=(1, rope_head_dim//2),
        strides=(0, 1),
        offsets=(0, 0),
        block_shape=(1, rope_head_dim//2),
        order=(1, 0),
    )
    x0 = tl.load(x0_block_ptr, boundary_check=(0, 1)) # (head_block_size, rope_head_dim//2)
    x1 = tl.load(x1_block_ptr, boundary_check=(0, 1)) # (head_block_size, rope_head_dim//2)
    cos = tl.load(cos_block_ptr) # (1, rope_head_dim//2)
    sin = tl.load(sin_block_ptr) # (1, rope_head_dim//2)
    
    y0 = (x0 * cos) - (x1 * sin) # (head_block_size, rope_head_dim//2)
    y1 = (x0 * sin) + (x1 * cos) # (head_block_size, rope_head_dim//2)
    # convert to the type of y
    y0 = y0.to(y.dtype.element_ty)
    y1 = y1.to(y.dtype.element_ty)
    tl.store(y0_block_ptr, y0, boundary_check=(0, 1)) # (head_block_size, rope_head_dim//2)
    tl.store(y1_block_ptr, y1, boundary_check=(0, 1)) # (head_block_size, rope_head_dim//2)

@triton.jit
def k_rope_cos_sin_kernel(
    x,
    y,
    cos,
    sin,
    batchsize,
    seq_len,
    rope_head_dim: tl.constexpr,
    x_batch_stride,
    x_seq_stride,
    y_batch_stride,
    y_seq_stride,
    batch_block_size: tl.constexpr,
):
    # x:  (batchsize, seq_len, rope_head_dim)
    # y:  (batchsize, seq_len, rope_head_dim)
    # cos: (max_seq_len, rope_head_dim // 2)
    # sin: (max_seq_len, rope_head_dim // 2)
    # grid: (seq_len, cdiv(batchsize, batch_block_size))
    batch_begin = tl.program_id(1) * batch_block_size
    seq_idx = tl.program_id(0)
    
    x0_block_ptr = tl.make_block_ptr(
        x + batch_begin * x_batch_stride + seq_idx * x_seq_stride,
        shape=(tl.minimum(batch_block_size, batchsize - batch_begin), rope_head_dim // 2),
        strides=(x_batch_stride, 1),
        offsets=(0, 0),
        block_shape=(batch_block_size, rope_head_dim // 2),
        order=(1, 0),
    )
    x1_block_ptr = tl.make_block_ptr(
        x + batch_begin * x_batch_stride + seq_idx * x_seq_stride + rope_head_dim // 2,
        shape=(tl.minimum(batch_block_size, batchsize - batch_begin), rope_head_dim // 2),
        strides=(x_batch_stride, 1),
        offsets=(0, 0),
        block_shape=(batch_block_size, rope_head_dim // 2),
        order=(1, 0),
    )
    y0_block_ptr = tl.make_block_ptr(
        y + batch_begin * y_batch_stride + seq_idx * y_seq_stride,
        shape=(tl.minimum(batch_block_size, batchsize - batch_begin), rope_head_dim // 2),
        strides=(y_batch_stride, 1),
        offsets=(0, 0),
        block_shape=(batch_block_size, rope_head_dim // 2),
        order=(1, 0),
    )
    y1_block_ptr = tl.make_block_ptr(
        y + batch_begin * y_batch_stride + seq_idx * y_seq_stride + rope_head_dim // 2,
        shape=(tl.minimum(batch_block_size, batchsize - batch_begin), rope_head_dim // 2),
        strides=(y_batch_stride, 1),
        offsets=(0, 0),
        block_shape=(batch_block_size, rope_head_dim // 2),
        order=(1, 0),
    )
    cos_block_ptr = tl.make_block_ptr(
        cos + seq_idx * (rope_head_dim // 2),
        shape=(1, rope_head_dim // 2),
        strides=(0, 1),
        offsets=(0, 0),
        block_shape=(1, rope_head_dim // 2),
        order=(1, 0),
    )
    sin_block_ptr = tl.make_block_ptr(
        sin + seq_idx * (rope_head_dim // 2),
        shape=(1, rope_head_dim // 2),
        strides=(0, 1),
        offsets=(0, 0),
        block_shape=(1, rope_head_dim // 2),
        order=(1, 0),
    )
    x0 = tl.load(x0_block_ptr, boundary_check=(0, )) # (batch_block_size, rope_head_dim//2)
    x1 = tl.load(x1_block_ptr, boundary_check=(0, )) # (batch_block_size, rope_head_dim//2)
    cos = tl.load(cos_block_ptr) # (1, rope_head_dim//2)
    sin = tl.load(sin_block_ptr) # (1, rope_head_dim//2)
    y0 = (x0 * cos) - (x1 * sin) # (batch_block_size, rope_head_dim//2)
    y1 = (x0 * sin) + (x1 * cos) # (batch_block_size, rope_head_dim//2)
    # convert to the type of y
    y0 = y0.to(y.dtype.element_ty)
    y1 = y1.to(y.dtype.element_ty)
    tl.store(y0_block_ptr, y0, boundary_check=(0, )) # (batch_block_size, rope_head_dim//2)
    tl.store(y1_block_ptr, y1, boundary_check=(0, )) # (batch_block_size, rope_head_dim//2)
    
thetas = {
    d_model: 10000 ** (-torch.arange(0, d_model//2,dtype=torch.bfloat16) / (d_model//2)) for d_model in [64,]
}
thetas = {
    d_model: theta.cuda() for d_model, theta in thetas.items()
}
max_seqlen = 8192
cos_tensors = {
    d_model: 
        torch.einsum('s,d->sd', torch.arange(0, max_seqlen,).cuda(), theta).cos()
    for d_model, theta in thetas.items()
}
sin_tensors = {
    d_model: 
        torch.einsum('s,d->sd', torch.arange(0, max_seqlen,).cuda(), theta).sin()
    for d_model, theta in thetas.items()
}
class Timer:
    def __init__(self, label: str):
        self.label = label
        self.start_time = None
        self.end_time = None
        self.interval = None
    def __enter__(self):
        # self.start = time.time()
        return self
    def __exit__(self, *args):
        # torch.cuda.synchronize()
        # self.end = time.time()
        # self.interval = (self.end - self.start) * 1000 # in ms
        # print(f"{self.label}: {self.interval:.4f} ms")
        return False
    
@triton.jit
def mla_decode_kernel(
    q_nope, kv_nope, 
    q_rope, k_rope,
    out,
    scale,
    batch_size,
    kv_len,
    q_nope_batch_stride,
    q_nope_head_stride,
    kv_nope_batch_stride,
    kv_nope_seq_stride,
    n_heads: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    HEAD_BLOCK_SIZE: tl.constexpr = 16,
    SEQ_BLOCK_SIZE: tl.constexpr = 32,
):
    # q_nope: (batch_size, n_heads, kv_lora_rank) : (q_nope_batch_stride, q_nope_head_stride, 1)
    # kv_nope: (batch_size, kv_len, kv_lora_rank) : (kv_nope_batch_stride, kv_nope_seq_stride, 1)
    # q_rope: (batch_size, n_heads, qk_rope_head_dim) : (n_heads * qk_rope_head_dim, qk_rope_head_dim, 1)
    # k_rope: (batch_size, kv_len, qk_rope_head_dim) : (kv_len * qk_rope_head_dim, qk_rope_head_dim, 1)
    # out: (batch_size, n_heads, kv_lora_rank) : (n_heads * kv_lora_rank, kv_lora_rank, 1)
    # grid: (cdiv(n_heads, HEAD_BLOCK_SIZE), batch_size)
    head_begin = tl.program_id(0) * HEAD_BLOCK_SIZE
    batch_idx = tl.program_id(1)
    
    q_rope_head_stride = qk_rope_head_dim
    q_rope_batch_stride = n_heads * qk_rope_head_dim
    q_rope_block_ptr = tl.make_block_ptr(
        q_rope + batch_idx * q_rope_batch_stride + head_begin * q_rope_head_stride,
        shape=(tl.minimum(HEAD_BLOCK_SIZE, n_heads - head_begin), qk_rope_head_dim),
        strides=(q_rope_head_stride, 1),
        offsets=(0, 0),
        block_shape=(HEAD_BLOCK_SIZE, qk_rope_head_dim),
        order=(1, 0),
    ) # (HEAD_BLOCK_SIZE, qk_rope_head_dim)
    
    k_rope_seq_stride = qk_rope_head_dim
    k_rope_batch_stride = kv_len * qk_rope_head_dim
    k_rope_block_ptr = tl.make_block_ptr(
        k_rope + batch_idx * k_rope_batch_stride,
        shape=(kv_len, qk_rope_head_dim),
        strides=(k_rope_seq_stride, 1),
        offsets=(0, 0),
        block_shape=(SEQ_BLOCK_SIZE, qk_rope_head_dim),
        order=(1, 0),
    ) # (SEQ_BLOCK_SIZE, qk_rope_head_dim)
    
    q_nope_block_ptr = tl.make_block_ptr(
        q_nope + batch_idx * q_nope_batch_stride + head_begin * q_nope_head_stride,
        shape=(tl.minimum(HEAD_BLOCK_SIZE, n_heads - head_begin), kv_lora_rank),
        strides=(q_nope_head_stride, 1),
        offsets=(0, 0),
        block_shape=(HEAD_BLOCK_SIZE, kv_lora_rank),
        order=(1, 0),
    ) # (HEAD_BLOCK_SIZE, kv_lora_rank)
    kv_nope_block_ptr = tl.make_block_ptr(
        kv_nope + batch_idx * kv_nope_batch_stride,
        shape=(kv_len, kv_lora_rank),
        strides=(kv_nope_seq_stride, 1),
        offsets=(0, 0),
        block_shape=(SEQ_BLOCK_SIZE, kv_lora_rank),
        order=(1, 0),
    ) # (SEQ_BLOCK_SIZE, kv_lora_rank)
    
    out_block_ptr = tl.make_block_ptr(
        out + batch_idx * (n_heads * kv_lora_rank) + head_begin * kv_lora_rank,
        shape=(tl.minimum(HEAD_BLOCK_SIZE, n_heads - head_begin), kv_lora_rank),
        strides=(kv_lora_rank, 1),
        offsets=(0, 0),
        block_shape=(HEAD_BLOCK_SIZE, kv_lora_rank),
        order=(1, 0),
    ) # (HEAD_BLOCK_SIZE, kv_lora_rank)
    
    # 
    qk_scale = scale * 1.44269504
    # initialize pointer to m and l
    m_i = tl.zeros([HEAD_BLOCK_SIZE], dtype=tl.float32) - float("inf") #(HEAD_BLOCK_SIZE, )
    l_i = tl.zeros([HEAD_BLOCK_SIZE], dtype=tl.float32) + 1.0 #(HEAD_BLOCK_SIZE, )
    acc = tl.zeros([HEAD_BLOCK_SIZE, kv_lora_rank], dtype=tl.float32) # (HEAD_BLOCK_SIZE, kv_lora_rank)
    
    # TODO: remove boundary check
    q_rope = tl.load(q_rope_block_ptr, boundary_check=(0,)) # (HEAD_BLOCK_SIZE, qk_rope_head_dim)
    q_nope = tl.load(q_nope_block_ptr, boundary_check=(0,)) # (HEAD_BLOCK_SIZE, kv_lora_rank)
    for seq_begin in range(0, kv_len, SEQ_BLOCK_SIZE):
        k_rope_block = tl.load(k_rope_block_ptr, boundary_check=(0,)) # (SEQ_BLOCK_SIZE, qk_rope_head_dim)
        qk = tl.dot(q_rope, k_rope_block.T) # (HEAD_BLOCK_SIZE, SEQ_BLOCK_SIZE)

        kv_nope_block = tl.load(kv_nope_block_ptr, boundary_check=(0,)) # (SEQ_BLOCK_SIZE, kv_lora_rank)
        qk += tl.dot(q_nope, kv_nope_block.T) # (HEAD_BLOCK_SIZE, SEQ_BLOCK_SIZE)
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
                
        p = tl.math.exp2(qk)
        
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        p = p.to(tl.bfloat16) # convert p to the same type as kv_nope
        
        acc = tl.dot(p, kv_nope_block, acc) # (HEAD_BLOCK_SIZE, kv_lora_rank)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        k_rope_block_ptr = tl.advance(k_rope_block_ptr, (SEQ_BLOCK_SIZE, 0)) 
        kv_nope_block_ptr = tl.advance(kv_nope_block_ptr, (SEQ_BLOCK_SIZE, 0)) 
    acc = acc / l_i[:, None] # (HEAD_BLOCK_SIZE, kv_lora_rank)
    tl.store(out_block_ptr, acc.to(tl.bfloat16), boundary_check=(0,)) # (HEAD_BLOCK_SIZE, kv_lora_rank)
def mla_decode_v0(
    config: Config,
    x: torch.Tensor,
    kv_cache: KVCache,
) :
    # dim:                7168
    # n_heads:             128
    # q_lora_rank:        1536
    # kv_lora_rank:       512
    # qk_nope_head_dim:   128
    # qk_rope_head_dim:   64
    # v_head_dim:        128
    # Q_proj_down_weight:   (1536, 7168)
    # Q_proj_up_weight:     ((128 + 64) * 128, 1536) = (24576, 1536)
    # KV_proj_down_weight:  (512 + 64, 7168) = (576, 7168)
    # KV_proj_up_weight:    ((128 + 128) * 128, 512) = (32768, 512)
    # wo_weight:            (7168, 128 * 128) = (7168, 16384)
    # kv_cache_shape:     (batchsize, max_seq_len, kv_lora_rank + qk_rope_head_dim)
    # Q_proj_down_weight:   (q_lora_rank, dim)
    # Q_proj_up_weight:     ((qk_nope_head_dim + qk_rope_head_dim) * n_heads, q_lora_rank)
    # KV_proj_down_weight:  (kv_lora_rank + qk_rope_head_dim, dim)
    # KV_proj_up_weight:    ((qk_nope_head_dim + v_head_dim) * n_heads, kv_lora_rank)
    # wo_weight:            (dim, v_head_dim * n_heads)
    # x:                (batchsize, 1, dim)
    with Timer(f"kv length: {kv_cache.seq_len + 1} down projection"):
        q_lora = x @ config.Q_proj_down_weight.T # (batchsize, 1, q_lora_rank)
        cur_kv_lora = x @ config.KV_proj_down_weight.T # (batchsize, 1, kv_lora_rank + qk_rope_head_dim)
    
    with Timer(f"kv length: {kv_cache.seq_len + 1} store kv_lora"):
        # store kv_lora in cache and return the updated cache
        kv_cache_data = kv_cache.get_data() # (batchsize, max_seq_len, kv_lora_rank + qk_rope_head_dim)
        kv_cache.seq_len += 1
        # TODO: use triton to write the kv_lora to the cache
        kv_cache_data[:, kv_cache.seq_len - 1, :] = cur_kv_lora.squeeze(1) 
        kv_len = kv_cache.seq_len # (batchsize, kv_len, kv_lora_rank + qk_rope_head_dim)
        kv_cache_data = kv_cache.get_data() # (batchsize, max_seq_len, kv_lora_rank + qk_rope_head_dim)
        kv_lora = kv_cache_data[:, :kv_cache.seq_len, :] # (batchsize, kv_len, kv_lora_rank + qk_rope_head_dim) : (max_seq_len * (kv_lora_rank + qk_rope_head_dim), kv_lora_rank + qk_rope_head_dim ,1)
        query_pos = kv_len - 1 #
    with Timer(f"kv length: {kv_cache.seq_len} up projection"):
        q_nope_and_rope = q_lora @ config.Q_proj_up_weight.T # (batchsize, 1, (qk_nope_head_dim + qk_rope_head_dim) * n_heads)
        q_nope_and_rope = q_nope_and_rope.view(config.batch_size, 1, config.n_heads, config.qk_nope_head_dim + config.qk_rope_head_dim) # (batchsize, 1, n_heads, qk_nope_head_dim + qk_rope_head_dim)
        
        q_nope, q_rope = torch.split(q_nope_and_rope, [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1) 
        # (batchsize, 1, n_heads, qk_nope_head_dim) : (n_heads * (qk_nope_head_dim + qk_rope_head_dim), n_heads * (qk_nope_head_dim + qk_rope_head_dim), (qk_nope_head_dim + qk_rope_head_dim), 1) 
        # (batchsize, 1, n_heads, qk_rope_head_dim) : (n_heads * (qk_nope_head_dim + qk_rope_head_dim), n_heads * (qk_nope_head_dim + qk_rope_head_dim), (qk_nope_head_dim + qk_rope_head_dim), 1)
        # print(f"q_nope shape: {q_nope.shape} stride: {q_nope.stride()} is_contiguous: {q_nope.is_contiguous()}")
        # print(f"q_rope shape: {q_rope.shape} stride: {q_rope.stride()} is_contiguous: {q_rope.is_contiguous()}")
    # TODO: inplace rope for q_rope
    with Timer(f"kv length: {kv_cache.seq_len} q_rope triton kernel"):
        q_rope_out = torch.empty(size=q_rope.shape, dtype=q_rope.dtype, device=q_rope.device) # (batchsize, 1, n_heads, qk_rope_head_dim)
        head_block_size = 16
        q_rope_cos_sin_kernel[triton.cdiv(config.n_heads, head_block_size), 1, config.batch_size](
            x = q_rope,
            y = q_rope_out,
            cos = cos_tensors[config.qk_rope_head_dim],
            sin = sin_tensors[config.qk_rope_head_dim],
            start_position=query_pos,
            batchsize=config.batch_size,
            seq_len=1,
            n_heads=config.n_heads,
            rope_head_dim=config.qk_rope_head_dim,
            x_batch_stride=q_rope.stride(0),
            x_seq_stride=q_rope.stride(1),
            x_head_stride=q_rope.stride(2),
            y_batch_stride=q_rope_out.stride(0),
            y_seq_stride=q_rope_out.stride(1),
            y_head_stride=q_rope_out.stride(2),
            head_block_size=head_block_size,
        )
    
    KV_proj_up_weight = config.KV_proj_up_weight.view(config.n_heads, config.qk_nope_head_dim + config.v_head_dim, config.kv_lora_rank) # (n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    k_nope_proj_up_weight, v_proj_up_weight = torch.split(KV_proj_up_weight, [config.qk_nope_head_dim, config.v_head_dim], dim=-2) 
    # (n_heads, qk_nope_head_dim, kv_lora_rank) : ((qk_nope_head_dim + v_head_dim) * kv_lora_rank, kv_lora_rank, 1)
    # (n_heads, v_head_dim, kv_lora_rank) : ((qk_nope_head_dim + v_head_dim) * kv_lora_rank, kv_lora_rank, 1)
    # print(f"k_nope_proj_up_weight shape: {k_nope_proj_up_weight.shape} stride: {k_nope_proj_up_weight.stride()} is_contiguous: {k_nope_proj_up_weight.is_contiguous()}")
    # print(f"v_proj_up_weight shape: {v_proj_up_weight.shape} stride: {v_proj_up_weight.stride()} is_contiguous: {v_proj_up_weight.is_contiguous()}")
    # TODO: check if contiguous is called or not
    with Timer(f"kv length: {kv_cache.seq_len} q knope projection"):
        # q_nope_projed = torch.einsum('bshd,hdc->bshc', q_nope, k_nope_proj_up_weight) # (batchsize, 1, n_heads, kv_lora_rank)
        #print(f"q_nope_projed shape: {q_nope_projed.shape} stride: {q_nope_projed.stride()} is_contiguous: {q_nope_projed.is_contiguous()}")
        
        # b1hd -> bhd
        q_nope = q_nope.squeeze(1) # (batchsize, n_heads, qk_nope_head_dim)
        # bhd -> hbd
        q_nope = q_nope.transpose(0, 1) # (n_heads, batchsize, qk_nope_head_dim)
        
        q_nope_projed = torch.bmm(q_nope, k_nope_proj_up_weight) # (n_heads, batchsize, kv_lora_rank)
        q_nope_projed = q_nope_projed.transpose(0, 1) # (batchsize, n_heads, kv_lora_rank)
        q_nope_projed = q_nope_projed.unsqueeze(1) # (batchsize, 1, n_heads, kv_lora_rank)
    
    kv_nope, k_rope = torch.split(kv_lora, [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1) 
    # (batchsize, kv_len, kv_lora_rank) : (max_seq_len * (kv_lora_rank + qk_rope_head_dim), kv_lora_rank + qk_rope_head_dim ,1)
    # (batchsize, kv_len, qk_rope_head_dim) : (max_seq_len * (kv_lora_rank + qk_rope_head_dim), kv_lora_rank + qk_rope_head_dim ,1)
    # print(f"kv_nope shape: {kv_nope.shape} stride: {kv_nope.stride()} is_contiguous: {kv_nope.is_contiguous()}")
    # print(f"k_rope shape: {k_rope.shape} stride: {k_rope.stride()} is_contiguous: {k_rope.is_contiguous()}")
    # TODO: out of place rope for k_rope
    with Timer(f"kv length: {kv_cache.seq_len} k_rope triton kernel"):
        k_rope_out = torch.empty(size=k_rope.shape, dtype=k_rope.dtype, device=k_rope.device) # (batchsize, kv_len, qk_rope_head_dim)
        batch_block_size = 32
        k_rope_cos_sin_kernel[kv_len, triton.cdiv(config.batch_size, batch_block_size)](
            x = k_rope,
            y = k_rope_out,
            cos = cos_tensors[config.qk_rope_head_dim],
            sin = sin_tensors[config.qk_rope_head_dim],
            batchsize=config.batch_size,
            seq_len=kv_len,
            rope_head_dim=config.qk_rope_head_dim,
            x_batch_stride=k_rope.stride(0),
            x_seq_stride=k_rope.stride(1),
            y_batch_stride=k_rope_out.stride(0),
            y_seq_stride=k_rope_out.stride(1),
            batch_block_size=batch_block_size,
            num_warps=8, # TODO: check if this is needed
            num_stages=1, # TODO: check if this is needed
        )
    # k_rope_out = torch_rope(thetas[config.qk_rope_head_dim],k_rope,0)
    
    with Timer(f"kv length: {kv_cache.seq_len} mla decode"):
        # TODO: MLA
        # y = torch.zeros(size=(config.batch_size, config.n_heads, config.kv_lora_rank), dtype=torch.bfloat16, device='cuda') # (batchsize, n_heads, kv_lora_rank)
        # MLA_HEAD_BLOCK_SIZE = 16
        # MLA_SEQ_BLOCK_SIZE = 32
        # mla_decode_kernel[triton.cdiv(config.n_heads, MLA_HEAD_BLOCK_SIZE), config.batch_size](
        #     q_nope=q_nope_projed, 
        #     kv_nope=kv_nope, 
        #     q_rope=q_rope_out, 
        #     k_rope=k_rope_out,
        #     out=y,
        #     scale=1.0 / math.sqrt(config.qk_nope_head_dim + config.qk_rope_head_dim),
        #     batch_size=config.batch_size,
        #     kv_len=kv_len,
        #     q_nope_batch_stride=q_nope_projed.stride(0),
        #     q_nope_head_stride=q_nope_projed.stride(2),
        #     kv_nope_batch_stride=kv_nope.stride(0),
        #     kv_nope_seq_stride=kv_nope.stride(1),
        #     n_heads=config.n_heads,
        #     qk_rope_head_dim=config.qk_rope_head_dim,
        #     kv_lora_rank=config.kv_lora_rank,
        #     HEAD_BLOCK_SIZE=MLA_HEAD_BLOCK_SIZE,
        #     SEQ_BLOCK_SIZE=MLA_SEQ_BLOCK_SIZE,
        #     num_stages=1, #
        #     num_warps=4, #
        #     waves_per_eu = 1, matrix_instr_nonkdim = 16, kpack = 2
        # )
        # y = y.unsqueeze(1) # (batchsize, 1, n_heads, kv_lora_rank)
        # print(f"q_nope_projed is contiguous: {q_nope_projed.is_contiguous()} shape: {q_nope_projed.shape} stride: {q_nope_projed.stride()}")
        # print(f"kv_nope is contiguous: {kv_nope.is_contiguous()} shape: {kv_nope.shape} stride: {kv_nope.stride()}")
        # print(f"q_rope_out is contiguous: {q_rope_out.is_contiguous()}")
        # print(f"k_rope_out is contiguous: {k_rope_out.is_contiguous()}")
        score = torch.einsum('bshc,bvc->bshv', q_nope_projed, kv_nope) # (batchsize, 1, n_heads, kv_len)
        score += torch.einsum('bshd,bvd->bshv', q_rope_out, k_rope_out) # (batchsize, 1, n_heads, kv_len)
        score = score / math.sqrt(config.qk_nope_head_dim + config.qk_rope_head_dim) # (batchsize, 1, n_heads, kv_len)
        attn = F.softmax(score, dim=-1) # (batchsize, 1, n_heads, kv_len)
        y = torch.einsum('bshv,bvc->bshc', attn, kv_nope) # (batchsize, 1, n_heads, kv_lora_rank)
        
    
    with Timer(f"kv length: {kv_cache.seq_len} v projection"):
        y = y.squeeze(1) # (batchsize, n_heads, kv_lora_rank)
        y = y.transpose(0, 1) # (n_heads, batchsize, kv_lora_rank)
        v_proj_up_weight = v_proj_up_weight.transpose(1, 2) # (n_heads, kv_lora_rank, v_head_dim)
        y = torch.bmm(y, v_proj_up_weight) # (n_heads, batchsize, v_head_dim)
        y = y.transpose(0, 1) # (batchsize, n_heads, v_head_dim)
        y = y.reshape(config.batch_size, -1) # (batchsize, n_heads * v_head_dim)
        y = torch.matmul(y, config.wo_weight.T) # (batchsize, dim)
        y = y.view(config.batch_size, 1, config.dim) # (batchsize, 1, dim)
    return y, kv_cache_data


def custom_kernel(data: input_t):
    config, x, kv_cache = data
    out = mla_decode_v0(
        config,
        x,
        kv_cache,
    )
    # torch.cuda.synchronize()
    # time.sleep(0.008)
    return out
    

def generate_input(batchsize, dim, dq, prefill, seed):
    # Sizes derived from: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    
    # Generate weights for linear layers
    Q_proj_down_weight = torch.randn((dq, dim), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dim)
    KV_proj_down_weight = torch.randn((512 + 64, dim), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dim)
    Q_proj_up_weight = torch.randn(((128 + 64) * 128, dq), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(dq)
    KV_proj_up_weight = torch.randn(((128 + 128) * 128, 512), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(512)
    wo_weight = torch.randn((dim, 128 * 128), dtype=torch.bfloat16, generator=gen, device='cuda') / math.sqrt(128 * 128)

    config = Config(
        batch_size=batchsize,
        dim=dim,
        q_lora_rank=dq,
        n_heads=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        seq_len=1,
        max_seq_len=8192,
        kv_cache_shape=(batchsize, 8192, 512 + 64),
        Q_proj_down_weight=Q_proj_down_weight,
        Q_proj_up_weight=Q_proj_up_weight,
        KV_proj_down_weight=KV_proj_down_weight,
        KV_proj_up_weight=KV_proj_up_weight,
        wo_weight=wo_weight,
    )
    x = torch.randn((config.batch_size, 1, config.dim), dtype=torch.bfloat16, generator=gen, device='cuda')
    
    # Pre-fill KV cache
    kv_cache = KVCache((config.batch_size, config.max_seq_len, config.kv_lora_rank + config.qk_rope_head_dim)).to('cuda')
    pre_filled_cache = torch.randn((config.batch_size, prefill, config.kv_lora_rank + config.qk_rope_head_dim), 
                                 dtype=torch.bfloat16, generator=gen, device='cuda')
    kv_cache(pre_filled_cache)

    return config, x, kv_cache

# warm up
config, x, kv_cache = generate_input(
    batchsize=128,
    dim=7168,
    dq=1536,
    prefill=4096,
    seed=42,
)
custom_kernel((config, x, kv_cache))


if __name__ == "__main__":
    # Example usage
    batchsize = 128
    dim = 7168
    dq = 1536
    prefill = 4096
    seed = 9817
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),  # 控制分析步骤
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(10):
            config, x, kv_cache = generate_input(batchsize, dim, dq, prefill, seed)
            custom_kernel((config, x, kv_cache))
    # prof.export_chrome_trace("mla_trace.json")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))


# https://github.com/vllm-project/vllm/blob/58738772410c5e0d60b61db39538a9b313d2d7ad/vllm/attention/ops/rocm_aiter_mla.py#L27
# https://github.com/Zyphra/vllm/blob/0c0fdae84f1da5e45518aafc7b32e8139055adae/vllm/v1/attention/backends/mla/rocm_aiter_mla.py#L189
# https://github.com/ROCm/rocm-blogs/blob/bcea202c9c2a471e24c32f0f65a98cea466c5ca2/blogs/artificial-intelligence/aiter-intergration-s/README.md?plain=1#L466
# https://github.com/ROCm/aiter/issues/200