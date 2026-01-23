import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from task import input_t, output_t
import time

# import mla

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

@torch.compile
def apply_rotary_pos_emb(q, k, cos, sin, seq_len):
    q_embed = (q * cos[seq_len - 1]) + (rotate_half(q) * sin[seq_len - 1])
    k_embed = (k * cos[:seq_len]) + (rotate_half(k) * sin[:seq_len])
    return q_embed, k_embed

@torch.compile
def apply_rotary_pos_emb_q(q, cos, sin, seq_len):
    q_embed = (q * cos[seq_len - 1]) + (rotate_half(q) * sin[seq_len - 1])
    return q_embed

@torch.compile
def apply_rotary_pos_emb_k(k, cos, sin, seq_len):
    k_embed = (k * cos[:seq_len]) + (rotate_half(k) * sin[:seq_len])
    return k_embed

def precompute_rope_cache(seq_len, dim, device='cuda', base=10000.0):
    assert dim % 2 == 0, "Embedding dimension must be even for RoPE."
    position = torch.arange(seq_len, device=device, dtype=torch.bfloat16)
    theta = base ** (-torch.arange(0, dim // 2, device=device, dtype=torch.bfloat16) / (dim // 2))
    angles = torch.outer(position, theta)
    out = torch.cat([angles, angles], dim=-1)
    return out.cos(), out.sin()

rope_cache = precompute_rope_cache(6145, 64)

@torch.compile
def apply_softmax(qk_nope, qk_rope, nope_dim, rope_dim):
    scores = (qk_nope + qk_rope) / math.sqrt(rope_dim + nope_dim)
    return F.softmax(scores, dim=-1)

# import os
# os.putenv('PYTORCH_TUNABLEOP_ENABLED', '1')
# os.putenv('PYTORCH_TUNABLEOP_TUNING', '0')
# os.putenv('PYTORCH_TUNABLEOP_RECORD_UNTUNED', '1')

# torch.cuda.tunable.enable()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start_cpu = 0

def tstart():
    global start_cpu
    start.record()
    start_cpu = time.perf_counter_ns()

def tend():
    end.record()
    torch.cuda.synchronize()
    end_cpu = time.perf_counter_ns()
    print(f"execution time: {start.elapsed_time(end)} ms, cpu: {(end_cpu - start_cpu) / 1e6} ms")

batch_size = 128
seq_len = 1
n_heads = 128
rope_dim = 64
nope_dim = 128
q_lora_dim = 1536
kv_lora_dim = 512
v_dim = 128
hidden_dim = 7168
        
g = torch.cuda.CUDAGraph()
input_static: input_t = None
output_static: output_t = None

def copy_input(input_dyn: input_t):
    global input_static
    config_static, x_static, kv_cache_wrapper_static = input_static
    config, x, kv_cache_wrapper = input_dyn
    prefill = kv_cache_wrapper.seq_len
    kv_cache_static = kv_cache_wrapper_static.get_data()
    kv_cache = kv_cache_wrapper.get_data()
    # copy input
    # kv_cache_static[:, :prefill] = kv_cache[:, :prefill]
    x_static.copy_(x)
    # copy weight
    config_static.Q_proj_down_weight.copy_(config.Q_proj_down_weight)
    config_static.Q_proj_up_weight.copy_(config.Q_proj_up_weight)
    config_static.KV_proj_down_weight.copy_(config.KV_proj_down_weight)
    config_static.KV_proj_up_weight.copy_(config.KV_proj_up_weight)
    # config_static.wo_weight.copy_(config.wo_weight)

def copy_output(output_dyn: output_t):
    global output_static
    o_static, kv_cache_static = output_static
    o, kv_cache = output_dyn
    o_static.copy_(o)


def custom_kernel(data: input_t) -> output_t:
    global output_static, input_static
    _, _, kv_cache_wrapper = data
    prefill = kv_cache_wrapper.seq_len
    if prefill < 6144:
        return custom_kernel_impl(data)
    if not input_static:
        custom_kernel_step_1(data)
        with torch.cuda.graph(g):
            output_static = custom_kernel_step_1(data)
        input_static = data
    # tstart()
    copy_input(data)
    # tend()
    # tstart()
    g.replay()
    # tend()
    return custom_kernel_step_2(data, *output_static)

@torch.compile
def custom_kernel_step_1(data: input_t) -> output_t:
    config, x, kv_cache_wrapper = data
    prefill = kv_cache_wrapper.seq_len
    kv_cache, qdown, qup, kvdown, kvup, wo = kv_cache_wrapper.get_data(), config.Q_proj_down_weight, config.Q_proj_up_weight, config.KV_proj_down_weight, config.KV_proj_up_weight, config.wo_weight
    batch_size, seq_len, n_heads = config.batch_size, config.seq_len, config.n_heads
    assert seq_len == 1
    nope_dim, rope_dim = config.qk_nope_head_dim, config.qk_rope_head_dim
    q_lora_dim, kv_lora_dim = config.q_lora_rank, config.kv_lora_rank
    v_dim = config.v_head_dim
    hidden_dim = config.dim
    # step 1
    # hidden -> kv_lora[-1] (kvdown_nope)
    # hidden -> k_rope[-1] (kvdown_rope)
    # kv_cache -> kv_lora, k_rope
    # hidden -> q_lora (qdown)
    kv_cache: torch.Tensor = kv_cache[:, :prefill + seq_len]
    kv_cache_last, q_lora = torch.einsum("b s d, l d -> b s l", x, torch.concat((kvdown, qdown), dim=0)).split([kv_lora_dim + rope_dim, q_lora_dim], dim=-1)
    # step 2
    # q_lora -> q_rope (qup_rope)
    # q_lora -> q_nope @ qup_nope @ kup_nope
    qup = qup.view(n_heads, nope_dim + rope_dim, q_lora_dim)
    kup_nope, vup = kvup.view(n_heads, nope_dim + v_dim, kv_lora_dim).split([nope_dim, v_dim], dim=-2)
    q_nope, q_rope = torch.einsum("b s l, h d l -> b s h d", q_lora, qup).split([nope_dim, rope_dim], dim=-1)
    q_absorb = torch.einsum("b s h d, h d l-> b s h l", q_nope, kup_nope)
    q_rope = q_rope.permute(0, 2, 1, 3)
    q_absorb = q_absorb.permute(0, 2, 1, 3)
    # step 2.5 apply rope
    cos, sin = rope_cache
    # 0.12 ms
    q_rope = apply_rotary_pos_emb_q(q_rope, cos, sin, prefill + seq_len)
    return q_absorb, kv_cache_last, q_rope

@torch.compile
def custom_kernel_step_2(data: input_t, q_absorb, kv_cache_last, q_rope) -> output_t:
    config, x, kv_cache_wrapper = data
    prefill = kv_cache_wrapper.seq_len
    kv_cache, qdown, qup, kvdown, kvup, wo = kv_cache_wrapper.get_data(), config.Q_proj_down_weight, config.Q_proj_up_weight, config.KV_proj_down_weight, config.KV_proj_up_weight, config.wo_weight
    kv_cache: torch.Tensor = kv_cache[:, :prefill + seq_len]

    kv_cache[:, -1:, :] = kv_cache_last
    kv_lora, k_rope = kv_cache.split([kv_lora_dim, rope_dim], dim=-1)

    cos, sin = rope_cache
    k_rope = apply_rotary_pos_emb_k(k_rope, cos, sin, prefill + seq_len)

    # step 3
    # q_nope @ kv_lora -> attn_nope
    # q_rope @ kv_rope -> attn_rope
    # softmax(attn_nope + attn_rope) -> attn
    attn_nope = torch.einsum("b h s l, b p l -> b h s p", q_absorb, kv_lora)
    attn_rope = torch.einsum("b h s d, b p d -> b h s p", q_rope, k_rope)
    # 0.3 ms
    attention = apply_softmax(attn_nope, attn_rope, nope_dim, rope_dim)
    # step 4
    # attn @ kv_lora @ vup @ wo-> output
    o = torch.einsum("b h s p, b p l -> b h s l", attention, kv_lora)
    qup = qup.view(n_heads, nope_dim + rope_dim, q_lora_dim)
    kup_nope, vup = kvup.view(n_heads, nope_dim + v_dim, kv_lora_dim).split([nope_dim, v_dim], dim=-2)
    o = torch.einsum("b h s l, h v l -> b h s v", o, vup)
    wo = wo.view(hidden_dim, n_heads, v_dim)
    output = torch.einsum("b h s v, d h v -> b s d", o, wo)
    # mla.mla_decode(x, kv_cache.get_data(), config.Q_proj_down_weight, config.Q_proj_up_weight, config.KV_proj_down_weight, config.KV_proj_up_weight, config.wo_weight, output)
    return output, kv_cache_wrapper.get_data()

# @torch.compile
def custom_kernel_impl(data: input_t) -> output_t:
    config, x, kv_cache_wrapper = data
    prefill = kv_cache_wrapper.seq_len
    kv_cache, qdown, qup, kvdown, kvup, wo = kv_cache_wrapper.get_data(), config.Q_proj_down_weight, config.Q_proj_up_weight, config.KV_proj_down_weight, config.KV_proj_up_weight, config.wo_weight
    batch_size, seq_len, n_heads = config.batch_size, config.seq_len, config.n_heads
    assert seq_len == 1
    nope_dim, rope_dim = config.qk_nope_head_dim, config.qk_rope_head_dim
    q_lora_dim, kv_lora_dim = config.q_lora_rank, config.kv_lora_rank
    v_dim = config.v_head_dim
    hidden_dim = config.dim
    # step 1
    # hidden -> kv_lora[-1] (kvdown_nope)
    # hidden -> k_rope[-1] (kvdown_rope)
    # kv_cache -> kv_lora, k_rope
    # hidden -> q_lora (qdown)
    kv_cache: torch.Tensor = kv_cache[:, :prefill + seq_len]
    kv_cache[:, -1:, :], q_lora = torch.einsum("b s d, l d -> b s l", x, torch.concat((kvdown, qdown), dim=0)).split([kv_lora_dim + rope_dim, q_lora_dim], dim=-1)
    kv_lora, k_rope = kv_cache.split([kv_lora_dim, rope_dim], dim=-1)
    # step 2
    # q_lora -> q_rope (qup_rope)
    # q_lora -> q_nope @ qup_nope @ kup_nope
    qup = qup.view(n_heads, nope_dim + rope_dim, q_lora_dim)
    kup_nope, vup = kvup.view(n_heads, nope_dim + v_dim, kv_lora_dim).split([nope_dim, v_dim], dim=-2)
    q_nope, q_rope = torch.einsum("b s l, h d l -> b s h d", q_lora, qup).split([nope_dim, rope_dim], dim=-1)
    q_absorb = torch.einsum("b s h d, h d l-> b s h l", q_nope, kup_nope)
    q_rope = q_rope.permute(0, 2, 1, 3)
    q_absorb = q_absorb.permute(0, 2, 1, 3)
    # step 2.5 apply rope
    cos, sin = rope_cache
    # 0.12 ms
    q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, prefill + seq_len)
    # print(f"{q_rope[0, 0, 0, -1]=}")
    # print(f"{q_nope[0, 0, 0, -1]=}")
    # print(f"{k_rope[0, 1, 0, -1]=}")
    # print(f"{k_nope[0, 0, 1, -1]=}")

    # step 3
    # q_nope @ kv_lora -> attn_nope
    # q_rope @ kv_rope -> attn_rope
    # softmax(attn_nope + attn_rope) -> attn
    attn_nope = torch.einsum("b h s l, b p l -> b h s p", q_absorb, kv_lora)
    attn_rope = torch.einsum("b h s d, b p d -> b h s p", q_rope, k_rope)
    # 0.3 ms
    attention = apply_softmax(attn_nope, attn_rope, nope_dim, rope_dim)
    # step 4
    # attn @ kv_lora @ vup @ wo-> output
    o = torch.einsum("b h s p, b p l -> b h s l", attention, kv_lora)
    o = torch.einsum("b h s l, h v l -> b h s v", o, vup)
    wo = wo.view(hidden_dim, n_heads, v_dim)
    output = torch.einsum("b h s v, d h v -> b s d", o, wo)
    # mla.mla_decode(x, kv_cache.get_data(), config.Q_proj_down_weight, config.Q_proj_up_weight, config.KV_proj_down_weight, config.KV_proj_up_weight, config.wo_weight, output)
    return output, kv_cache_wrapper.get_data()
