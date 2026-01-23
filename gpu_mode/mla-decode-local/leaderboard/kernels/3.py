import math
from dataclasses import dataclass
import torch
from torch import nn
import numpy as np
import time
import torch.nn.functional as F
from utils import verbose_allclose
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import os
import sys

     
if "PYTORCH_ROCM_ARCH" not in os.environ:
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942:xnack-"

kernel_cpp = r"""
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <fstream>
#include <unistd.h>  
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <torch/torch.h>
#define CHECK_HIPBLASLT(err) if (err != HIPBLAS_STATUS_SUCCESS) { @
  std::cerr << "hipBLASLt error at line " << __LINE__ <<  std::endl; exit(1); }
  
#define WARP_SIZE 64
using bf16 = hip_bfloat16;

int divUp(int x,int n){
return (x + n-1)/n;
}

template<int n_split>
__global__ void rope_kernel(bf16* rope_out, const bf16* rope, int start_pos, int seq_len, int batch_size, int stride_bs, int stride_seq) {
  constexpr int d_model = 64;//Must be equal to WARP_SIZE
  constexpr int d_model_2 = d_model/2;
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  int pos = tid % d_model;
  int seq = (tid / d_model) % seq_len;
  int bs = tid / d_model / seq_len;
  
  if (bs < batch_size){
    auto pos_2 = bf16(tid % d_model_2);
    auto theta = bf16(powf(bf16(10000.0f),bf16(-pos_2)/bf16(d_model_2)));
    auto idx_theta2 = bf16(start_pos+seq)*theta;
    auto c = bf16(cosf(idx_theta2));
    auto s = bf16(sinf(idx_theta2));

    bf16 x_r[n_split];
    #pragma unroll
    for (int split=0;split<n_split;split++){
      int index = ((batch_size/n_split)*split + bs)*stride_bs+
                  seq*stride_seq+
                  pos;
       x_r[split] = rope[index];
    }

    #pragma unroll
    for (int split=0;split<n_split;split++){
      //TODO:add parameters for output strides
      int index = ((batch_size/n_split)*split + bs)*d_model*seq_len+
                  seq*d_model+
                  pos;
         
      auto x = x_r[split];

      int src_lane = (pos +d_model_2)% d_model;
      auto x_rot = __shfl(x, src_lane, 64);
      x_rot = (pos < d_model_2) ? -x_rot : x_rot;

      auto out  = bf16(x*c)+bf16(x_rot*s);
      rope_out[index] = out;

    }
  }
}

void compute_rope(bf16* rope_out, const bf16* rope, int start_pos, int seq_len, int batch_size,int stride_bs, int stride_seq) {
  constexpr int dim_rope = 64;
  constexpr int n_split = 32;
  dim3 threadsPerBlock(WARP_SIZE*4);
  dim3 blocksPerGrid(divUp(seq_len*dim_rope*batch_size/n_split, threadsPerBlock.x));
  hipLaunchKernelGGL((rope_kernel<n_split>), blocksPerGrid, threadsPerBlock, 0, 0, rope_out,rope, start_pos,seq_len,batch_size,stride_bs,stride_seq);
}

void compute_ropes_wrapper(
  uintptr_t k_rope_out_ptr,
  uintptr_t k_rope_ptr,
  const std::vector<int>& k_rope_stride,
  uintptr_t q_rope_out_ptr,
  uintptr_t q_rope_ptr,
  int batch_size,
  int n_heads,
  int seq_len){

  bf16 *d_k_rope_out = reinterpret_cast<bf16*>(k_rope_out_ptr);
  bf16 *d_q_rope_out = reinterpret_cast<bf16*>(q_rope_out_ptr);
  const bf16 *d_k_rope = reinterpret_cast<const bf16*>(k_rope_ptr);
  const bf16 *d_q_rope = reinterpret_cast<const bf16*>(q_rope_ptr);

  int stride_bs_k = k_rope_stride[0];
  int stride_seq_k = k_rope_stride[1];
  
  int stride_bs_q = 64;
  int stride_seq_q = 64;

  compute_rope(d_k_rope_out,d_k_rope, 0, seq_len, batch_size,stride_bs_k,stride_seq_k);
  compute_rope(d_q_rope_out,d_q_rope, seq_len-1, 1, batch_size*n_heads,stride_bs_q,stride_seq_q);
  
}

__global__ void store_to_kv_cache_kernel(bf16* kv_cache, const bf16 *kv, int batch_size, int seq_id) {
  constexpr int dim_kv = 576;
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int bs = tid/dim_kv;
  int col = tid % dim_kv;
  int dst_row = bs*8192 + seq_id;
  int dstIndex = dst_row*dim_kv+col;
  if (bs<batch_size && dst_row<batch_size*8192){
    kv_cache[dstIndex] = kv[bs*dim_kv+col];
  }
}

void store_to_kv_cache(bf16* kv_cache, const bf16 *kv, int batch_size, int seq_id)  {
    constexpr int dim_kv = 576;
    dim3 threadsPerBlock(WARP_SIZE*4);
    dim3 blocksPerGrid(divUp(batch_size*dim_kv, threadsPerBlock.x));
    hipLaunchKernelGGL((store_to_kv_cache_kernel), blocksPerGrid, threadsPerBlock, 0, 0, kv_cache,kv, batch_size, seq_id);
}


void store_to_kv_cache_wrapper(
  uintptr_t kv_lora_ptr,
  uintptr_t kv_cache_ptr,

  int batch_size,
  int seq_id){

  const bf16 *d_kv_lora = reinterpret_cast<const bf16*>(kv_lora_ptr);
  bf16 *d_kv_cache = reinterpret_cast<bf16*>(kv_cache_ptr);

  store_to_kv_cache(d_kv_cache, d_kv_lora, batch_size, seq_id);
  
}

static rocblas_handle handle;
static bool hasInit = false;
static size_t workspaceSize = (4 << 20)*4; 

    // Create hipBLASLt handle
static hipblasLtHandle_t handle_hipblas;
    
struct ConfigBlasLt{
  hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
  void* d_workspace;
  hipblasLtMatmulHeuristicResult_t heuristicResult;
  hipblasLtMatmulDesc_t matmulDesc;
};

ConfigBlasLt initBlasLt(int M, int N, int K, int batchCount, int strideMK, int strideKN){

  long long strideA = strideMK;
  long long strideB = strideKN;
  long long strideC = M * N;
  float alpha = 1.0f, beta = 0.0f;
 
  hipblasLtMatmulDesc_t matmulDesc;
  CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F,  HIP_R_32F));
  
  hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16BF, K, M, K));
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16BF, N, K, N));
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16BF, N, M, N));

  CHECK_HIPBLASLT(hipblasLtMatrixLayoutSetAttribute(layoutA,
      HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutSetAttribute(layoutA,
      HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));

  CHECK_HIPBLASLT(hipblasLtMatrixLayoutSetAttribute(layoutB,
      HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutSetAttribute(layoutB,
      HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));

  CHECK_HIPBLASLT(hipblasLtMatrixLayoutSetAttribute(layoutC,
      HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutSetAttribute(layoutC,
      HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));

  hipblasLtMatmulPreference_t preference;
  CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&preference));

  void* d_workspace;
  hipMalloc(&d_workspace, workspaceSize);
  CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(preference,
      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  hipblasLtMatmulHeuristicResult_t heuristicResult;
  int returnedAlgoCount = 0;
  CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(handle_hipblas, matmulDesc,
      layoutB, layoutA, layoutC, layoutC, preference, 1, &heuristicResult, &returnedAlgoCount));

  if (returnedAlgoCount == 0) {
      std::cerr << "No suitable algorithm found." << std::endl;
      throw new std::runtime_error("No suitable algorithm found.");
  }

  return{
    layoutA, layoutB, layoutC,
    d_workspace,
    heuristicResult,
    matmulDesc
  };
 
}

static ConfigBlasLt config_lora;

void init(){
    if (!hasInit){
        rocblas_create_handle(&handle);        
        
        CHECK_HIPBLASLT(hipblasLtCreate(&handle_hipblas));
  
        int batchCount = 128;
        int M = 128;
        int N = 512;
        int K = 1536;
        int strideMK = 0;
        int strideKN = K*N;
        config_lora = initBlasLt( M,  N,  K,  batchCount, strideMK, strideKN);
    
        hasInit = true;
    }
}

void matmul_blas_rm_cm(const bf16*dA, const bf16 *dB, bf16 *dC, int M, int N, int K, int strideA, float alpha = 1.0f){
  const float beta  = 0.0f;
  rocblas_gemm_ex(
    handle,
    rocblas_operation_transpose,
    rocblas_operation_none,
    N, M, K,
    &alpha,
    dB, rocblas_datatype_bf16_r, K,
    dA, rocblas_datatype_bf16_r, strideA,
    &beta,
    dC, rocblas_datatype_bf16_r, N,
    dC, rocblas_datatype_bf16_r, N,
    rocblas_datatype_f32_r,
    rocblas_gemm_algo_standard,
    0, 0
  );
}

void down_proj(
  uintptr_t x_ptr,
  uintptr_t q_proj_down_ptr,
  uintptr_t q_lora_ptr,
  uintptr_t kv_proj_down_ptr,
  uintptr_t kv_lora_ptr,
  int batch_size,
  int dim_q_down,
  int dim_kv_down,
  int dim_embed){

  const bf16 *d_input = reinterpret_cast<const bf16*>(x_ptr);
  const bf16 *d_q_proj_down = reinterpret_cast<const bf16*>(q_proj_down_ptr);
  const bf16 *d_kv_proj_down = reinterpret_cast<const bf16*>(kv_proj_down_ptr);
  
  bf16 *d_q_lora = reinterpret_cast<bf16*>(q_lora_ptr);
  bf16 *d_kv_lora = reinterpret_cast<bf16*>(kv_lora_ptr);

   matmul_blas_rm_cm(d_input, d_q_proj_down, d_q_lora, batch_size, dim_q_down, dim_embed,dim_embed);
   matmul_blas_rm_cm(d_input, d_kv_proj_down, d_kv_lora, batch_size, dim_kv_down, dim_embed,dim_embed);
  
}


void matmul_blas_cm_rm_batched_stride(const bf16*dA, const bf16 *dB, bf16 *dC, int M, int N, int K, int n_batches, int strideMK, int strideKN){
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  rocblas_gemm_strided_batched_ex(
    handle,
    rocblas_operation_none,
    rocblas_operation_transpose,
    N, M, K,
    &alpha,
    dB, rocblas_datatype_bf16_r, N,strideKN,
    dA, rocblas_datatype_bf16_r, M,strideMK,
    &beta,
    dC, rocblas_datatype_bf16_r, N,N*M,
    dC, rocblas_datatype_bf16_r, N,N*M,
    n_batches,
    rocblas_datatype_f32_r,
    rocblas_gemm_algo_standard,
    0, 0
  );
}

void merged_matmul(
  uintptr_t Q_proj_up_ptr,
  uintptr_t KV_proj_up_ptr,
  uintptr_t merged_ptr,
  int batch_size,
  int q_lora_dim,
  int kv_lora_dim){
  
  const bf16 *d_Wq_Up_nope = reinterpret_cast<const bf16*>(Q_proj_up_ptr);
  const bf16 *d_Wk_up = reinterpret_cast<const bf16*>(KV_proj_up_ptr);
  bf16 *d_merged = reinterpret_cast<bf16*>(merged_ptr);
  
  const int nb_heads = 128;
  const int M = q_lora_dim;
  const int N = kv_lora_dim;
  const int K = 128;
  const int strideMK = 192*q_lora_dim;//128+64
  const int strideKN = 256*kv_lora_dim; //128+128
    
  matmul_blas_cm_rm_batched_stride(d_Wq_Up_nope, d_Wk_up, d_merged, M,  N,  K, batch_size,strideMK, strideKN);
  
}

void matmul_blas_rm_cm_batched_stride(const bf16*dA, const bf16 *dB, const bf16 *dC, bf16 *dD, int M, int N, int K, int n_batches, int strideMK, int strideKN, float alpha, float beta){

  rocblas_gemm_strided_batched_ex(
    handle,
    rocblas_operation_transpose,
    rocblas_operation_none,
    N, M, K,
    &alpha,
    dB, rocblas_datatype_bf16_r, K,strideKN,
    dA, rocblas_datatype_bf16_r, K,strideMK,
    &beta,
    dC, rocblas_datatype_bf16_r, N,N*M,
    dD, rocblas_datatype_bf16_r, N,N*M,
    n_batches,
    rocblas_datatype_f32_r,
    rocblas_gemm_algo_standard,
    0, 0
  );
}

void combine_rope(
  uintptr_t ScoreNoRope_ptr,
  uintptr_t q_rope_out_ptr,
  uintptr_t k_rope_out_ptr,
  uintptr_t ScoreOut_ptr,
  int M,
  int N,
  int K,
  float scale){
  
  const bf16 *d_q_rope_out = reinterpret_cast<const bf16*>(q_rope_out_ptr);
  const bf16 *d_k_rope_out = reinterpret_cast<const bf16*>(k_rope_out_ptr);
  const bf16 *d_ScoreNoRope = reinterpret_cast<const bf16*>(ScoreNoRope_ptr);
  bf16 *  d_ScoreOut = reinterpret_cast<bf16*>(ScoreOut_ptr);
  
  const int nb_heads = 128;
  int strideMK = K*M;
  int strideKN = K*N;

  matmul_blas_rm_cm_batched_stride(d_q_rope_out, d_k_rope_out, d_ScoreNoRope, d_ScoreOut, M,  N,  K, nb_heads,strideMK, strideKN,scale,scale);
}

void matmul_blas_rm_rm_batched_stride(const bf16*dA, const bf16 *dB, bf16 *dC, int M, int N, int K, int n_batches, int strideMK, int strideKN){
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  
  rocblas_gemm_strided_batched_ex(
    handle,
    rocblas_operation_none,
    rocblas_operation_none,
    N, M, K,
    &alpha,
    dB, rocblas_datatype_bf16_r, N,strideKN,
    dA, rocblas_datatype_bf16_r, K,strideMK,
    &beta,
    dC, rocblas_datatype_bf16_r, N,N*M,
    dC, rocblas_datatype_bf16_r, N,N*M,
    n_batches,
    rocblas_datatype_f32_r,
    rocblas_gemm_algo_solution_index,
    0, 0
  );
  
}


void matmul_hipblaslt_rm_rm_batched_stride(const ConfigBlasLt & config , const bf16*d_A, const bf16 *d_B, bf16 *d_C ){

  float alpha = 1.0f, beta = 0.0f;
  // Run matmul
  CHECK_HIPBLASLT(hipblasLtMatmul(handle,
      config.matmulDesc,
      &alpha, d_B, config.layoutB,
              d_A, config.layoutA,
      &beta,  d_C, config.layoutC,
              d_C, config.layoutC,
      &config.heuristicResult.algo,
      config.d_workspace, workspaceSize, 0));
    
}
void matmul_qlora(
  uintptr_t q_lora_ptr,
  uintptr_t W_merged_ptr,
  uintptr_t ScoreNoRope_ptr,
  int M,
  int N,
  int K){
  
  const bf16 *d_q_lora = reinterpret_cast<const bf16*>(q_lora_ptr);
  const bf16 *d_W_merged = reinterpret_cast<const bf16*>(W_merged_ptr);
  bf16 *d_ScoreNoRope1 = reinterpret_cast< bf16*>(ScoreNoRope_ptr);
  
  const int nb_heads = 128;
  const int strideMK = 0;
  const int strideKN = K*N; 
   
  matmul_hipblaslt_rm_rm_batched_stride(config_lora,d_q_lora, d_W_merged, d_ScoreNoRope1);
  
}

torch::Tensor matmul_cpp(torch::Tensor a, torch::Tensor b) {
    return at::bmm(a, b);
}




PYBIND11_MODULE(rope, m) {
  m.def("init", &init, "Init code");
  m.def("rope", &compute_ropes_wrapper, "Rope kernel");
  m.def("storekv", &store_to_kv_cache_wrapper, "KVCache kernel");
  m.def("down_proj", &down_proj, "Down Proj kernel");
  m.def("merged_matmul", &merged_matmul, "merged_matmul kernel");
  m.def("combine_rope", &combine_rope, "combine_rope kernel");
  m.def("matmul_qlora", &matmul_qlora, "matmul_qlora kernel");
  m.def("matmul_cpp", &matmul_cpp, "matmul_test kernel");
  
  
}
"""

        
hip_module = load_inline(
    name="rope",
    cpp_sources="",
    cuda_sources=kernel_cpp.replace('@', chr(92)),
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-std=c++17 -O3"],
    no_implicit_headers=True,
)


class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple) -> None:
        super().__init__()
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device='cuda'))
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

     
class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        self.q_lora = torch.empty(config.batch_size, 1, self.q_lora_rank,dtype=torch.bfloat16, device='cuda')
        self.kv_lora = torch.empty(config.batch_size, 1, self.kv_lora_rank+64,dtype=torch.bfloat16, device='cuda')
        self.ScoreNoRope1 = torch.empty(config.batch_size, 128, 512,dtype=torch.bfloat16, device='cuda')
        self.W_merged = torch.empty(config.batch_size, self.q_lora_rank, self.kv_lora_rank,dtype=torch.bfloat16, device='cuda')
        self.eps = 1e-6
        self.config = config
        hip_module.init()
        
 
        
   
    def forward(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:

        batch_size, seq_len, model_dim = x.size()
          
        hip_module.down_proj(
                x.data_ptr(),
                self.config.Q_proj_down_weight.data_ptr(),
                self.q_lora.data_ptr(),
                self.config.KV_proj_down_weight.data_ptr(),
                self.kv_lora.data_ptr(),
                batch_size,
                self.q_lora_rank,
                self.kv_lora_rank+64,
                x.shape[2])
               
        hip_module.merged_matmul(
          self.config.Q_proj_up_weight.data_ptr(),
          self.config.KV_proj_up_weight.data_ptr(),
          self.W_merged.data_ptr(),
          batch_size,
          self.q_lora_rank,
          self.kv_lora_rank)
  
        hip_module.storekv(
            self.kv_lora.data_ptr(),
            kv_cache.get_data().data_ptr(),
            batch_size,
            kv_cache.seq_len)
        
        
        hip_module.matmul_qlora(
          self.q_lora.data_ptr(),
          self.W_merged.data_ptr(),
          self.ScoreNoRope1.data_ptr(),
          128,
          512,
          1536)
          
        
        kv_cache.seq_len+=1
        kv_len = kv_cache.seq_len
        
        query_pos = kv_len - 1
        kv_lora = torch.as_strided(kv_cache.data, size=(128, kv_len, 576), stride=kv_cache.data.stride())
        
        kvDown_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)              
        
        kv_down_permute = kvDown_nope.permute(0,2,1)
        
        ScoreNoRope1 = self.ScoreNoRope1.permute(1,0,2)
        ScoreNoRope_t=ScoreNoRope1.view(batch_size,self.n_heads,512) #[bs,n_head,512])

        ScoreNoRope = hip_module.matmul_cpp(ScoreNoRope_t,kv_down_permute)
        
        WqNopeRope = self.config.Q_proj_up_weight.t().view(self.q_lora_rank,self.n_heads,self.nope_head_dim+self.rope_head_dim)
        _, Wq_Up_rope = torch.split(WqNopeRope, [self.nope_head_dim, self.rope_head_dim], dim=-1)

        Wkv_up = self.config.KV_proj_up_weight.t().view(self.kv_lora_rank,self.n_heads,self.nope_head_dim+self.v_head_dim)
        _, Wv_up = torch.split(Wkv_up, [self.nope_head_dim, self.v_head_dim], dim=-1)
        
        Wv_up = Wv_up.permute(1, 0, 2)  # shape: (n_head, dim, head)        
        Wq_Up_rope0 = Wq_Up_rope.view(self.q_lora_rank,self.n_heads,self.rope_head_dim)
        Wq_Up_rope0 = Wq_Up_rope0.permute(1,0,2)
        QRope0 = torch.matmul(self.q_lora.permute(1,0,2),Wq_Up_rope0)#1

        
        QRope = QRope0.permute(1,0,2).view(128, 1, 128, 64).contiguous()
        q_rope = QRope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
 
        k_rope_out = torch.empty(batch_size, kv_len, 64,dtype=torch.bfloat16, device='cuda')
        q_rope_out = torch.empty(batch_size, self.n_heads,1, 64,dtype=torch.bfloat16, device='cuda')
        hip_module.rope(k_rope_out.data_ptr(),
                        k_rope.data_ptr(),
                        k_rope.stride(),
                        q_rope_out.data_ptr(),
                        q_rope.data_ptr(),
                        batch_size,
                        self.n_heads,
                        kv_len)
        
        k_rope_out = k_rope_out.unsqueeze(1)

        ScoreOut = torch.empty(batch_size, 128, kv_len,dtype=torch.bfloat16, device='cuda')
        hip_module.combine_rope(
          ScoreNoRope.data_ptr(),
          q_rope_out.data_ptr(),
          k_rope_out.data_ptr(),
          ScoreOut.data_ptr(),
          ScoreOut.shape[1],
          kv_len,
          self.rope_head_dim,
          1.0 / math.sqrt(self.rope_head_dim + self.nope_head_dim))
        
        AttnOutSMax = F.softmax(ScoreOut, dim=-1)#.to(torch.bfloat16)
        
        kv_down = kvDown_nope.unsqueeze(1)
        y0_c = torch.matmul(AttnOutSMax.squeeze(2),kv_down.squeeze(1))
        y0_c=y0_c.unsqueeze(2)
        
        y0_c = torch.matmul(y0_c.permute(1,0,2,3).squeeze(2),Wv_up)
        y0_c = y0_c.permute(1,0,2)
        y0_c = torch.matmul(y0_c.flatten(start_dim=1),self.config.wo_weight.t())
        
        return y0_c.view(batch_size,1,self.dim), kv_cache.get_data()

model = None
def custom_kernel(data: input_t) -> output_t:
      config, x, kv_cache = data
      global model
      if model == None:
          model = MLA(config).to('cuda:0')
   
      model.config = config
      return model(x, kv_cache)