#!/bin/bash

rm -rf ~/.triton  # rm cache
rm -rf ~/.cache/torch_extensions  # rm cache

cd gpu_mode
# export CUSTOM_KERNEL_MODULE="mla-decode-local/MI300X_kernels/TTT_MLA.py"
# python mla-decode-local/eval_local.py mla-decode-local/MI300X_kernels/TTT_MLA.out

for i in 1 2 3; do
# export CUSTOM_KERNEL_MODULE="mla-decode-local/MI300X_kernels/human_1st.py"
# python mla-decode-local/eval_local.py mla-decode-local/MI300X_kernels/human_1st_${i}.out
# export CUSTOM_KERNEL_MODULE="mla-decode-local/MI300X_kernels/human_2nd.py"
# python mla-decode-local/eval_local.py mla-decode-local/MI300X_kernels/human_2nd_${i}.out
# export CUSTOM_KERNEL_MODULE="mla-decode-local/MI300X_kernels/human_3rd.py"
# python mla-decode-local/eval_local.py mla-decode-local/MI300X_kernels/human_3rd_${i}.out
sleep 10
done