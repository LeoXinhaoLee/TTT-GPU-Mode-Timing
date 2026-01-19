#!/bin/bash

rm -rf ~/.triton  # rm cache
rm -rf ~/.cache/torch_extensions  # rm cache

cd gpu_mode
export CUSTOM_KERNEL_MODULE="bioml/trimul/MI300X_kernels/TTT_MI300X.py"
python bioml/trimul/eval_local.py bioml/trimul/MI300X_kernels/TTT_MI300X.out

# export CUSTOM_KERNEL_MODULE="mla-decode-local/MI300X_kernels/human_1st.py"
# python mla-decode-local/eval_local.py mla-decode-local/MI300X_kernels/human_1st.out
