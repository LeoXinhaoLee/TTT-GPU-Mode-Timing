#!/bin/bash

rm -rf ~/.triton  # rm cache
rm -rf ~/.cache/torch_extensions  # rm cache

cd gpu_mode
export CUSTOM_KERNEL_MODULE="bioml/trimul/MI300X_kernels/TTT_MI300X.py"
python bioml/trimul/eval_local.py bioml/trimul/MI300X_kernels/TTT_MI300X.out

