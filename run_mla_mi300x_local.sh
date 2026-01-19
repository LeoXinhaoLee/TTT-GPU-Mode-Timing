#!/bin/bash

rm -rf ~/.triton  # rm cache
rm -rf ~/.cache/torch_extensions  # rm cache

cd gpu_mode
export CUSTOM_KERNEL_MODULE="mla-decode-local/MI300X_kernels/TTT_MLA.py"
python mla-decode-local/eval_local.py mla-decode-local/MI300X_kernels/TTT_MLA.out