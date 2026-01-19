#!/bin/bash

# rm -rf ~/.triton  # rm cache
# rm -rf ~/.cache/torch_extensions  # rm cache

# cd gpu_mode
# export CUSTOM_KERNEL_MODULE="mla-decode-local/MI300X_kernels/TTT_MLA.py"
# python mla-decode-local/eval_local.py mla-decode-local/MI300X_kernels/TTT_MLA.out

set -x

cd gpu_mode

folder="0118_oss_h200_top_20_all"
for step_range in "45:49"; do
    
    for round in 1 2 3; do
        
        mkdir -p mla-decode-local/${folder}/step_${step_range}/out/round_${round}
        for kid in {0..19}; do
            rm -rf ~/.triton  # rm cache
            rm -rf ~/.cache/torch_extensions  # rm cache
            export CUSTOM_KERNEL_MODULE="mla-decode-local/${folder}/step_${step_range}/kernels/sample_${kid}.py"
            timeout 100 python mla-decode-local/eval_local.py mla-decode-local/${folder}/step_${step_range}/out/round_${round}/sample_${kid}.out
            sleep 10
        done
    
    done

done
