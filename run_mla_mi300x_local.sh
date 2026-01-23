#!/bin/bash

# rm -rf ~/.triton  # rm cache
# rm -rf ~/.cache/torch_extensions  # rm cache

# cd gpu_mode
# export CUSTOM_KERNEL_MODULE="mla-decode-local/MI300X_kernels/TTT_MLA.py"
# python mla-decode-local/eval_local.py mla-decode-local/MI300X_kernels/TTT_MLA.out

set -x

cd gpu_mode

# folder="0118_oss_h200_top_20_all"
# for step_range in "45:49"; do
    
#     for round in 1 2 3; do
        
#         echo "Running round ${round} for step range ${step_range}"
#         mkdir -p mla-decode-local/${folder}/step_${step_range}/out/round_${round}
#         for kid in {0..19}; do
#             rm -rf ~/.triton  # rm cache
#             rm -rf ~/.cache/torch_extensions  # rm cache
#             export CUSTOM_KERNEL_MODULE="mla-decode-local/${folder}/step_${step_range}/kernels/sample_${kid}.py"
#             timeout 100 python mla-decode-local/eval_local.py mla-decode-local/${folder}/step_${step_range}/out/round_${round}/sample_${kid}.out
#             sleep 10
#         done
    
#     done

# done

# folder="TTT_best_candidates"
# for round in 1 2 3 4 5; do
    
#     echo "Running round ${round}"
#     mkdir -p mla-decode-local/${folder}/out/round_${round}
#     for kernel_file in mla-decode-local/${folder}/kernels/*.py; do
#         if [ ! -f "$kernel_file" ]; then
#             continue
#         fi
#         kernel_basename=$(basename "$kernel_file" .py)
#         rm -rf ~/.triton  # rm cache
#         rm -rf ~/.cache/torch_extensions  # rm cache
#         export CUSTOM_KERNEL_MODULE="$kernel_file"
#         timeout 100 python mla-decode-local/eval_local.py mla-decode-local/${folder}/out/round_${round}/${kernel_basename}.out
#         sleep 15
#     done

# done

# device="1"

# for round in {1..10}; do

#     folder="leaderboard"
#     mkdir -p mla-decode-local/${folder}/out_device_${device}/round_${round}
#     for kid in {1..5}; do
#         rm -rf ~/.triton
#         rm -rf /root/.cache/torch_extensions
#         export CUSTOM_KERNEL_MODULE="mla-decode-local/${folder}/kernels/${kid}.py"
#         timeout 200 python mla-decode-local/eval_local.py mla-decode-local/${folder}/out_device_${device}/round_${round}/sample_${kid}.out
#         sleep 15
#     done

#     folder="0118_oss_h200_top_20_all"
#     for step_range in "45:49" "40:44" "35:39" "30:34"; do
#         mkdir -p mla-decode-local/${folder}/step_${step_range}/out_device_${device}/round_${round}
#         for kid in {0..19}; do
#             rm -rf ~/.triton
#             rm -rf /root/.cache/torch_extensions
#             export CUSTOM_KERNEL_MODULE="mla-decode-local/${folder}/step_${step_range}/kernels/sample_${kid}.py"
#             timeout 120 python mla-decode-local/eval_local.py mla-decode-local/${folder}/step_${step_range}/out_device_${device}/round_${round}/sample_${kid}.out
#             sleep 15
#         done
    
#     done

# done

# device="1"

# for round in {1..3}; do

#     folder="0119_oss_h200_softmax_exp_resume_26"
#     for step_range in "40:49" "30:39"; do
#         mkdir -p mla-decode-local/${folder}/step_${step_range}/out_device_${device}/round_${round}
#         for kid in {0..19}; do
#             rm -rf ~/.triton
#             rm -rf /root/.cache/torch_extensions
#             export CUSTOM_KERNEL_MODULE="mla-decode-local/${folder}/step_${step_range}/kernels/sample_${kid}.py"
#             timeout 120 python mla-decode-local/eval_local.py mla-decode-local/${folder}/step_${step_range}/out_device_${device}/round_${round}/sample_${kid}.out
#             sleep 15
#         done
#     done


#     folder="0118_oss_h200_top_20_all"
#     for step_range in "50:59"; do
#         mkdir -p mla-decode-local/${folder}/step_${step_range}/out_device_${device}/round_${round}
#         for kid in {0..19}; do
#             rm -rf ~/.triton
#             rm -rf /root/.cache/torch_extensions
#             export CUSTOM_KERNEL_MODULE="mla-decode-local/${folder}/step_${step_range}/kernels/sample_${kid}.py"
#             timeout 120 python mla-decode-local/eval_local.py mla-decode-local/${folder}/step_${step_range}/out_device_${device}/round_${round}/sample_${kid}.out
#             sleep 15
#         done
#     done

# done


folder="TTT_best_candidates/kernels"
export CUSTOM_KERNEL_MODULE="mla-decode-local/${folder}/step_30:34_sample_4.py"
python mla-decode-local/eval_local.py mla-decode-local/${folder}/out/sample_4.out

