#!/bin/bash

set -x

rm -rf ~/.triton
rm -rf /root/.cache/torch_extensions

cd gpu_mode
# export CUSTOM_KERNEL_MODULE="bioml/trimul/MI300X_kernels/TTT_MI300X.py"
# python bioml/trimul/eval_local.py bioml/trimul/MI300X_kernels/TTT_MI300X.out

for round in {1..10}; do

    # folder="bioml/trimul/BoN_naive_top_20/A100_top_20"
    # folder="bioml/trimul/BoN_naive_top_20/best"
    # folder="bioml/trimul/leaderboard_kernels/MI300X"
    folder="bioml/trimul/gen_kernels"
    for py_file in ${folder}/kernels/*.py; do
        rm -rf ~/.triton
        rm -rf /root/.cache/torch_extensions
        py_file_name=$(basename "${py_file%.py}")
        out_file="${folder}/out/${py_file_name}.out"
        export CUSTOM_KERNEL_MODULE="$py_file"
        timeout 360 python bioml/trimul/eval_local.py "$out_file"
    done

    # Summarize geometric-mean from all .out files
    mkdir -p "${folder}/out"
    summary_file="${folder}/out/geometric_mean_summary.tsv"
    echo -e "sample_id\tgeometric_mean" > "$summary_file"

    geometric_means=()
    for out_file in ${folder}/out/*.out; do
        if [ -f "$out_file" ]; then
            sample_id=$(basename "${out_file%.out}")
            # Extract geometric-mean value from the .out file
            geo_mean=$(grep "^geometric-mean:" "$out_file" | awk '{print $2}')
            if [ -n "$geo_mean" ]; then
                echo -e "$sample_id\t$geo_mean" >> "$summary_file"
                geometric_means+=("$geo_mean")
            fi
        fi
    done

    # Calculate overall geometric mean
    if [ ${#geometric_means[@]} -gt 0 ]; then
        # Use awk to calculate geometric mean: exp(sum(log(values)) / count)
        overall_geo_mean=$(printf '%s\n' "${geometric_means[@]}" | awk '{
            if ($1 > 0) {
                log_sum += log($1)
                count++
            }
        } END {
            if (count > 0) {
                print exp(log_sum / count)
            }
        }')
        echo -e "overall\t$overall_geo_mean" >> "$summary_file"
        echo "Summary written to: $summary_file"
    fi

    mv ${folder}/out ${folder}/out_${round}

done
