#!/usr/bin/env python3
"""
Script to summarize runtime data from .out files.
Extracts geometric-mean values from each round and calculates statistics.
"""

import os
import re
import statistics
from pathlib import Path
from collections import defaultdict

def extract_geometric_mean(filepath):
    """Extract geometric-mean value from .out file."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('geometric-mean:'):
                    # Extract the number after the colon
                    match = re.search(r'geometric-mean:\s*([\d.]+)', line)
                    if match:
                        return float(match.group(1))
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def extract_sample_id(filename):
    """Extract sample ID from filename like 'sample_0.out' -> 0"""
    match = re.search(r'sample_(\d+)\.out', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    # base_dir = Path("gpu_mode/mla-decode-local/0118_oss_h200_top_20_all/step_45:49/out")
    # base_dir = Path("gpu_mode/mla-decode-local/0118_oss_h200_top_20_all/step_15:19/out")
    # base_dir = Path("gpu_mode/mla-decode-local/0119_oss_h200_BoN/step_0:49/out")
    base_dir = Path("gpu_mode/0119_oss_h200_softmax_exp_resume_26")

    # Extract step_range from base_dir path (e.g., "step_45:49" -> "45:49")
    step_range = None
    for part in base_dir.parts:
        if part.startswith("step_"):
            step_range = part.replace("step_", "")
            break
    
    if step_range is None:
        raise ValueError(f"Could not extract step_range from path: {base_dir}")
    
    # Dictionary to store runtimes: {sample_id: [runtime_1, runtime_2, runtime_3]}
    data = defaultdict(lambda: [None, None, None])
    
    # Process each round
    for round_num in [1, 2, 3]:
        round_dir = base_dir / f"round_{round_num}"
        if not round_dir.exists():
            print(f"Warning: {round_dir} does not exist")
            continue
        
        # Process all .out files in this round
        for out_file in sorted(round_dir.glob("sample_*.out")):
            sample_id = extract_sample_id(out_file.name)
            if sample_id is None:
                continue
            
            runtime = extract_geometric_mean(out_file)
            if runtime is not None:
                data[sample_id][round_num - 1] = runtime
    
    # Prepare output data
    output_rows = []
    for sample_id in sorted(data.keys()):
        runtimes = data[sample_id]
        runtime_1, runtime_2, runtime_3 = runtimes
        
        # Calculate average and std (only for non-None values)
        valid_runtimes = [r for r in runtimes if r is not None]
        if valid_runtimes:
            avg_runtime = statistics.mean(valid_runtimes)
            std_runtime = statistics.stdev(valid_runtimes) if len(valid_runtimes) > 1 else 0.0
        else:
            avg_runtime = None
            std_runtime = None
        
        output_rows.append({
            'step_range': step_range,
            'sample_id': sample_id,
            'runtime_1': runtime_1,
            'runtime_2': runtime_2,
            'runtime_3': runtime_3,
            'avg_runtime': avg_runtime,
            'std_runtime': std_runtime
        })
    
    # Sort by avg_runtime (ascending), with None values last
    output_rows.sort(key=lambda x: (x['avg_runtime'] is None, x['avg_runtime'] or float('inf')))
    
    # Write to TSV
    output_file = f"runtime_summary_{step_range}.tsv"
    with open(output_file, 'w') as f:
        # Write header
        f.write("step_range\tsample_id\truntime_1\truntime_2\truntime_3\tavg_runtime\tstd_runtime\n")
        
        # Write data rows
        for row in output_rows:
            f.write(f"{row['step_range']}\t{row['sample_id']}\t")
            f.write(f"{row['runtime_1'] or 'N/A'}\t{row['runtime_2'] or 'N/A'}\t{row['runtime_3'] or 'N/A'}\t")
            f.write(f"{row['avg_runtime'] or 'N/A'}\t{row['std_runtime'] or 'N/A'}\n")
    
    print(f"Summary written to {output_file}")
    print(f"Processed {len(output_rows)} samples")

if __name__ == "__main__":
    main()
