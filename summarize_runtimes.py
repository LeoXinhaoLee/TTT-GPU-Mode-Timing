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
    """Extract sample ID from filename like 'sample_0.out' -> 0 or 'step_30:34_sample_6.out' -> 'step_30:34_sample_6'"""
    # Try to extract numeric sample ID first (for backward compatibility)
    match = re.search(r'sample_(\d+)\.out', filename)
    if match:
        return int(match.group(1))
    # Otherwise, use the basename without extension as the identifier
    return filename.replace('.out', '')

def main():
    # base_dir = Path("gpu_mode/mla-decode-local/0118_oss_h200_top_20_all/step_45:49/out")
    # base_dir = Path("gpu_mode/mla-decode-local/0118_oss_h200_top_20_all/step_20:24/out")
    base_dir = Path("gpu_mode/mla-decode-local/TTT_best_candidates/out")
    
    # Extract step_range from base_dir path (e.g., "step_45:49" -> "45:49")
    step_range = None
    for part in base_dir.parts:
        if part.startswith("step_"):
            step_range = part.replace("step_", "")
            break
    
    if step_range is None:
        raise ValueError(f"Could not extract step_range from path: {base_dir}")
    
    # Auto-detect all round directories
    round_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("round_")])
    if not round_dirs:
        raise ValueError(f"No round directories found in {base_dir}")
    
    # Extract round numbers and sort them
    round_nums = []
    for round_dir in round_dirs:
        match = re.search(r'round_(\d+)', round_dir.name)
        if match:
            round_nums.append(int(match.group(1)))
    round_nums = sorted(round_nums)
    
    print(f"Found {len(round_nums)} rounds: {round_nums}")
    
    # Dictionary to store runtimes: {sample_id: {round_num: runtime}}
    data = defaultdict(dict)
    
    # Process each round
    for round_num in round_nums:
        round_dir = base_dir / f"round_{round_num}"
        if not round_dir.exists():
            print(f"Warning: {round_dir} does not exist")
            continue
        
        # Process all .out files in this round
        for out_file in sorted(round_dir.glob("*.out")):
            sample_id = extract_sample_id(out_file.name)
            if sample_id is None:
                continue
            
            runtime = extract_geometric_mean(out_file)
            if runtime is not None:
                data[sample_id][round_num] = runtime
    
    # Prepare output data
    output_rows = []
    for sample_id in sorted(data.keys(), key=lambda x: (isinstance(x, int), x)):
        runtimes_dict = data[sample_id]
        
        # Get runtimes in order of round numbers
        runtimes = [runtimes_dict.get(round_num) for round_num in round_nums]
        
        # Calculate average and std (only for non-None values)
        valid_runtimes = [r for r in runtimes if r is not None]
        if valid_runtimes:
            avg_runtime = statistics.mean(valid_runtimes)
            std_runtime = statistics.stdev(valid_runtimes) if len(valid_runtimes) > 1 else 0.0
        else:
            avg_runtime = None
            std_runtime = None
        
        row_data = {
            'step_range': step_range,
            'sample_id': sample_id,
            'avg_runtime': avg_runtime,
            'std_runtime': std_runtime
        }
        # Add runtime columns for each round
        for round_num in round_nums:
            row_data[f'runtime_{round_num}'] = runtimes_dict.get(round_num)
        
        output_rows.append(row_data)
    
    # Write to TSV
    output_file = f"runtime_summary_{step_range}.tsv"
    with open(output_file, 'w') as f:
        # Write header
        header_parts = ["step_range", "sample_id"]
        header_parts.extend([f"runtime_{round_num}" for round_num in round_nums])
        header_parts.extend(["avg_runtime", "std_runtime"])
        f.write("\t".join(header_parts) + "\n")
        
        # Write data rows
        for row in output_rows:
            row_parts = [str(row['step_range']), str(row['sample_id'])]
            row_parts.extend([str(row.get(f'runtime_{round_num}', 'N/A') or 'N/A') for round_num in round_nums])
            row_parts.extend([str(row['avg_runtime']) if row['avg_runtime'] is not None else 'N/A',
                             str(row['std_runtime']) if row['std_runtime'] is not None else 'N/A'])
            f.write("\t".join(row_parts) + "\n")
    
    print(f"Summary written to {output_file}")
    print(f"Processed {len(output_rows)} samples")

if __name__ == "__main__":
    main()
