#!/usr/bin/env python3
"""
Simple script to calculate mean and std of geometric-mean values
from BoN_best/out/round_{i}/*.out files.
"""

import re
import statistics
from pathlib import Path

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

def main():
    base_dir = Path("gpu_mode/mla-decode-local/BoN_best/out")
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return
    
    # Collect all geometric-mean values from all rounds
    all_values = []
    
    # Find all round directories
    for round_dir in sorted(base_dir.glob("round_*")):
        if not round_dir.is_dir():
            continue
        
        round_num = round_dir.name
        print(f"Processing {round_num}...")
        
        # Process all .out files in this round
        for out_file in sorted(round_dir.glob("*.out")):
            value = extract_geometric_mean(out_file)
            if value is not None:
                all_values.append(value)
                print(f"  {out_file.name}: {value:.6f}")
            else:
                print(f"  {out_file.name}: No geometric-mean found")
    
    # Calculate statistics
    if not all_values:
        print("\nNo valid geometric-mean values found!")
        return
    
    mean_value = statistics.mean(all_values)
    std_value = statistics.stdev(all_values) if len(all_values) > 1 else 0.0
    
    print(f"\n{'='*50}")
    print(f"Total files processed: {len(all_values)}")
    print(f"Mean:  {mean_value:.6f}")
    print(f"Std:   {std_value:.6f}")
    print(f"{'='*50}")
    
    # Write to TSV file
    output_file = "bon_stats.tsv"
    with open(output_file, 'w') as f:
        # Write header
        f.write("total_files\tmean\tstd\n")
        # Write data
        f.write(f"{len(all_values)}\t{mean_value:.6f}\t{std_value:.6f}\n")
    
    print(f"\nResults written to {output_file}")

if __name__ == "__main__":
    main()
