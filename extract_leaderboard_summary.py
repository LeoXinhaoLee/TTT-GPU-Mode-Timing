#!/usr/bin/env python3
"""Extract and summarize geometric-mean values from leaderboard kernel output files.

Extracts data from:
- bioml/trimul/leaderboard_kernels/MI300X/out_{round}/sample_{id}.out
- bioml/trimul/gen_kernels/out_{round}/{kernel_name}.out
and summarizes to TSV files.
"""

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

BASE_DIRS = [
    Path("gpu_mode/bioml/trimul/leaderboard_kernels/MI300X"),
    Path("gpu_mode/bioml/trimul/gen_kernels"),
]


def extract_geometric_mean(file_path: Path) -> Optional[float]:
    """Extract geometric-mean value from an output file.
    
    Returns:
        The geometric-mean as a float, or None if not found.
    """
    try:
        content = file_path.read_text()
        # Look for "geometric-mean: <value>" line
        match = re.search(r"^geometric-mean:\s*([\d.]+)", content, re.MULTILINE)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
    return None


def extract_check_status(file_path: Path) -> Optional[str]:
    """Extract the final check status from an output file.
    
    Returns:
        "pass" or "fail" from the last check: line, or None if not found.
    """
    try:
        content = file_path.read_text()
        # Find all check: lines and get the last one (final status)
        matches = list(re.finditer(r"^check:\s*(pass|fail)", content, re.MULTILINE))
        if matches:
            return matches[-1].group(1)
    except Exception:
        pass
    return None


def extract_sample_id_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Extract sample_id from filename, handling both naming patterns.
    
    Returns:
        Tuple of (sample_id, source_type) or None if pattern doesn't match.
        source_type is either "sample" for sample_{id}.out or "kernel" for {kernel_name}.out
    """
    # Try sample_{id}.out pattern (leaderboard_kernels)
    sample_match = re.match(r"sample_(\d+)\.out", filename)
    if sample_match:
        return (sample_match.group(1), "sample")
    
    # Try {kernel_name}.out pattern (gen_kernels)
    kernel_match = re.match(r"(.+)\.out", filename)
    if kernel_match:
        return (kernel_match.group(1), "kernel")
    
    return None


def process_directory(base_dir: Path):
    """Process a single base directory and return sample_data.
    
    Returns:
        Dictionary mapping (sample_id, base_dir) to data dict.
    """
    if not base_dir.exists():
        print(f"Warning: Base directory {base_dir} does not exist, skipping", file=sys.stderr)
        return {}
    
    # Collect data for each sample_id
    # Structure: sample_data[(sample_id, base_dir)] = {
    #     'geometric_means': [list of values],
    #     'num_success': count,
    #     'total_rounds': count
    # }
    sample_data = {}
    
    # Find all out_* directories
    out_dirs = sorted([d for d in base_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("out_")])
    
    if not out_dirs:
        print(f"Warning: No out_* directories found in {base_dir}", file=sys.stderr)
        return {}
    
    # Process each round directory
    for out_dir in out_dirs:
        # Extract round number from directory name (e.g., "out_1" -> 1)
        round_match = re.match(r"out_(\d+)", out_dir.name)
        if not round_match:
            continue
        
        # Find all .out files
        for out_file in sorted(out_dir.glob("*.out")):
            # Skip summary files
            if "summary" in out_file.name.lower():
                continue
            
            # Extract sample_id from filename
            result = extract_sample_id_from_filename(out_file.name)
            if not result:
                continue
            
            sample_id, source_type = result
            key = (sample_id, base_dir)
            
            if key not in sample_data:
                sample_data[key] = {
                    'geometric_means': [],
                    'num_success': 0,
                    'total_rounds': 0
                }
            
            sample_data[key]['total_rounds'] += 1
            
            # Extract check status
            check_status = extract_check_status(out_file)
            if check_status == "pass":
                sample_data[key]['num_success'] += 1
            
            # Extract geometric-mean (extract even if failed, for avg_runtime calculation)
            geo_mean = extract_geometric_mean(out_file)
            if geo_mean is not None:
                sample_data[key]['geometric_means'].append(geo_mean)
    
    return sample_data


def main():
    # Process all base directories and write summary for each
    for base_dir in BASE_DIRS:
        if not base_dir.exists():
            print(f"Warning: Base directory {base_dir} does not exist, skipping", file=sys.stderr)
            continue
        
        print(f"Processing {base_dir}...", file=sys.stderr)
        dir_data = process_directory(base_dir)
        
        if not dir_data:
            print(f"Warning: No data found in {base_dir}", file=sys.stderr)
            continue
        
        output_path = base_dir / "summary.tsv"
        
        with open(output_path, "w") as f:
            # Write header
            f.write("sample_id\tnum_success\ttotal_rounds\tavg_runtime\tstd_runtime\n")
            
            # Write data rows, sorted appropriately
            def sort_key(key):
                sample_id, _ = key
                # For numeric sample IDs, sort numerically; otherwise alphabetically
                try:
                    return (0, int(sample_id))
                except ValueError:
                    return (1, sample_id)
            
            for key in sorted(dir_data.keys(), key=sort_key):
                sample_id, _ = key
                data = dir_data[key]
                geo_means = data['geometric_means']
                
                if geo_means:
                    avg_runtime = statistics.mean(geo_means)
                    if len(geo_means) > 1:
                        std_runtime = statistics.stdev(geo_means)
                    else:
                        std_runtime = 0.0
                else:
                    avg_runtime = 0.0
                    std_runtime = 0.0
                
                f.write(f"{sample_id}\t{data['num_success']}\t{data['total_rounds']}\t"
                       f"{avg_runtime:.6f}\t{std_runtime:.6f}\n")
        
        print(f"Summary written to: {output_path}", file=sys.stderr)
        print(f"Processed {len(dir_data)} entries from {base_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
