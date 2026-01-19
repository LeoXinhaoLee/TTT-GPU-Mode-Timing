#!/usr/bin/env python3
"""Extract leaderboard scores from test output files and output to TSV.

Usage:
    python extract_results.py [--output results.tsv]
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, StatisticsError
from typing import Optional

# Pattern to match: "Overall leaderboard score (microseconds, geom): 2354.804 us"
SCORE_PATTERN = re.compile(r"Overall leaderboard score \(microseconds, geom\): ([\d.]+) us")

# Pattern to match: "wall_time_seconds: 452.494"
RUNTIME_PATTERN = re.compile(r"wall_time_seconds: ([\d.]+)")

# Base directory for the experiment
BASE_DIR = Path(__file__).parent / "gen" / "0102-trimul-per-no-terminal-est-none-tinker-gpt-oss-120b-high-train-resp-28k-kl-1e-1-BS-512-x-run-1"


def extract_score_from_file(file_path: Path) -> Optional[float]:
    """Extract the leaderboard score from an output file.
    
    Returns:
        The score as a float, or None if not found.
    """
    try:
        content = file_path.read_text()
        match = SCORE_PATTERN.search(content)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
    return None


def extract_runtime_from_file(file_path: Path) -> Optional[float]:
    """Extract the wall time runtime from an output file.
    
    Returns:
        The runtime in seconds as a float, or None if not found.
    """
    try:
        content = file_path.read_text()
        match = RUNTIME_PATTERN.search(content)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
    return None


def extract_test_round(test_dir_name: str) -> Optional[int]:
    """Extract test round number from directory name like 'test_1'."""
    match = re.search(r"test_(\d+)", test_dir_name)
    if match:
        return int(match.group(1))
    return None


def extract_sample_id(filename: str) -> Optional[int]:
    """Extract sample ID from filename like 'test_sample_0.out'."""
    match = re.search(r"test_sample_(\d+)\.out", filename)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract leaderboard scores from test output files."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent / "results.tsv",
        help="Output TSV file path (default: results.tsv)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help=f"Base directory containing step_*_top10 directories (default: {BASE_DIR})",
    )
    args = parser.parse_args()

    base_dir: Path = args.base_dir
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Structure: {(gpu, step_range, sample_id): {test_round: (score, runtime)}}
    data_by_key: dict[tuple[str, str, int], dict[int, tuple[float, float]]] = defaultdict(dict)

    # Find all step_*_top10 directories
    step_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("step_") and d.name.endswith("_top10")])
    if not step_dirs:
        raise FileNotFoundError(f"No step_*_top10 directories found in {base_dir}")

    for step_dir in step_dirs:
        step_range = step_dir.name.replace("_top10", "")
        out_dir = step_dir / "out"
        
        if not out_dir.exists():
            print(f"Warning: {out_dir} does not exist, skipping", file=sys.stderr)
            continue

        # Find all test_* directories
        test_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("test_")])
        
        for test_dir in test_dirs:
            test_round = extract_test_round(test_dir.name)
            if test_round is None:
                print(f"Warning: Could not extract test round from {test_dir.name}", file=sys.stderr)
                continue
            
            # Find all GPU directories (A100, B200, H100, etc.)
            for gpu_dir in test_dir.iterdir():
                if not gpu_dir.is_dir() or gpu_dir.name.endswith(".out"):
                    continue
                
                gpu = gpu_dir.name
                
                # Find all test_sample_*.out files
                for out_file in gpu_dir.glob("test_sample_*.out"):
                    sample_id = extract_sample_id(out_file.name)
                    if sample_id is None:
                        print(f"Warning: Could not extract sample ID from {out_file.name}", file=sys.stderr)
                        continue

                    score = extract_score_from_file(out_file)
                    runtime = extract_runtime_from_file(out_file)
                    
                    if score is not None:
                        key = (gpu, step_range, sample_id)
                        data_by_key[key][test_round] = (score, runtime)

    # Calculate statistics and write TSV
    output_path: Path = args.output
    with open(output_path, "w") as f:
        # Write header
        header = ["gpu", "step_range", "sample_id", "avg_score", "std_score", "run_1", "run_2", "run_3"]
        f.write("\t".join(header) + "\n")

        # Write data rows, sorted by gpu, step_range, sample_id
        for (gpu, step_range, sample_id) in sorted(data_by_key.keys()):
            test_data = data_by_key[(gpu, step_range, sample_id)]
            
            # Extract scores for statistics
            scores = [score for score, _ in test_data.values()]
            avg_score = mean(scores)
            try:
                std_score = stdev(scores)
            except StatisticsError:
                # StatisticsError is raised when there are fewer than 2 values
                std_score = 0.0
            
            # Get runtimes for each test round
            runtime_1 = test_data.get(1, (None, None))[1]
            runtime_2 = test_data.get(2, (None, None))[1]
            runtime_3 = test_data.get(3, (None, None))[1]
            
            row = [
                gpu,
                step_range,
                str(sample_id),
                f"{avg_score:.3f}",
                f"{std_score:.3f}",
                f"{runtime_1:.3f}" if runtime_1 is not None else "N/A",
                f"{runtime_2:.3f}" if runtime_2 is not None else "N/A",
                f"{runtime_3:.3f}" if runtime_3 is not None else "N/A",
            ]
            f.write("\t".join(row) + "\n")

    print(f"Extracted scores for {len(data_by_key)} unique (gpu, step_range, sample_id) combinations", file=sys.stderr)
    print(f"Output written to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
