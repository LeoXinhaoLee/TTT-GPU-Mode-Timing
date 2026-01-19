#!/usr/bin/env python3
"""
Helper script to run tasks on Modal directly, using task definitions.

Supported tasks:
  - trimul: BioML TriMul task (bioml/trimul/task.yml)

Usage (from project root):
    python src/run_trimul_modal.py --submission path/to/submission.py --task trimul

You must:
  1. Be authenticated with Modal (`modal token new`)
  2. Have deployed the Modal app:
         cd src/runners
         modal deploy modal_runner_archs.py
"""

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from libkernelbot.consts import ModalGPU, SubmissionMode
from libkernelbot.launchers import ModalLauncher
from libkernelbot.report import RunProgressReporter
from libkernelbot.run_eval import FullResult
from libkernelbot.submission import compute_score
from libkernelbot.task import LeaderboardTask, build_task_config, make_task_definition


PROJECT_ROOT = Path(__file__).resolve().parent
TRIMUL_TASK_YAML = PROJECT_ROOT / "bioml" / "trimul" / "task.yml"


class SimpleReporter(RunProgressReporter):
    """Minimal reporter that prints to console."""

    async def _update_message(self):
        print(f"[{self.title}]")
        for line in self.lines:
            print(f"  {line}")

    async def display_report(self, title: str, report):
        print(f"\n=== {title} ===")
        print(f"Report has {len(report.data)} items")


def load_task(task_name: str = "trimul") -> LeaderboardTask:
    """Load a LeaderboardTask from its YAML definition.
    
    Args:
        task_name: One of "trimul" or "nvfp4_gemm"
        
    Returns:
        The loaded LeaderboardTask
    """
    task_map = {
        "trimul": TRIMUL_TASK_YAML,
    }
    
    if task_name not in task_map:
        valid = ", ".join(task_map.keys())
        raise ValueError(f"Invalid task name '{task_name}'. Valid tasks: {valid}")
    
    task_yaml = task_map[task_name]
    if not task_yaml.exists():
        raise FileNotFoundError(
            f"Could not find task definition at {task_yaml}. "
            "Run this script from the project root."
        )
    definition = make_task_definition(task_yaml)
    return definition.task


def load_trimul_task() -> LeaderboardTask:
    """Load the TriMul LeaderboardTask from its YAML definition.
    
    Deprecated: Use load_task("trimul") instead.
    """
    return load_task("trimul")


async def run_on_modal(
    submission_code: str,
    gpu_type: str = "T4",
    mode: str = "test",
    task_name: str = "trimul",
    app_name: str = "discord-bot-runner",
    timeout_seconds: Optional[int] = None,
) -> tuple[FullResult, LeaderboardTask]:
    """
    Run a submission on Modal using the official task definition.

    Args:
        submission_code: Contents of the user's `submission.py`
        gpu_type: One of ModalGPU names (T4, L4, A100, H100, H200, B200, L4x4)
        mode: One of: test, benchmark, leaderboard, profile, private
        task_name: One of "trimul" or "nvfp4_gemm"
        timeout_seconds: Optional timeout in seconds (default: 600). Only works after redeploying modal_runner_archs.py once.
    """
    # Load task from task YAML
    task = load_task(task_name)

    # Map CLI mode to SubmissionMode enum
    try:
        mode_enum = SubmissionMode(mode)
    except ValueError as e:
        valid = ", ".join(m.value for m in SubmissionMode)
        raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid}") from e

    # Build config using the same path as the backend
    config = build_task_config(
        task=task,
        submission_content=submission_code,
        arch=None,  # Python task – arch unused
        mode=mode_enum,
    )
    
    # Add timeout_seconds to config if provided (will be read by modal_run_config)
    if timeout_seconds is not None:
        config["timeout_seconds"] = timeout_seconds

    # Set up Modal launcher
    launcher = ModalLauncher(add_include_dirs=[], app_name=app_name)
    gpu_enum = ModalGPU[gpu_type.upper()]

    task_display_name = task_name.upper() if task_name == "nvfp4_gemm" else task_name.capitalize()
    reporter = SimpleReporter(f"{task_display_name} on {gpu_enum.name} (Modal)")

    print(f"Submitting {task_display_name} task to Modal on {gpu_enum.name} with mode='{mode_enum.value}'...")

    result = await launcher.run_submission(config, gpu_enum, reporter)
    return result, task


def print_benchmark_details(result: FullResult):
    """Print per-benchmark statistics for a leaderboard run if available."""
    if "leaderboard" not in result.runs:
        return

    run_res = result.runs["leaderboard"].run
    if not run_res or not run_res.result:
        return

    data = run_res.result
    if "benchmark-count" not in data:
        return

    num_benchmarks = int(data["benchmark-count"])
    print(f"\nLeaderboard benchmarks: {num_benchmarks}")
    for i in range(num_benchmarks):
        prefix = f"benchmark.{i}."
        mean_ns = float(data.get(prefix + "mean", 0.0))
        std_ns = float(data.get(prefix + "std", 0.0))
        best_ns = float(data.get(prefix + "best", 0.0))
        worst_ns = float(data.get(prefix + "worst", 0.0))

        mean_s = mean_ns / 1e9 if mean_ns else 0.0
        std_s = std_ns / 1e9 if std_ns else 0.0
        best_s = best_ns / 1e9 if best_ns else 0.0
        worst_s = worst_ns / 1e9 if worst_ns else 0.0

        print(f"  Benchmark {i}:")
        print(f"    mean:   {mean_s:.6f} s")
        print(f"    std:    {std_s:.6f} s")
        print(f"    best:   {best_s:.6f} s")
        print(f"    worst:  {worst_s:.6f} s")


def print_result(result: FullResult, task: LeaderboardTask | None = None):
    """Pretty print a FullResult and optionally the leaderboard score."""
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Success: {result.success}")

    if not result.success:
        print("\nSystem Info:")
        if result.system.gpu:
            print(f"  GPU: {result.system.gpu}")
        if result.system.cpu:
            print(f"  CPU: {result.system.cpu}")
        print(f"  Requeues: {result.system.requeues}")
        print(f"Error: {result.error}")
        return

    print("\nSystem Info:")
    print(f"  GPU: {result.system.gpu}")
    print(f"  CPU: {result.system.cpu}")
    print(f"  Requeues: {result.system.requeues}")
    print(f"  Torch: {result.system.torch}")
    print(f"  Runtime: {result.system.runtime}")

    print(f"\nRuns: {len(result.runs)}")
    for run_name, run_result in result.runs.items():
        print(f"\n  Run: {run_name}")
        print(f"    Start: {run_result.start}")
        print(f"    End:   {run_result.end}")

        if run_result.compilation:
            comp = run_result.compilation
            print("    Compilation:")
            print(f"      Success:  {comp.success}")
            if not comp.success:
                print(f"      ExitCode: {comp.exit_code}")
                print(f"      Stderr:   {comp.stderr}...")

        if run_result.run:
            run = run_result.run
            print("    Execution:")
            print(f"      Success:   {run.success}")
            print(f"      Passed:    {run.passed}")
            print(f"      Duration:  {run.duration:.2f}s")
            print(f"      Exit Code: {run.exit_code}")

            if run.stdout:
                print(f"      Stdout:\n{run.stdout[:500]}{'...' if len(run.stdout) > 500 else ''}")
            if run.stderr:
                print(f"      Stderr:\n{run.stderr}{'...' if len(run.stderr) > 500 else ''}")

    # If we have a task and a leaderboard run, print per-benchmark stats and score
    if task is not None and "leaderboard" in result.runs:
        print_benchmark_details(result)
        try:
            score_seconds = compute_score(result, task, submission_id=-1)
            score_us = score_seconds * 1_000_000
            print(f"\nOverall leaderboard score (microseconds, {task.ranking_by.value}): {score_us:.3f} us")
        except Exception as e:
            print(f"\nCould not compute leaderboard score: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run submission on Modal using the official task definition.",
    )
    parser.add_argument(
        "--submission",
        "-s",
        required=True,
        help="Path to your submission.py file.",
    )
    parser.add_argument(
        "--task",
        "-t",
        default="trimul",
        help="Task to run (default: trimul).",
    )
    parser.add_argument(
        "--gpu",
        "-g",
        default="H100",
        choices=[g.name for g in ModalGPU],
        help="Modal GPU type to use (default: T4).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="leaderboard",
        choices=[m.value for m in SubmissionMode],
        help="Submission mode (default: leaderboard).",
    )
    parser.add_argument(
        "--app-name",
        "-a",
        default="discord-bot-runner-short-timeout",
        help="Modal app name to use (default: discord-bot-runner).",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    submission_path = Path(args.submission)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    submission_code = submission_path.read_text()

    result, task = await run_on_modal(
        submission_code=submission_code,
        gpu_type=args.gpu,
        mode=args.mode,
        task_name=args.task,
        app_name=args.app_name,
    )

    print_result(result, task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")

