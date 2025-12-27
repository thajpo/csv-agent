"""
Full pipeline orchestrator.

Runs all data generation stages sequentially to avoid resource conflicts.

Usage:
    uv run python -m src.datagen.run_all --both        # Full pipeline (default)
    uv run python -m src.datagen.run_all --synth       # Synthetic only
    uv run python -m src.datagen.run_all --llm         # LLM only
    uv run python -m src.datagen.run_all --triangulate # Episodes only (skip question gen)
    uv run python -m src.datagen.run_all --test        # Quick e2e test (1 question, 1 trace)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.core.config import config


def run_stage(name: str, cmd: list[str]) -> bool:
    """
    Run a pipeline stage.

    Returns True if stage produced usable output (exit 0 or 1).
    Returns False only if stage completely failed (exit 2+).

    Exit codes:
        0 = complete success
        1 = partial success (some failures, but output generated)
        2+ = total failure (no output)
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✓ {name} completed in {elapsed:.1f}s")
        return True
    elif result.returncode == 1:
        print(f"\n⚠ {name} completed with some failures in {elapsed:.1f}s (continuing)")
        return True  # Partial success is still success
    else:
        print(f"\n✗ {name} failed completely (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full data generation pipeline")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--both",
        action="store_true",
        default=True,
        help="Run both synthetic and LLM pipelines (default)",
    )
    mode.add_argument(
        "--synth",
        action="store_true",
        help="Only run synthetic pipeline",
    )
    mode.add_argument(
        "--llm",
        action="store_true",
        help="Only run LLM pipeline",
    )
    parser.add_argument(
        "--triangulate",
        action="store_true",
        help="Skip question generation, only run triangulation/episodes",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit questions per dataset",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick e2e test: 1 dataset, 1 question, 1 consistency trace",
    )
    args = parser.parse_args()

    run_synthetic = args.both or args.synth
    run_llm = args.both or args.llm

    # Handle default case (no flags = --both)
    if not args.synth and not args.llm:
        run_synthetic = True
        run_llm = True

    # Test mode: minimal settings for fast iteration
    max_questions = args.max_questions
    n_consistency = None
    max_datasets = None
    if args.test:
        max_questions = max_questions or 1
        n_consistency = 1
        max_datasets = 1
        print("🧪 TEST MODE: 1 dataset, 1 question, 1 consistency trace\n")

    stages_run = 0
    stages_failed = 0

    # Stage 1: Synthetic Questions
    if run_synthetic and not args.triangulate:
        cmd = ["uv", "run", "python", "-m", "src.datagen.synthetic.generator"]
        if max_datasets:
            cmd.extend(["--max-datasets", str(max_datasets)])
        if run_stage("Stage 1a: Generate Synthetic Questions", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 2: Synthetic Episodes
    if run_synthetic:
        cmd = [
            "uv", "run", "python", "-m", "src.datagen.synthetic_episodes",
            "--questions-dir", str(config.questions_synthetic_dir),
            "--output", str(config.episodes_synthetic_jsonl),
        ]
        if max_questions:
            cmd.extend(["--max-questions", str(max_questions)])

        if run_stage("Stage 2a: Generate Synthetic Episodes", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 3: LLM Questions
    if run_llm and not args.triangulate:
        cmd = ["uv", "run", "python", "-m", "src.datagen.question_gen"]
        if max_datasets:
            cmd.extend(["--max-datasets", str(max_datasets)])
        if run_stage("Stage 1b: Generate LLM Questions", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 4: LLM Episodes (with triangulation)
    if run_llm:
        cmd = [
            "uv", "run", "python", "-m", "src.datagen.episode_gen",
            "--questions-dir", str(config.questions_llm_dir),
            "--output", str(config.episodes_llm_jsonl),
            "--skip-difficulty-filter",
        ]
        if max_questions:
            cmd.extend(["--max-questions", str(max_questions)])
        if n_consistency:
            cmd.extend(["--n-consistency", str(n_consistency)])

        if run_stage("Stage 2b: Generate LLM Episodes (triangulated)", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Stages run: {stages_run}")
    print(f"  Stages failed: {stages_failed}")

    if run_synthetic:
        synth_episodes = Path(config.episodes_synthetic_jsonl)
        if synth_episodes.exists():
            count = sum(1 for _ in open(synth_episodes))
            print(f"  Synthetic episodes: {count}")

    if run_llm:
        llm_episodes = Path(config.episodes_llm_jsonl)
        if llm_episodes.exists():
            count = sum(1 for _ in open(llm_episodes))
            print(f"  LLM episodes: {count}")

    print()

    return 1 if stages_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
