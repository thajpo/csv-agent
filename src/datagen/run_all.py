"""
Full pipeline orchestrator.

Runs all data generation stages sequentially to avoid resource conflicts.

Usage:
    uv run python -m src.datagen.run_all                    # Full pipeline
    uv run python -m src.datagen.run_all --synthetic-only   # Just synthetic
    uv run python -m src.datagen.run_all --llm-only         # Just LLM-based
    uv run python -m src.datagen.run_all --skip-questions   # Skip question gen
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.core.config import config


def run_stage(name: str, cmd: list[str]) -> bool:
    """Run a pipeline stage, return True if successful."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✓ {name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n✗ {name} failed (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full data generation pipeline")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Only run synthetic pipeline (skip LLM-based)",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Only run LLM pipeline (skip synthetic)",
    )
    parser.add_argument(
        "--skip-questions",
        action="store_true",
        help="Skip question generation (use existing questions)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit questions per dataset (for testing)",
    )
    args = parser.parse_args()

    run_synthetic = not args.llm_only
    run_llm = not args.synthetic_only

    stages_run = 0
    stages_failed = 0

    # Stage 1: Synthetic Questions
    if run_synthetic and not args.skip_questions:
        cmd = ["uv", "run", "python", "-m", "src.datagen.synthetic.generator"]
        if run_stage("Stage 1a: Generate Synthetic Questions", cmd):
            stages_run += 1
        else:
            stages_failed += 1
            print("Aborting: synthetic question generation failed")
            return 1

    # Stage 2: Synthetic Episodes
    if run_synthetic:
        cmd = [
            "uv", "run", "python", "-m", "src.datagen.synthetic_episodes",
            "--questions-dir", str(config.questions_synthetic_dir),
            "--output", str(config.episodes_synthetic_jsonl),
        ]
        if args.max_questions:
            cmd.extend(["--max-questions", str(args.max_questions)])

        if run_stage("Stage 2a: Generate Synthetic Episodes", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 3: LLM Questions
    if run_llm and not args.skip_questions:
        cmd = ["uv", "run", "python", "-m", "src.datagen.question_gen"]
        if run_stage("Stage 1b: Generate LLM Questions", cmd):
            stages_run += 1
        else:
            stages_failed += 1
            print("Aborting: LLM question generation failed")
            return 1

    # Stage 4: LLM Episodes (with triangulation)
    if run_llm:
        cmd = [
            "uv", "run", "python", "-m", "src.datagen.episode_gen",
            "--questions-dir", str(config.questions_llm_dir),
            "--output", str(config.episodes_llm_jsonl),
            "--skip-difficulty-filter",
        ]
        if args.max_questions:
            cmd.extend(["--max-questions", str(args.max_questions)])

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
