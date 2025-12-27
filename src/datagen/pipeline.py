"""
Full pipeline orchestrator.

Runs all data generation stages sequentially to avoid resource conflicts.

Usage (via CLI):
    csvagent run --both        # Full pipeline (default)
    csvagent run --synth       # Synthetic only
    csvagent run --llm         # LLM only
    csvagent run --triangulate # Episodes only (skip question gen)
    csvagent run --test        # Quick e2e test (1 question, 1 trace)
"""

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
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")

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


def main(
    mode: str = "both",
    triangulate: bool = False,
    test: bool = False,
    max_questions: int | None = None,
) -> int:
    """
    Run full data generation pipeline.

    Args:
        mode: "synth", "llm", or "both"
        triangulate: Skip question generation, only run episodes
        test: Quick e2e test (1 dataset, 1 question, 1 consistency trace)
        max_questions: Limit questions per dataset

    Returns:
        0 if all stages succeeded, 1 if any failed
    """
    run_synthetic = mode in ("synth", "both")
    run_llm = mode in ("llm", "both")

    # Test mode: minimal settings for fast iteration
    n_consistency = None
    max_datasets = None
    if test:
        max_questions = max_questions or 1
        n_consistency = 1
        max_datasets = 1
        print("🧪 TEST MODE: 1 dataset, 1 question, 1 consistency trace\n")

    stages_run = 0
    stages_failed = 0

    # Stage 1: Synthetic Questions
    if run_synthetic and not triangulate:
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
            "uv",
            "run",
            "python",
            "-m",
            "src.datagen.validate_synthetic",
            "--questions-dir",
            str(config.questions_synthetic_dir),
            "--output",
            str(config.episodes_synthetic_jsonl),
        ]
        if max_questions:
            cmd.extend(["--max-questions", str(max_questions)])

        if run_stage("Stage 2a: Generate Synthetic Episodes", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 3: LLM Questions
    if run_llm and not triangulate:
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
            "uv",
            "run",
            "python",
            "-m",
            "src.datagen.episode_gen",
            "--questions-dir",
            str(config.questions_llm_dir),
            "--output",
            str(config.episodes_llm_jsonl),
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
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
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
