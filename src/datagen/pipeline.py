"""
Full pipeline orchestrator.

Runs all data generation stages sequentially to avoid resource conflicts.

Usage (via CLI):
    csvagent run --all         # Full pipeline
    csvagent run --template    # Template mode only
    csvagent run --procedural  # Procedural mode only
    csvagent run --llm-gen     # LLM generation mode only
    csvagent run --test        # Quick e2e test (1 question, 1 trace)
"""

import subprocess
import time
import asyncio
from pathlib import Path

from src.core.config import config
from src.datagen.validate_synthetic import main as validate_synthetic_main


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


def run_synthetic_stage(
    name: str,
    questions_dir: str,
    output_path: str,
    max_questions: int | None,
    source: str,
) -> bool:
    """Run synthetic episode generation in-process with source-scoped filtering."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = asyncio.run(
        validate_synthetic_main(
            questions_dir=questions_dir,
            output_path=output_path,
            max_questions=max_questions,
            source=source,
        )
    )
    elapsed = time.time() - start

    if result == 0:
        print(f"\n✓ {name} completed in {elapsed:.1f}s")
        return True
    elif result == 1:
        print(f"\n⚠ {name} completed with some failures in {elapsed:.1f}s (continuing)")
        return True
    else:
        print(f"\n✗ {name} failed completely (exit code {result})")
        return False


def main(
    mode: str = "all",
    test: bool = False,
    max_questions: int | None = None,
) -> int:
    """
    Run full data generation pipeline.

    Args:
        mode: "template", "procedural", "llm_gen", or "all"
        test: Quick e2e test (1 dataset, 1 question, 1 consistency trace)
        max_questions: Limit questions per dataset

    Returns:
        0 if all stages succeeded, 1 if any failed
    """
    run_template = mode in ("template", "all")
    run_procedural = mode in ("procedural", "all")
    run_llm = mode in ("llm_gen", "all")

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

    # Stage 1a: Template Questions
    if run_template:
        cmd = ["uv", "run", "python", "-m", "src.datagen.synthetic.generator"]
        if max_datasets:
            cmd.extend(["--max-datasets", str(max_datasets)])
        if run_stage("Stage 1a: Generate Template Questions", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 2a: Template Episodes
    if run_template:
        if run_synthetic_stage(
            "Stage 2a: Generate Template Episodes",
            questions_dir=str(config.questions_template_dir),
            output_path=str(config.episodes_template_jsonl),
            max_questions=max_questions,
            source="template",
        ):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 1b: Procedural Questions
    if run_procedural:
        cmd = ["uv", "run", "python", "-m", "src.datagen.synthetic.programs.runner"]
        if max_datasets:
            cmd.extend(["--max-datasets", str(max_datasets)])
        if run_stage("Stage 1b: Generate Procedural Questions", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 2b: Procedural Episodes
    if run_procedural:
        if run_synthetic_stage(
            "Stage 2b: Generate Procedural Episodes",
            questions_dir=str(config.questions_procedural_dir),
            output_path=str(config.episodes_procedural_jsonl),
            max_questions=max_questions,
            source="procedural",
        ):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 1c: LLM Questions
    if run_llm:
        cmd = ["uv", "run", "python", "-m", "src.datagen.question_gen"]
        if max_datasets:
            cmd.extend(["--max-datasets", str(max_datasets)])
        if run_stage("Stage 1c: Generate LLM Questions", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Stage 2c: LLM Episodes
    if run_llm:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "src.datagen.episode_gen",
            "--questions-dir",
            str(config.questions_llm_gen_dir),
            "--output",
            str(config.episodes_llm_gen_jsonl),
            "--skip-difficulty-filter",
        ]
        if max_questions:
            cmd.extend(["--max-questions", str(max_questions)])
        if n_consistency:
            cmd.extend(["--n-consistency", str(n_consistency)])

        if run_stage("Stage 2c: Generate LLM Episodes", cmd):
            stages_run += 1
        else:
            stages_failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Stages run: {stages_run}")
    print(f"  Stages failed: {stages_failed}")

    if run_template:
        template_episodes = Path(config.episodes_template_jsonl)
        if template_episodes.exists():
            count = sum(1 for _ in open(template_episodes))
            print(f"  Template episodes: {count}")

    if run_procedural:
        procedural_episodes = Path(config.episodes_procedural_jsonl)
        if procedural_episodes.exists():
            count = sum(1 for _ in open(procedural_episodes))
            print(f"  Procedural episodes: {count}")

    if run_llm:
        llm_episodes = Path(config.episodes_llm_gen_jsonl)
        if llm_episodes.exists():
            count = sum(1 for _ in open(llm_episodes))
            print(f"  LLM episodes: {count}")

    print()

    return 1 if stages_failed > 0 else 0
