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

SOURCE_SPECS = (
    {
        "mode": "template",
        "question_stage": "Stage 1a: Generate Template Questions",
        "episode_stage": "Stage 2a: Generate Template Episodes",
        "question_module": "src.datagen.synthetic.generator",
        "question_dir_attr": "questions_template_dir",
        "episode_path_attr": "episodes_template_jsonl",
    },
    {
        "mode": "procedural",
        "question_stage": "Stage 1b: Generate Procedural Questions",
        "episode_stage": "Stage 2b: Generate Procedural Episodes",
        "question_module": "src.datagen.synthetic.programs.runner",
        "question_dir_attr": "questions_procedural_dir",
        "episode_path_attr": "episodes_procedural_jsonl",
    },
    {
        "mode": "llm_gen",
        "question_stage": "Stage 1c: Generate LLM Questions",
        "episode_stage": "Stage 2c: Generate LLM Episodes",
        "question_module": "src.datagen.question_gen",
        "question_dir_attr": "questions_llm_gen_dir",
        "episode_path_attr": "episodes_llm_gen_jsonl",
    },
)


def _source_specs_for_mode(mode: str) -> list[dict[str, str]]:
    return [spec for spec in SOURCE_SPECS if mode in (spec["mode"], "all")]


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
    source_specs = _source_specs_for_mode(mode)
    selected_modes = {spec["mode"] for spec in source_specs}

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

    for spec in source_specs:
        question_cmd = ["uv", "run", "python", "-m", spec["question_module"]]
        if max_datasets:
            question_cmd.extend(["--max-datasets", str(max_datasets)])

        if run_stage(spec["question_stage"], question_cmd):
            stages_run += 1
        else:
            stages_failed += 1

        if spec["mode"] == "llm_gen":
            episode_cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "src.datagen.episode_gen",
                "--questions-dir",
                str(getattr(config, spec["question_dir_attr"])),
                "--output",
                str(getattr(config, spec["episode_path_attr"])),
                "--skip-difficulty-filter",
            ]
            if max_questions:
                episode_cmd.extend(["--max-questions", str(max_questions)])
            if n_consistency:
                episode_cmd.extend(["--n-consistency", str(n_consistency)])

            if run_stage(spec["episode_stage"], episode_cmd):
                stages_run += 1
            else:
                stages_failed += 1
            continue

        if run_synthetic_stage(
            spec["episode_stage"],
            questions_dir=str(getattr(config, spec["question_dir_attr"])),
            output_path=str(getattr(config, spec["episode_path_attr"])),
            max_questions=max_questions,
            source=spec["mode"],
        ):
            stages_run += 1
        else:
            stages_failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Stages run: {stages_run}")
    print(f"  Stages failed: {stages_failed}")

    if "template" in selected_modes:
        template_episodes = Path(config.episodes_template_jsonl)
        if template_episodes.exists():
            count = sum(1 for _ in open(template_episodes))
            print(f"  Template episodes: {count}")

    if "procedural" in selected_modes:
        procedural_episodes = Path(config.episodes_procedural_jsonl)
        if procedural_episodes.exists():
            count = sum(1 for _ in open(procedural_episodes))
            print(f"  Procedural episodes: {count}")

    if "llm_gen" in selected_modes:
        llm_episodes = Path(config.episodes_llm_gen_jsonl)
        if llm_episodes.exists():
            count = sum(1 for _ in open(llm_episodes))
            print(f"  LLM episodes: {count}")

    print()

    return 1 if stages_failed > 0 else 0
