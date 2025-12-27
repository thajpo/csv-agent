"""
Synthetic episode generation pipeline.

This script validates synthetic questions by running a single teacher trace:
1. Loads questions from questions_synthetic_dir
2. Runs 1 teacher trace per question (WITH hint)
3. Validates answer matches ground_truth_hash
4. Saves successful episodes to disk, logs failures

Unlike episode_gen.py, this skips triangulation since synthetic questions
have known ground truth from template execution.

Usage:
    uv run python -m src.datagen.synthetic_episodes \
        --questions-dir data/questions_synthetic \
        --output data/episodes/episodes_synthetic.jsonl

    # Parallel mode (process multiple datasets concurrently)
    uv run python -m src.datagen.synthetic_episodes \
        --questions-dir data/questions_synthetic \
        --output data/episodes/episodes_synthetic.jsonl \
        --parallel --n-workers 4
"""

import asyncio
import json
import sys
import signal
import argparse
from pathlib import Path
from datetime import datetime
import uuid
from collections import defaultdict
import time

from src.datagen.teacher import execute_teacher_trace
from src.datagen.ui import EpisodeGenUI
from src.core.prompts import generate_data_overview
from src.core.types import (
    EpisodeJSONL,
    Question,
    QADict,
    TriangulationMetadataDict,
    TimingMetadataDict,
)
from src.core.config import config
from src.utils.docker import cleanup_csv_sandbox_containers
from src.utils.hashing import hash_artifact


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🛑 Interrupted! Cleaning up containers...")
    cleanup_csv_sandbox_containers()
    print("✓ Cleanup complete")
    sys.exit(0)


def load_questions(questions_path: str) -> tuple[list[dict], list[str] | None]:
    """
    Load questions from JSON file.

    Returns:
        (questions, expected_columns) - expected_columns is None for legacy format
    """
    with open(questions_path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        return data["questions"], data.get("dataset_columns")

    return data, None


def load_dataset_description(csv_path: Path) -> str:
    """Load dataset description from meta.json."""
    meta_path = csv_path.parent / "meta.json"
    if not meta_path.exists():
        meta_path = csv_path.with_suffix(".meta.json")

    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                return (
                    meta.get("description")
                    or meta.get("subtitle")
                    or meta.get("title")
                    or ""
                )
        except Exception:
            pass
    return ""


async def validate_single_question(
    csv_path: str,
    question_dict: dict,
    teacher_model: str,
    max_turns: int,
    sampling_args: dict,
    dataset_description: str,
    data_overview: str,
    ui: EpisodeGenUI,
) -> tuple[bool, dict | None, float, str | None]:
    """
    Validate a single synthetic question by running teacher trace.

    Returns:
        (success, trace_dict, elapsed_time, error_message)
    """
    start_time = time.time()

    try:
        # Run teacher trace WITH hint (gold mode)
        trace, conversation, system_prompt, elapsed_seconds = await execute_teacher_trace(
            csv_path=csv_path,
            question=question_dict.get("question", ""),
            model=teacher_model,
            hint=question_dict.get("hint"),  # Synthetic questions always use hint
            max_turns=max_turns,
            sampling_args=sampling_args,
            dataset_description=dataset_description,
            data_overview=data_overview,
            ui=ui,
        )

        elapsed = time.time() - start_time

        # Check if trace succeeded
        if not trace["success"]:
            return False, trace, elapsed, f"Trace failed: {trace.get('error', 'unknown')}"

        # Compare answer hash to ground truth
        expected_hash = question_dict.get("ground_truth_hash")
        if expected_hash is None:
            return False, trace, elapsed, "No ground_truth_hash in question"

        actual_hash = trace.get("final_answer_hash")
        if actual_hash != expected_hash:
            expected_answer = question_dict.get("_ground_truth")
            actual_answer = trace.get("final_answer")

            # Try tolerant comparison if hashes don't match exactly
            if expected_answer is not None and actual_answer is not None:
                # Simple float tolerance check
                try:
                    if isinstance(expected_answer, (int, float)) and isinstance(actual_answer, (int, float)):
                        if abs(expected_answer - actual_answer) <= config.float_tolerance:
                            return True, trace, elapsed, None
                except (TypeError, ValueError):
                    pass

            # Debug: show what differed
            import json
            exp_str = json.dumps(expected_answer, default=str)[:200] if expected_answer else "None"
            act_str = json.dumps(actual_answer, default=str)[:200] if actual_answer else "None"
            return False, trace, elapsed, f"Answer mismatch:\n      Expected: {exp_str}\n      Got:      {act_str}"

        return True, trace, elapsed, None

    except Exception as e:
        elapsed = time.time() - start_time
        return False, None, elapsed, str(e)


async def process_dataset(
    csv_path: Path,
    questions: list[dict],
    teacher_model: str,
    max_turns: int,
    sampling_args: dict,
    ui: EpisodeGenUI,
) -> tuple[list[EpisodeJSONL], list[dict]]:
    """
    Process all questions for a single dataset.

    Returns:
        (episodes, failures) - episodes are successful, failures are logged
    """
    dataset_description = load_dataset_description(csv_path)
    data_overview = generate_data_overview(str(csv_path))

    episodes = []
    failures = []

    for i, q_dict in enumerate(questions, 1):
        question_preview = q_dict.get("question", "")[:60]
        ui.base.print_status(f"  [{i}/{len(questions)}] {question_preview}...")

        success, trace, elapsed, error = await validate_single_question(
            csv_path=str(csv_path),
            question_dict=q_dict,
            teacher_model=teacher_model,
            max_turns=max_turns,
            sampling_args=sampling_args,
            dataset_description=dataset_description,
            data_overview=data_overview,
            ui=ui,
        )

        if success and trace:
            question_obj = Question.from_dict(q_dict)

            episode = EpisodeJSONL(
                episode_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                csv_source=str(csv_path),
                question=QADict(
                    id=question_obj.id,
                    question_text=question_obj.question_text,
                    hint=question_obj.hint,
                    difficulty=question_obj.difficulty,
                    n_steps=question_obj.n_steps,
                    template_name=question_obj.template_name,
                    template_params=question_obj.template_params,
                    output_type=question_obj.output_type,
                    output_schema=question_obj.output_schema,
                    ground_truth_hash=question_obj.ground_truth_hash,
                    ground_truth=question_obj.ground_truth,
                ),
                gold_trace=trace,
                consistency_traces=[],  # No consistency for synthetic
                verified=True,
                triangulation=TriangulationMetadataDict(
                    n_consistency_runs=0,
                    n_consistency_succeeded=0,
                    majority_answer_hash=None,
                    majority_count=0,
                    gold_matches_majority=True,
                ),
                timing=TimingMetadataDict(
                    gold_elapsed=elapsed,
                    consistency_elapsed=[],
                    total_elapsed=elapsed,
                    avg_elapsed=elapsed,
                ),
                source="synthetic",
            )
            episodes.append(episode)
            ui.base.print_success(f"    ✓ Validated ({elapsed:.1f}s)")
        else:
            failures.append({
                "question": q_dict.get("question", "")[:100],
                "template_name": q_dict.get("template_name"),
                "variant_index": q_dict.get("variant_index"),
                "error": error,
                "elapsed": elapsed,
            })
            ui.base.print_warning(f"    ✗ Failed: {error}")

    return episodes, failures


async def process_dataset_task(
    dataset_name: str,
    csv_path: str,
    questions: list[dict],
    teacher_model: str,
    max_turns: int,
    sampling_args: dict,
    ui: EpisodeGenUI,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[str, list[EpisodeJSONL], list[dict]]:
    """
    Wrapper for process_dataset that respects semaphore for parallel execution.

    Returns:
        (dataset_name, episodes, failures)
    """
    if semaphore:
        async with semaphore:
            ui.base.print_section(f"Processing {dataset_name}: {len(questions)} questions")
            episodes, failures = await process_dataset(
                csv_path=Path(csv_path),
                questions=questions,
                teacher_model=teacher_model,
                max_turns=max_turns,
                sampling_args=sampling_args,
                ui=ui,
            )
            return dataset_name, episodes, failures
    else:
        ui.base.print_section(f"Processing {dataset_name}: {len(questions)} questions")
        episodes, failures = await process_dataset(
            csv_path=Path(csv_path),
            questions=questions,
            teacher_model=teacher_model,
            max_turns=max_turns,
            sampling_args=sampling_args,
            ui=ui,
        )
        return dataset_name, episodes, failures


async def main(
    questions_dir: str,
    output_path: str,
    max_questions: int | None = None,
    parallel: bool = False,
    n_workers: int = 4,
):
    ui = EpisodeGenUI()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    teacher_model = config.teacher_model
    max_turns = config.max_turns
    sampling_args = {
        "temperature": config.sampling_args.temperature,
        "max_tokens": config.sampling_args.max_tokens,
    }

    # Find all question files
    questions_dir_path = Path(questions_dir)
    if not questions_dir_path.exists():
        ui.base.print_error(f"Questions directory not found: {questions_dir}")
        return 1

    # Get CSV sources from config
    csv_sources = config.csv_sources
    if isinstance(csv_sources, str):
        csv_sources = [csv_sources]

    # Build mapping: dataset_name -> csv_path
    csv_by_name = {}
    for csv_path in csv_sources:
        csv_path_obj = Path(csv_path)
        if csv_path_obj.name == "data.csv":
            dataset_name = csv_path_obj.parent.name
        else:
            dataset_name = csv_path_obj.stem
        csv_by_name[dataset_name] = csv_path

    # Find question files
    question_files = list(questions_dir_path.glob("*/questions.json"))
    if not question_files:
        ui.base.print_error(f"No questions found in {questions_dir}")
        return 1

    mode_str = f"parallel ({n_workers} workers)" if parallel else "sequential"
    ui.base.print_section(f"Found {len(question_files)} datasets with questions [{mode_str}]")

    # Output file
    output_jsonl = Path(output_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.unlink()

    # Prepare tasks
    tasks_to_run = []
    for qf in question_files:
        dataset_name = qf.parent.name
        csv_path = csv_by_name.get(dataset_name)

        if not csv_path:
            ui.base.print_warning(f"Skipping {dataset_name}: no matching CSV found")
            continue

        questions, _ = load_questions(str(qf))

        if max_questions and len(questions) > max_questions:
            questions = questions[:max_questions]

        tasks_to_run.append((dataset_name, csv_path, questions))

    all_episodes = []
    all_failures = []
    failure_by_template = defaultdict(int)

    if parallel and len(tasks_to_run) > 1:
        # Parallel execution with semaphore
        semaphore = asyncio.Semaphore(n_workers)
        coros = [
            process_dataset_task(
                dataset_name=name,
                csv_path=csv_path,
                questions=questions,
                teacher_model=teacher_model,
                max_turns=max_turns,
                sampling_args=sampling_args,
                ui=ui,
                semaphore=semaphore,
            )
            for name, csv_path, questions in tasks_to_run
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                ui.base.print_error(f"Dataset failed with exception: {result}")
                continue
            dataset_name, episodes, failures = result
            all_episodes.extend(episodes)
            all_failures.extend(failures)
            for f in failures:
                failure_by_template[f.get("template_name", "unknown")] += 1
    else:
        # Sequential execution
        for name, csv_path, questions in tasks_to_run:
            _, episodes, failures = await process_dataset_task(
                dataset_name=name,
                csv_path=csv_path,
                questions=questions,
                teacher_model=teacher_model,
                max_turns=max_turns,
                sampling_args=sampling_args,
                ui=ui,
            )
            all_episodes.extend(episodes)
            all_failures.extend(failures)
            for f in failures:
                failure_by_template[f.get("template_name", "unknown")] += 1

    # Write episodes
    with open(output_jsonl, "w") as f:
        for episode in all_episodes:
            f.write(json.dumps(episode.model_dump(), default=str) + "\n")

    # Summary
    ui.base.print_section("SUMMARY")
    ui.base.print_key_value("Output file", str(output_jsonl))
    ui.base.print_key_value("Total episodes", len(all_episodes))
    ui.base.print_key_value("Total failures", len(all_failures))

    if all_failures:
        ui.base.print_empty_line()
        ui.base.print_status("Failures by template:")
        for template, count in sorted(failure_by_template.items(), key=lambda x: -x[1]):
            ui.base.print_status(f"  {template}: {count}")

    return 0 if not all_failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic episode generation pipeline.")
    parser.add_argument(
        "--questions-dir",
        type=str,
        required=True,
        help="Directory containing question files (e.g., data/questions_synthetic)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path (e.g., data/episodes/episodes_synthetic.jsonl)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Max questions per dataset (for testing)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process multiple datasets in parallel",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    try:
        sys.exit(
            asyncio.run(
                main(
                    questions_dir=args.questions_dir,
                    output_path=args.output,
                    max_questions=args.max_questions,
                    parallel=args.parallel,
                    n_workers=args.n_workers,
                )
            )
        )
    except KeyboardInterrupt:
        sys.exit(0)
