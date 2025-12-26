"""
Episode generation pipeline.

This script:
1. Loads questions from JSON files (one per dataset)
2. Runs teacher triangulation on each question
3. Saves verified episodes to disk

Usage:
    python -m src.datagen.episode_gen           # Sequential processing
    python -m src.datagen.episode_gen --parallel  # Parallel CSV processing
"""
import asyncio
import json
import sys
import signal
import argparse
from pathlib import Path
from datetime import datetime
import uuid
from typing import Any
from dataclasses import dataclass

from src.datagen.teacher import batch_triangulate
from src.datagen.ui import EpisodeGenUI
from src.core.prompts import generate_data_overview
from src.core.types import Episode, EpisodeJSONL, Question, ExecutionTrace
from src.core.config import config
from src.utils.docker import cleanup_csv_sandbox_containers


@dataclass
class CSVTask:
    """Represents a single CSV processing task."""
    csv_path: str
    dataset_name: str
    dataset_description: str
    questions: list[dict]
    questions_file: Path

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

    # New format: {"dataset_columns": [...], "questions": [...]}
    if isinstance(data, dict) and "questions" in data:
        return data["questions"], data.get("dataset_columns")

    # Legacy format: plain list of questions
    return data, None


def filter_by_difficulty(
    questions: list[dict],
    distribution: dict[str, float],
    total_target: int,
) -> tuple[list[dict], bool]:
    """
    Select questions matching target difficulty distribution.

    Args:
        questions: All questions from questions.json
        distribution: {difficulty: fraction} e.g., {"EASY": 0.30, ...}
        total_target: Total questions desired

    Returns:
        (filtered_questions, success)
        success=False if any difficulty has insufficient questions
    """
    result = []
    for difficulty, fraction in distribution.items():
        count_needed = round(total_target * fraction)
        matching = [q for q in questions if q.get("difficulty") == difficulty]
        if len(matching) < count_needed:
            return [], False  # Insufficient questions for this difficulty
        result.extend(matching[:count_needed])  # First N (deterministic)
    return result, True


def gather_csv_tasks(
    csv_sources: list[str],
    base_questions_dir: Path,
    ui: EpisodeGenUI,
    skip_difficulty_filter: bool = False,
) -> list[CSVTask]:
    """
    Gather all valid CSV tasks with their questions and metadata.

    Returns a list of CSVTask objects for CSVs that have valid questions and descriptions.
    """
    tasks = []

    for csv_path in csv_sources:
        # Derive dataset name (must match question_gen.py logic)
        csv_path_obj = Path(csv_path)
        if csv_path_obj.name == "data.csv":
            dataset_name = csv_path_obj.parent.name
        else:
            dataset_name = csv_path_obj.stem

        # Determine dataset description (sibling meta.json or sidecar {name}.meta.json)
        dataset_description = None
        meta_path = csv_path_obj.parent / "meta.json"
        if not meta_path.exists():
            meta_path = csv_path_obj.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta_data = json.load(f)
                    dataset_description = meta_data.get("description") or meta_data.get("subtitle")
            except Exception as e:
                ui.base.print_warning(f"Failed to read metadata from {meta_path}: {e}")

        if not dataset_description or not dataset_description.strip():
            ui.base.print_error(f"ERROR: No description found for {dataset_name}")
            ui.base.print_info("Hint", f"Create {dataset_name}.meta.json with a 'description' field.")
            continue

        # Locate questions (structure: questions/[dataset_name]/questions.json)
        questions_file = base_questions_dir / dataset_name / "questions.json"

        if not questions_file.exists():
            ui.base.print_warning(f"Skipping {dataset_name}: No questions found at {questions_file}")
            continue

        questions, expected_columns = load_questions(str(questions_file))

        # Validate dataset columns match (prevents running questions against wrong dataset)
        if expected_columns is not None:
            import pandas as pd
            actual_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
            if set(expected_columns) != set(actual_columns):
                missing = set(expected_columns) - set(actual_columns)
                extra = set(actual_columns) - set(expected_columns)
                ui.base.print_error(f"ERROR: Column mismatch for {dataset_name}")
                if missing:
                    ui.base.print_error(f"  Missing columns: {sorted(missing)}")
                if extra:
                    ui.base.print_warning(f"  Extra columns: {sorted(extra)}")
                ui.base.print_info("Hint", "Regenerate questions with: uv run python -m src.datagen.question_gen")
                continue

        # Filter by difficulty distribution (unless skipped)
        if not skip_difficulty_filter:
            filtered_questions, filter_ok = filter_by_difficulty(
                questions,
                config.question_difficulty_distribution,
                config.num_questions_to_generate,
            )
            if not filter_ok:
                ui.base.print_warning(
                    f"Skipping {dataset_name}: insufficient questions for difficulty distribution "
                    f"(need {config.num_questions_to_generate} total with distribution {config.question_difficulty_distribution})"
                )
                continue
            questions = filtered_questions

        tasks.append(CSVTask(
            csv_path=csv_path,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            questions=questions,
            questions_file=questions_file,
        ))

    return tasks


async def process_csv_task(
    task: CSVTask,
    teacher_model: str,
    n_consistency: int,
    max_turns: int,
    sampling_args: dict,
    float_tol: float,
    verified_only: bool,
    ui: EpisodeGenUI,
) -> list[EpisodeJSONL]:
    """
    Process a single CSV task and return generated episodes.

    This is the core worker function for both sequential and parallel modes.
    Each CSV gets its own MultiTenantContainer with fork-based workers.
    """
    # Generate data overview
    data_overview = generate_data_overview(task.csv_path)

    # Run batch triangulation (creates container internally)
    results = await batch_triangulate(
        csv_path=task.csv_path,
        questions=task.questions,
        model=teacher_model,
        n_consistency=n_consistency,
        dataset_description=task.dataset_description,
        data_overview=data_overview,
        max_turns=max_turns,
        sampling_args=sampling_args,
        ui=ui,
        float_tol=float_tol,
    )

    # Convert results to EpisodeJSONL objects
    episodes = []
    for (
        q_dict,
        gold_trace,
        gold_conversation,
        system_prompt,
        consistency_results,
        verified,
        timing_metadata,
        majority_answer_hash,
        majority_count,
    ) in results:
        # Skip unverified if verified_only is True
        if verified_only and not verified:
            continue

        # Create Question object with auto-generated ID
        question_obj = Question.from_dict(q_dict)

        # Extract consistency traces
        consistency_traces = [trace for trace, _ in consistency_results]
        consistency_conversations = [conv for _, conv in consistency_results]

        # Create Episode object
        episode = Episode(
            id=str(uuid.uuid4()),
            question=question_obj,
            teacher_trace=gold_trace,
            consistency_traces=consistency_traces,
            verified=verified,
            timestamp=datetime.now(),
        )

        # Convert to JSONL format
        episode_jsonl = EpisodeJSONL.from_episode(
            episode=episode,
            gold_conversation=gold_conversation,
            system_prompt=system_prompt,
            consistency_conversations=consistency_conversations,
            csv_source=task.csv_path,
            timing_metadata=timing_metadata,
            majority_answer_hash=majority_answer_hash,
            majority_count=majority_count,
        )

        episodes.append(episode_jsonl)

    return episodes


async def main(
    parallel: bool = False,
    questions_dir: str | None = None,
    output_path: str | None = None,
    n_consistency: int | None = None,
    max_questions: int | None = None,
    skip_difficulty_filter: bool = False,
    difficulties: list[str] | None = None,
):
    # Create global UI instance
    ui = EpisodeGenUI()
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # config is already imported from src.core.config
    teacher_model = config.teacher_model
    n_consistency = n_consistency if n_consistency is not None else config.n_consistency
    max_turns = config.max_turns
    float_tol = config.float_tolerance
    verified_only = config.verified_only
    temperature = config.sampling_args.temperature
    max_tokens = config.sampling_args.max_tokens

    # Handle single csv or list of csvs
    csv_sources = config.csv_sources
    if isinstance(csv_sources, str):
        csv_sources = [csv_sources]

    # Output as single JSONL file
    output_jsonl = Path(output_path) if output_path else Path(config.episodes_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if output_jsonl.exists():
        output_jsonl.unlink()

    # Get parent directory of questions
    base_questions_dir = Path(questions_dir) if questions_dir else Path(config.questions_json).parent

    # Sampling args
    sampling_args = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Gather all valid CSV tasks
    tasks = gather_csv_tasks(csv_sources, base_questions_dir, ui, skip_difficulty_filter)

    # Filter by specific difficulties if requested
    if difficulties:
        allowed = set(d.upper() for d in difficulties)
        for task in tasks:
            task.questions = [q for q in task.questions if q.get("difficulty", "").upper() in allowed]
        # Remove tasks with no questions after filtering
        tasks = [t for t in tasks if t.questions]

    # Limit questions per dataset if specified
    if max_questions is not None:
        for task in tasks:
            if len(task.questions) > max_questions:
                task.questions = task.questions[:max_questions]

    if not tasks:
        ui.base.print_error("No valid CSV tasks found. Check questions and metadata files.")
        return 1

    ui.base.print_section(f"Found {len(tasks)} CSV datasets to process")
    for task in tasks:
        ui.base.print_status(f"  • {task.dataset_name}: {len(task.questions)} questions")

    all_episodes = []

    if parallel and len(tasks) > 1:
        # Parallel mode: process CSVs concurrently with throttling
        max_concurrent = config.max_concurrent_containers
        ui.base.print_section(f"🚀 PARALLEL MODE: Processing {len(tasks)} CSVs (max {max_concurrent} concurrent)")
        ui.base.print_info("Note", "Each CSV gets its own container with fork-based workers")

        # Semaphore limits concurrent containers to avoid resource exhaustion
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_task_wrapper(task: CSVTask) -> tuple[CSVTask, list[EpisodeJSONL]]:
            async with semaphore:
                episodes = await process_csv_task(
                    task=task,
                    teacher_model=teacher_model,
                    n_consistency=n_consistency,
                    max_turns=max_turns,
                    sampling_args=sampling_args,
                    float_tol=float_tol,
                    verified_only=verified_only,
                    ui=ui,  # Output may interleave in parallel mode
                )
                return (task, episodes)

        results = await asyncio.gather(*[
            process_task_wrapper(task) for task in tasks
        ])

        # Aggregate results
        for task, episodes in results:
            n_verified = sum(1 for ep in episodes if ep.verified)
            ui.base.print_success(f"✓ {task.dataset_name}: {len(episodes)} episodes ({n_verified} verified)")
            all_episodes.extend(episodes)

    else:
        # Sequential mode: process CSVs one at a time with full UI
        for i, task in enumerate(tasks, 1):
            ui.base.print_section(f"Processing CSV {i}/{len(tasks)}: {task.csv_path}")
            ui.base.print_status(f"Loaded {len(task.questions)} questions")

            # Display pipeline header for this CSV
            ui.print_pipeline_header(
                n_questions=len(task.questions),
                n_consistency=n_consistency,
                csv_path=task.csv_path,
                model=teacher_model,
                float_tol=float_tol,
                output_file=str(output_jsonl)
            )

            episodes = await process_csv_task(
                task=task,
                teacher_model=teacher_model,
                n_consistency=n_consistency,
                max_turns=max_turns,
                sampling_args=sampling_args,
                float_tol=float_tol,
                verified_only=verified_only,
                ui=ui,
            )

            n_verified = sum(1 for ep in episodes if ep.verified)
            ui.base.print_success(f"✓ Saved {len(episodes)} episodes for {task.dataset_name} ({n_verified} verified)")
            all_episodes.extend(episodes)

    # Write all episodes to output file
    total_verified = sum(1 for ep in all_episodes if ep.verified)

    with open(output_jsonl, 'w') as f:
        for episode in all_episodes:
            f.write(json.dumps(episode.model_dump(), default=str) + '\n')

    # Display final summary
    ui.base.print_section("PIPELINE COMPLETE")
    ui.base.print_key_value("Output file", str(output_jsonl))
    ui.base.print_key_value("Total sources", len(tasks))
    ui.base.print_key_value("Total episodes saved", len(all_episodes))
    ui.base.print_key_value("Total verified", total_verified)

    ui.base.print_empty_line()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Episode generation pipeline.")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process multiple CSVs in parallel (one container per CSV)"
    )
    parser.add_argument(
        "--questions-dir",
        type=str,
        default=None,
        help="Base directory for questions (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: from config)"
    )
    parser.add_argument(
        "--n-consistency",
        type=int,
        default=None,
        help="Number of consistency traces (default: from config)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Max questions per dataset (default: all)"
    )
    parser.add_argument(
        "--skip-difficulty-filter",
        action="store_true",
        help="Skip difficulty distribution filtering (use all questions)"
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        nargs="+",
        default=None,
        help="Only include these difficulties (e.g., --difficulties HARD VERY_HARD)"
    )
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(main(
            parallel=args.parallel,
            questions_dir=args.questions_dir,
            output_path=args.output,
            n_consistency=args.n_consistency,
            max_questions=args.max_questions,
            skip_difficulty_filter=args.skip_difficulty_filter,
            difficulties=args.difficulties,
        )))
    except KeyboardInterrupt:
        # Already handled in signal_handler, but just in case
        sys.exit(0)
