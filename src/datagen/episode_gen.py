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

def load_questions(questions_path: str) -> list[dict]:
    """Load questions from JSON file."""
    with open(questions_path) as f:
        return json.load(f)


def gather_csv_tasks(
    csv_sources: list[str],
    base_questions_dir: Path,
    legacy_mode: bool,
    ui: EpisodeGenUI,
) -> list[CSVTask]:
    """
    Gather all valid CSV tasks with their questions and metadata.

    Returns a list of CSVTask objects for CSVs that have valid questions and descriptions.
    """
    tasks = []

    for csv_path in csv_sources:
        dataset_name = Path(csv_path).stem

        # Determine dataset description (Sidecar Metadata only)
        dataset_description = None
        meta_path = Path(csv_path).with_suffix(".meta.json")
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

        # Locate questions (Modern structure: question/[dataset_name]/questions.json)
        questions_file = base_questions_dir / dataset_name / "questions.json"

        # Legacy fallback only if --legacy flag is provided
        if not questions_file.exists() and legacy_mode and len(csv_sources) == 1:
            legacy_path = Path(config.questions_json)
            if legacy_path.exists():
                questions_file = legacy_path
                ui.base.print_status(f"Using legacy flat file: {questions_file}")

        if not questions_file.exists():
            ui.base.print_warning(f"Skipping {dataset_name}: No questions found at {questions_file}")
            continue

        questions = load_questions(str(questions_file))

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
    ui: EpisodeGenUI | None = None,
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
    for q_dict, gold_trace, gold_conversation, system_prompt, consistency_results, verified, timing_metadata in results:
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
        )

        episodes.append(episode_jsonl)

    return episodes


async def main(legacy_mode: bool = False, parallel: bool = False):
    # Create global UI instance
    ui = EpisodeGenUI()
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # config is already imported from src.core.config
    teacher_model = config.teacher_model
    n_consistency = config.n_consistency
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
    output_jsonl = Path(config.episodes_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if output_jsonl.exists():
        output_jsonl.unlink()

    # Get parent directory of questions
    base_questions_dir = Path(config.questions_json).parent

    # Sampling args
    sampling_args = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Gather all valid CSV tasks
    tasks = gather_csv_tasks(csv_sources, base_questions_dir, legacy_mode, ui)

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
                    ui=None,  # No UI in parallel mode (avoids interleaved output)
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
    parser.add_argument("--legacy", action="store_true", help="Allow fallback to legacy flat questions file")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process multiple CSVs in parallel (one container per CSV)"
    )
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(main(legacy_mode=args.legacy, parallel=args.parallel)))
    except KeyboardInterrupt:
        # Already handled in signal_handler, but just in case
        sys.exit(0)
