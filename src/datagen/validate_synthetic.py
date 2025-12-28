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
    uv run python -m src.datagen.validate_synthetic \
        --questions-dir data/questions_synthetic \
        --output data/episodes/episodes_synthetic.jsonl

    # Parallel mode (process multiple datasets concurrently)
    uv run python -m src.datagen.validate_synthetic \
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

from src.datagen.teacher import answers_match
from datetime import datetime
import uuid
from collections import defaultdict
import time

from src.datagen.teacher import execute_teacher_trace
from src.datagen.pipeline_ui import EpisodeGenUI
from src.core.prompts import generate_data_overview
from csv_spec import (
    EpisodeJSONL,
    Question,
    QADict,
    TriangulationMetadataDict,
    TimingMetadataDict,
)
from src.core.config import config
from src.utils.docker import cleanup_csv_sandbox_containers, cleanup_session, generate_session_id
from csv_spec import hash_artifact
from src.gui.progress_writer import ProgressWriter, NoOpProgressWriter


def make_signal_handler(session_id: str):
    """Create a signal handler that cleans up only this session's containers."""
    def handler(signum, frame):
        print(f"\n\n🛑 Interrupted! Cleaning up session {session_id} containers...")
        cleanup_session(session_id)
        print("✓ Cleanup complete")
        sys.exit(0)
    return handler


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
    session_id: str | None = None,
) -> tuple[bool, dict | None, float, str | None]:
    """
    Validate a single synthetic question by running teacher trace.

    Returns:
        (success, trace_dict, elapsed_time, error_message)
    """
    start_time = time.time()

    try:
        # Run teacher trace WITH hint (gold mode)
        (
            trace,
            conversation,
            system_prompt,
            elapsed_seconds,
        ) = await execute_teacher_trace(
            csv_path=csv_path,
            question=question_dict.get("question", ""),
            model=teacher_model,
            hint=question_dict.get("hint"),  # Synthetic questions always use hint
            max_turns=max_turns,
            sampling_args=sampling_args,
            dataset_description=dataset_description,
            data_overview=data_overview,
            ui=ui,
            session_id=session_id,
        )

        elapsed = time.time() - start_time

        # Check if trace succeeded
        if not trace["success"]:
            return (
                False,
                trace,
                elapsed,
                f"Trace failed: {trace.get('error', 'unknown')}",
            )

        # Compare answer hash to ground truth (supports multiple valid answers)
        expected_hashes = question_dict.get("ground_truth_hashes") or [question_dict.get("ground_truth_hash")]
        expected_hashes = [h for h in expected_hashes if h is not None]
        if not expected_hashes:
            return False, trace, elapsed, "No ground_truth_hash in question"

        actual_hash = trace.get("final_answer_hash")
        actual_answer = trace.get("final_answer")

        # Fast path: exact hash match against any valid answer
        if actual_hash in expected_hashes:
            return True, trace, elapsed, None

        # Tolerant comparison: check against all valid answers
        expected_answers = question_dict.get("_ground_truths") or [question_dict.get("_ground_truth")]
        expected_answers = [a for a in expected_answers if a is not None]

        if actual_answer is not None:
            for exp_hash, exp_answer in zip(expected_hashes, expected_answers):
                if answers_match(
                    exp_hash,
                    actual_hash,
                    exp_answer,
                    actual_answer,
                    float_tol=config.float_tolerance,
                ):
                    return True, trace, elapsed, None

        # Debug: show what differed (use primary expected answer)
        exp_str = (
            json.dumps(expected_answers[0], default=str)[:200]
            if expected_answers
            else "None"
        )
        act_str = (
            json.dumps(actual_answer, default=str)[:200]
            if actual_answer
            else "None"
        )
        return (
            False,
            trace,
            elapsed,
            f"Answer mismatch:\n      Expected: {exp_str}\n      Got:      {act_str}",
        )

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
    session_id: str | None = None,
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
            session_id=session_id,
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
            failure_record = {
                "question": q_dict.get("question", "")[:100],
                "template_name": q_dict.get("template_name"),
                "variant_index": q_dict.get("variant_index"),
                "error": error,
                "elapsed": elapsed,
            }
            # Include full expected/actual for mismatch analysis
            if "Answer mismatch" in (error or ""):
                failure_record["expected"] = q_dict.get("_ground_truth")
                failure_record["actual"] = trace.get("final_answer") if trace else None
            failures.append(failure_record)
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
    session_id: str | None = None,
) -> tuple[str, list[EpisodeJSONL], list[dict]]:
    """
    Wrapper for process_dataset that respects semaphore for parallel execution.

    Returns:
        (dataset_name, episodes, failures)
    """
    if semaphore:
        async with semaphore:
            ui.base.print_section(
                f"Processing {dataset_name}: {len(questions)} questions"
            )
            episodes, failures = await process_dataset(
                csv_path=Path(csv_path),
                questions=questions,
                teacher_model=teacher_model,
                max_turns=max_turns,
                sampling_args=sampling_args,
                ui=ui,
                session_id=session_id,
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
            session_id=session_id,
        )
        return dataset_name, episodes, failures


async def main(
    questions_dir: str,
    output_path: str,
    max_questions: int | None = None,
    parallel: bool = False,
    n_workers: int = 4,
    gui_progress: str | None = None,
    skip_existing: set | None = None,
    difficulties: list[str] | None = None,
):
    ui = EpisodeGenUI()

    # Generate session ID for container isolation
    session_id = generate_session_id()
    ui.base.print_status(f"Session ID: {session_id}")

    # Register signal handler for Ctrl+C (session-scoped cleanup)
    signal_handler = make_signal_handler(session_id)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup progress writer for GUI
    if gui_progress:
        progress = ProgressWriter(output_path=gui_progress, stage="synthetic_episodes")
    else:
        progress = NoOpProgressWriter()

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
    ui.base.print_section(
        f"Found {len(question_files)} datasets with questions [{mode_str}]"
    )

    # Output file
    output_jsonl = Path(output_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Append mode if we're skipping existing, otherwise overwrite
    append_mode = skip_existing is not None and len(skip_existing) > 0
    if not append_mode and output_jsonl.exists():
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

        # Filter out already-processed questions
        if skip_existing:
            original_count = len(questions)
            questions = [
                q
                for q in questions
                if q.get("id", q.get("ground_truth_hash", "")) not in skip_existing
            ]
            skipped = original_count - len(questions)
            if skipped > 0:
                ui.base.console.print(
                    f"  [dim]{dataset_name}: skipping {skipped} already processed[/dim]"
                )

        # Filter by difficulty if specified
        if difficulties:
            allowed = {d.upper() for d in difficulties}
            questions = [q for q in questions if q.get("difficulty", "").upper() in allowed]

        if max_questions and len(questions) > max_questions:
            questions = questions[:max_questions]

        if not questions:
            continue  # Skip dataset if all questions already processed

        tasks_to_run.append((dataset_name, csv_path, questions))

    # Initialize progress tracking for GUI
    for name, _, questions in tasks_to_run:
        progress.set_dataset(name, len(questions))

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
                session_id=session_id,
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

            # Update progress for GUI
            progress.update_dataset(
                dataset_name,
                done=len(episodes) + len(failures),
                verified=len(episodes),
                failed=len(failures),
            )
            progress.log(
                f"✓ {dataset_name}: {len(episodes)} verified, {len(failures)} failed"
            )
    else:
        # Sequential execution
        for name, csv_path, questions in tasks_to_run:
            progress.set_current(name)
            _, episodes, failures = await process_dataset_task(
                dataset_name=name,
                csv_path=csv_path,
                questions=questions,
                teacher_model=teacher_model,
                max_turns=max_turns,
                sampling_args=sampling_args,
                ui=ui,
                session_id=session_id,
            )
            all_episodes.extend(episodes)
            all_failures.extend(failures)
            for f in failures:
                failure_by_template[f.get("template_name", "unknown")] += 1

            # Update progress for GUI
            progress.update_dataset(
                name,
                done=len(episodes) + len(failures),
                verified=len(episodes),
                failed=len(failures),
            )
            progress.log(f"✓ {name}: {len(episodes)} verified, {len(failures)} failed")

    # Write episodes
    write_mode = "a" if append_mode else "w"
    with open(output_jsonl, write_mode) as f:
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

        # Write failures to log file for later investigation
        failures_log = output_jsonl.parent / "failures_synthetic.jsonl"
        with open(failures_log, "w") as f:
            for failure in all_failures:
                f.write(json.dumps(failure, default=str) + "\n")
        ui.base.print_status(f"Failures logged to: {failures_log}")

    # Mark progress complete
    progress.log(
        f"Completed: {len(all_episodes)} episodes, {len(all_failures)} failures"
    )
    progress.complete()

    # Exit codes: 0=success, 1=partial success (some data), 2=total failure
    if len(all_episodes) == 0:
        return 2
    elif len(all_failures) > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic episode generation pipeline."
    )
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
    parser.add_argument(
        "--gui-progress",
        type=str,
        default=None,
        help="Path to write progress JSON for GUI polling",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific difficulties (e.g., HARD VERY_HARD)",
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
                    gui_progress=args.gui_progress,
                    difficulties=args.difficulties,
                )
            )
        )
    except KeyboardInterrupt:
        sys.exit(0)
