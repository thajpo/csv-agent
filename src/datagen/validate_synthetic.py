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

import asyncio
import json
import sys
import signal
import argparse
from pathlib import Path
from collections import defaultdict
import time

from src.datagen.pipeline_ui import EpisodeGenUI
from src.core.prompts import generate_data_overview
from csv_spec import (
    EpisodeJSONL,
    TimingMetadataDict,
)
from src.core.config import config
from src.utils.docker import (
    cleanup_csv_sandbox_containers,
    cleanup_session,
    generate_session_id,
)
from src.gui.progress_writer import ProgressWriter, NoOpProgressWriter
from src.datagen.manifest import (
    DatagenManifest,
    compute_dataset_hash,
    compute_synthetic_fingerprint_from_question,
)
from src.datagen.shared.questions_io import load_questions
from src.datagen.shared.dataset_meta import (
    load_dataset_meta,
    generate_description_from_overview,
)
from src.datagen.shared.verification import verify_synthetic, VerificationResult
from src.datagen.shared.episode_factory import create_episode


def make_signal_handler(session_id: str):
    """Create a signal handler that cleans up only this session's containers."""

    def handler(signum, frame):
        print(f"\n\n🛑 Interrupted! Cleaning up session {session_id} containers...")
        cleanup_session(session_id)
        print("✓ Cleanup complete")
        sys.exit(0)

    return handler


def load_dataset_description(csv_path: Path) -> str:
    """Load dataset description using shared module (synthesize if missing)."""
    dataset_name, dataset_description = load_dataset_meta(str(csv_path))

    # Generate description from data_overview if missing
    if not dataset_description or not dataset_description.strip():
        data_overview = generate_data_overview(str(csv_path))
        dataset_description = generate_description_from_overview(data_overview)
        print(f"{dataset_name}: No description found, synthesized from data_overview")

    return dataset_description


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
    Uses shared verification module for consistency.

    Returns:
        (success, trace_dict, elapsed_time, error_message)
    """
    start_time = time.time()

    try:
        # Use shared verification module
        result = await verify_synthetic(
            question=question_dict,
            csv_path=csv_path,
            model=teacher_model,
            hint=question_dict.get("hint"),
            max_turns=max_turns,
            sampling_args=sampling_args,
            dataset_description=dataset_description,
            data_overview=data_overview,
            ui=ui,
            session_id=session_id,
            float_tol=config.float_tolerance,
        )

        elapsed = time.time() - start_time

        if not result.success:
            return False, result.trace, elapsed, result.error or "Verification failed"

        # result.match indicates if answer matches ground truth
        if result.match:
            return True, result.trace, elapsed, None
        else:
            # Answer mismatch - get debug info
            actual_answer = result.trace.get("final_answer") if result.trace else None
            act_str = (
                json.dumps(actual_answer, default=str)[:200]
                if actual_answer
                else "None"
            )
            return (
                False,
                result.trace,
                elapsed,
                f"Answer mismatch: got {act_str}",
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
    manifest: DatagenManifest | None = None,
    dataset_hash: str | None = None,
) -> tuple[list[EpisodeJSONL], list[dict]]:
    """
    Process all questions for a single dataset.

    Returns:
        (episodes, failures) - episodes are successful, failures are logged
    """
    dataset_description = load_dataset_description(csv_path)
    data_overview = generate_data_overview(str(csv_path))
    dataset_name = (
        csv_path.stem if csv_path.name != "data.csv" else csv_path.parent.name
    )

    episodes = []
    failures = []

    for i, q_dict in enumerate(questions, 1):
        question_preview = (
            q_dict.get("question_text") or q_dict.get("question_mechanical") or ""
        )[:60]
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

        # Compute fingerprint for manifest recording
        fingerprint = None
        if manifest is not None and dataset_hash is not None:
            fingerprint = compute_synthetic_fingerprint_from_question(
                q_dict, dataset_hash
            )

        if success and trace:
            # Build verification result from validation output
            verification_result = VerificationResult(
                success=True,
                match=True,
                trace=trace,
                traces=[],
                majority_answer_hash=trace.get("final_answer_hash"),
                error=None,
            )

            # Use episode factory to create episode
            episode = await create_episode(
                question=q_dict,
                verification_result=verification_result,
                source=q_dict.get("source", "template"),
                csv_path=str(csv_path),
            )

            # Update timing with actual elapsed time
            episode.timing = TimingMetadataDict(
                gold_elapsed=elapsed,
                consistency_elapsed=[],
                total_elapsed=elapsed,
                avg_elapsed=elapsed,
            )

            episodes.append(episode)
            ui.base.print_success(f"    ✓ Validated ({elapsed:.1f}s)")

            # Record success to manifest
            if manifest is not None and fingerprint is not None:
                manifest.record_synthetic(
                    fingerprint=fingerprint,
                    status="success",
                    dataset=dataset_name,
                    template_name=q_dict.get("template_name", "unknown"),
                    template_params=q_dict.get("template_params"),
                    episode_id=episode.episode_id,
                    model=teacher_model,
                    elapsed_seconds=elapsed,
                )
        else:
            failure_record = {
                "question": (
                    q_dict.get("question_text")
                    or q_dict.get("question_mechanical")
                    or ""
                )[:100],
                "template_name": q_dict.get("template_name"),
                "variant_index": q_dict.get("variant_index"),
                "error": error,
                "elapsed": elapsed,
            }
            # Include full expected/actual for mismatch analysis
            if "Answer mismatch" in (error or ""):
                failure_record["expected"] = q_dict.get("ground_truth")
                failure_record["actual"] = trace.get("final_answer") if trace else None
            failures.append(failure_record)
            ui.base.print_warning(f"    ✗ Failed: {error}")

            # Record failure to manifest
            if manifest is not None and fingerprint is not None:
                manifest.record_synthetic(
                    fingerprint=fingerprint,
                    status="failure",
                    dataset=dataset_name,
                    template_name=q_dict.get("template_name", "unknown"),
                    template_params=q_dict.get("template_params"),
                    model=teacher_model,
                    elapsed_seconds=elapsed,
                )

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
    manifest: DatagenManifest | None = None,
    dataset_hash: str | None = None,
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
                manifest=manifest,
                dataset_hash=dataset_hash,
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
            manifest=manifest,
            dataset_hash=dataset_hash,
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
    append_output: bool = False,
    difficulties: list[str] | None = None,
    retry_failed: bool = False,
    source: str | None = None,
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

    # Append mode can be explicit, or inferred from skip-existing semantics.
    append_mode = append_output or (
        skip_existing is not None and len(skip_existing) > 0
    )
    if not append_mode and output_jsonl.exists():
        output_jsonl.unlink()

    # Load manifest for caching
    manifest = DatagenManifest()
    manifest.load()
    stats = manifest.stats()
    if stats["synthetic_total"] > 0:
        ui.base.print_status(
            f"Manifest loaded: {stats['synthetic_success']} success, "
            f"{stats['synthetic_failure']} failures"
        )

    # Cache dataset hashes to avoid recomputing
    dataset_hashes: dict[str, str] = {}

    # Prepare tasks
    tasks_to_run = []
    for qf in question_files:
        dataset_name = qf.parent.name
        csv_path = csv_by_name.get(dataset_name)

        if not csv_path:
            ui.base.print_warning(f"Skipping {dataset_name}: no matching CSV found")
            continue

        questions = load_questions(str(qf))
        if source:
            source_filters = {
                "template": lambda q: q.get("source") == "template",
                "procedural": lambda q: q.get("source") == "procedural",
            }
            questions = [q for q in questions if source_filters[source](q)]

        # Filter out already-processed questions using manifest
        # Compute dataset hash (cached per dataset)
        if csv_path not in dataset_hashes:
            dataset_hashes[csv_path] = compute_dataset_hash(csv_path)
        dataset_hash = dataset_hashes[csv_path]

        original_count = len(questions)
        filtered_questions = []
        for q in questions:
            fingerprint = compute_synthetic_fingerprint_from_question(q, dataset_hash)
            if fingerprint is None:
                # Can't compute fingerprint (no template_name), include it
                filtered_questions.append(q)
                continue
            # Check manifest - include_failures=True means skip failures too (unless retry_failed)
            if manifest.has_synthetic(fingerprint, include_failures=not retry_failed):
                continue  # Skip - already processed
            filtered_questions.append(q)
        questions = filtered_questions
        skipped = original_count - len(questions)
        if skipped > 0:
            ui.base.console.print(
                f"  [dim]{dataset_name}: skipping {skipped} cached[/dim]"
            )

        # Filter by difficulty if specified
        if difficulties:
            allowed = {d.upper() for d in difficulties}
            questions = [
                q for q in questions if q.get("difficulty", "").upper() in allowed
            ]

        if max_questions and len(questions) > max_questions:
            questions = questions[:max_questions]

        if not questions:
            continue  # Skip dataset if all questions already processed

        # Get dataset hash for this task (for manifest recording)
        task_dataset_hash = dataset_hashes.get(csv_path)
        tasks_to_run.append((dataset_name, csv_path, questions, task_dataset_hash))

    # Initialize progress tracking for GUI
    for name, _, questions, _ in tasks_to_run:
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
                manifest=manifest,
                dataset_hash=task_hash,
            )
            for name, csv_path, questions, task_hash in tasks_to_run
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
        for name, csv_path, questions, task_hash in tasks_to_run:
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
                manifest=manifest,
                dataset_hash=task_hash,
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
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry questions that previously failed validation",
    )
    parser.add_argument(
        "--source",
        choices=["template", "procedural"],
        default=None,
        help="Only process questions for this source",
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
                    retry_failed=args.retry_failed,
                    source=args.source,
                )
            )
        )
    except KeyboardInterrupt:
        sys.exit(0)
