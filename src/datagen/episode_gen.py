"""
Episode generation pipeline.

This module:
1. Loads questions from JSON files (one per dataset)
2. Runs teacher triangulation on each question
3. Saves verified episodes to disk

Usage (via CLI):
    csvagent generate episodes --llm     # LLM episodes
    csvagent generate episodes --synth   # Synthetic episodes
"""

import asyncio
import json
import sys
import signal
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from src.datagen.teacher import batch_triangulate
from src.datagen.pipeline_ui import EpisodeGenUI
from src.core.prompts import generate_data_overview
from csv_spec import (
    EpisodeJSONL,
    TimingMetadataDict,
    BatchTriangulationResult,
)
from src.core.config import config
from src.utils.docker import (
    cleanup_csv_sandbox_containers,
    cleanup_session,
    generate_session_id,
    check_resource_availability,
)
from src.envs.container_pool import ContainerPool
from src.datagen.manifest import (
    DatagenManifest,
    compute_dataset_hash,
    compute_llm_fingerprint,
)
from src.datagen.shared.questions_io import load_questions
from src.datagen.shared.dataset_meta import (
    load_dataset_meta,
    generate_description_from_overview,
)
from src.datagen.shared.episode_factory import create_episode
from src.datagen.shared.verification import VerificationResult


@dataclass
class CSVTask:
    """Represents a single CSV processing task."""

    csv_path: str
    dataset_name: str
    dataset_description: str
    questions: list[dict]
    questions_file: Path


def make_signal_handler(session_id: str):
    """Create a signal handler that cleans up only this session's containers."""

    def handler(signum, frame):
        print(f"\n\n🛑 Interrupted! Cleaning up session {session_id} containers...")
        cleanup_session(session_id)
        print("✓ Cleanup complete")
        sys.exit(0)

    return handler


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
    allocations = []
    base_total = 0

    for idx, (difficulty, fraction) in enumerate(distribution.items()):
        exact = total_target * fraction
        base = int(exact)
        fractional_part = exact - base  # Used for rounding remainder allocation
        allocations.append([idx, difficulty, base, fractional_part])
        base_total += base

    remainder = total_target - base_total
    if remainder > 0:
        allocations.sort(key=lambda row: (-row[3], row[0]))
        for i in range(remainder):
            allocations[i][2] += 1

    counts_by_difficulty = {difficulty: base for _, difficulty, base, _ in allocations}

    for difficulty, _fraction in distribution.items():
        count_needed = counts_by_difficulty.get(difficulty, 0)
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
        # Load dataset metadata using shared module
        csv_path_obj = Path(csv_path)
        dataset_name, dataset_description = load_dataset_meta(csv_path)
        csv_path_obj = Path(csv_path)

        # Generate description from data_overview if missing
        if not dataset_description or not dataset_description.strip():
            data_overview = generate_data_overview(str(csv_path))
            dataset_description = generate_description_from_overview(data_overview)
            ui.base.print_warning(
                f"{dataset_name}: No description found, synthesized from data_overview"
            )

        # Locate questions (structure: questions/[dataset_name]/questions.json)
        questions_file = base_questions_dir / dataset_name / "questions.json"

        if not questions_file.exists():
            ui.base.print_warning(
                f"Skipping {dataset_name}: No questions found at {questions_file}"
            )
            continue

        questions = load_questions(str(questions_file))

        # Filter by difficulty distribution (unless skipped)
        if not skip_difficulty_filter:
            filtered_questions, filter_ok = filter_by_difficulty(
                questions,
                config.question_difficulty_distribution,
                config.num_questions_to_generate,
            )
            if not filter_ok:
                # Use all available questions instead of skipping the CSV entirely
                ui.base.print_warning(
                    f"{dataset_name}: insufficient questions for target distribution "
                    f"(need {config.num_questions_to_generate} with {config.question_difficulty_distribution}). "
                    f"Using all {len(questions)} available questions instead."
                )
                # Don't filter - use all questions as-is
            else:
                questions = filtered_questions

        tasks.append(
            CSVTask(
                csv_path=csv_path,
                dataset_name=dataset_name,
                dataset_description=dataset_description,
                questions=questions,
                questions_file=questions_file,
            )
        )

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
    external_container: Any = None,
    manifest: DatagenManifest | None = None,
    dataset_hash: str | None = None,
) -> list[EpisodeJSONL]:
    """
    Process a single CSV task and return generated episodes.

    This is the core worker function for both sequential and parallel modes.

    Args:
        external_container: Optional pre-created container from ContainerPool.
                           If provided, the container is reused instead of created.
        manifest: Optional manifest for recording results
        dataset_hash: Optional dataset hash for fingerprinting
    """
    # Generate data overview
    data_overview = generate_data_overview(task.csv_path)

    # Run batch triangulation (uses external container if provided)
    results = await batch_triangulate(
        csv_path=task.csv_path,
        questions=task.questions,
        model=teacher_model,
        n_consistency=n_consistency,
        n_question_slots=config.n_question_slots,
        dynamic_triangulation=config.dynamic_triangulation,
        consistency_by_difficulty=config.triangulation_by_difficulty,
        dataset_description=task.dataset_description,
        data_overview=data_overview,
        max_turns=max_turns,
        sampling_args=sampling_args,
        external_container=external_container,
        ui=ui,
        float_tol=float_tol,
    )

    episodes = []
    failures = []
    for r in results:
        question_text = (
            r.question.get("question_text")
            or r.question.get("question_mechanical")
            or r.question.get("question", "")
        )

        # Compute fingerprint for manifest recording
        fingerprint = None
        if manifest is not None and dataset_hash is not None:
            fingerprint = compute_llm_fingerprint(question_text, dataset_hash)

        if not r.verified:
            # Log unverified for later analysis
            failures.append(
                {
                    "question": question_text[:100],
                    "gold_answer": r.gold_trace.get("final_answer")
                    if r.gold_trace
                    else None,
                    "gold_success": r.gold_trace.get("success")
                    if r.gold_trace
                    else False,
                    "majority_answer": r.majority_answer_hash,
                    "majority_count": r.majority_count,
                    "n_consistency": len(r.consistency_results),
                }
            )

            # Record failure to manifest
            if manifest is not None and fingerprint is not None:
                manifest.record_llm(
                    fingerprint=fingerprint,
                    status="failure",
                    dataset=task.dataset_name,
                    question_text=question_text,
                    model=teacher_model,
                    n_consistency=len(r.consistency_results),
                    elapsed_seconds=r.timing_metadata.get("avg_elapsed"),
                )

            if verified_only:
                continue

        # Build verification result from triangulation output
        consistency_traces = [trace for trace, _ in r.consistency_results]
        verification_result = VerificationResult(
            success=r.verified,
            match=r.verified,
            trace=r.gold_trace,
            traces=consistency_traces,
            majority_answer_hash=r.majority_answer_hash,
            error=None,
        )

        # Use episode factory to create episode
        episode = await create_episode(
            question=r.question,
            verification_result=verification_result,
            source="llm",
            csv_path=task.csv_path,
        )

        # Update timing with actual timing from triangulation
        episode.timing = TimingMetadataDict(
            gold_elapsed=r.timing_metadata["gold_elapsed"],
            consistency_elapsed=r.timing_metadata["consistency_elapsed"],
            total_elapsed=r.timing_metadata["total_elapsed"],
            avg_elapsed=r.timing_metadata["avg_elapsed"],
        )

        episodes.append(episode)

        # Record success to manifest (only for verified episodes)
        if r.verified and manifest is not None and fingerprint is not None:
            manifest.record_llm(
                fingerprint=fingerprint,
                status="success",
                dataset=task.dataset_name,
                question_text=question_text,
                episode_id=episode.episode_id,
                model=teacher_model,
                n_consistency=len(r.consistency_results),
                elapsed_seconds=r.timing_metadata.get("avg_elapsed"),
            )

    return episodes, failures


async def main(
    questions_dir: str | None = None,
    output_path: str | None = None,
    n_consistency: int | None = None,
    max_questions: int | None = None,
    skip_difficulty_filter: bool = False,
    difficulties: list[str] | None = None,
    skip_existing: set | None = None,
    retry_failed: bool = False,
):
    # Create global UI instance
    ui = EpisodeGenUI()

    # Generate session ID for container isolation
    session_id = generate_session_id()
    ui.base.print_status(f"Session ID: {session_id}")

    # Register signal handler for Ctrl+C (session-scoped cleanup)
    signal_handler = make_signal_handler(session_id)
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

    # Output as single JSONL file (must be specified explicitly)
    if output_path is None:
        ui.base.print_error(
            "ERROR: --output is required. Use one of:\n"
            f"  --output {config.episodes_synthetic_jsonl}  (for synthetic questions)\n"
            f"  --output {config.episodes_llm_jsonl}  (for LLM questions)"
        )
        return 1
    output_jsonl = Path(output_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Append mode if we're skipping existing, otherwise overwrite
    append_mode = skip_existing is not None and len(skip_existing) > 0
    if not append_mode and output_jsonl.exists():
        output_jsonl.unlink()

    # Get parent directory of questions (must be specified explicitly)
    if questions_dir is None:
        ui.base.print_error(
            "ERROR: --questions-dir is required. Use one of:\n"
            f"  --questions-dir {config.questions_synthetic_dir}  (for synthetic questions)\n"
            f"  --questions-dir {config.questions_llm_dir}  (for LLM questions)"
        )
        return 1
    base_questions_dir = Path(questions_dir)

    # Sampling args
    sampling_args = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Gather all valid CSV tasks
    tasks = gather_csv_tasks(
        csv_sources, base_questions_dir, ui, skip_difficulty_filter
    )

    # Filter by specific difficulties if requested
    if difficulties:
        allowed = set(d.upper() for d in difficulties)
        for task in tasks:
            task.questions = [
                q for q in task.questions if q.get("difficulty", "").upper() in allowed
            ]
        # Remove tasks with no questions after filtering
        tasks = [t for t in tasks if t.questions]

    # Load manifest for caching
    dataset_hashes: dict[str, str] = {}
    manifest = DatagenManifest()
    manifest.load()
    stats = manifest.stats()
    if stats["llm_total"] > 0:
        ui.base.print_status(
            f"Manifest loaded: {stats['llm_success']} success, "
            f"{stats['llm_failure']} failures"
        )

    # Filter questions using manifest
    original_total = sum(len(t.questions) for t in tasks)
    for task in tasks:
        # Compute dataset hash (cached)
        if task.csv_path not in dataset_hashes:
            dataset_hashes[task.csv_path] = compute_dataset_hash(task.csv_path)
        dataset_hash = dataset_hashes[task.csv_path]

        filtered_questions = []
        for q in task.questions:
            question_text = (
                q.get("question_text")
                or q.get("question_mechanical")
                or q.get("question", "")
            )
            fingerprint = compute_llm_fingerprint(question_text, dataset_hash)
            # include_failures=True means skip failures too (unless retry_failed)
            if manifest.has_llm(fingerprint, include_failures=not retry_failed):
                continue  # Skip - already processed
            filtered_questions.append(q)
        task.questions = filtered_questions

    # Remove tasks with no questions after filtering
    tasks = [t for t in tasks if t.questions]
    new_total = sum(len(t.questions) for t in tasks)
    if original_total > new_total:
        ui.base.print_status(f"Skipping {original_total - new_total} cached questions")

    # Legacy skip_existing support (for backward compatibility)
    if skip_existing:
        original_total = sum(len(t.questions) for t in tasks)
        for task in tasks:
            task.questions = [
                q
                for q in task.questions
                if q.get("id", q.get("ground_truth_hash", "")) not in skip_existing
            ]
        # Remove tasks with no questions after filtering
        tasks = [t for t in tasks if t.questions]
        new_total = sum(len(t.questions) for t in tasks)
        if original_total > new_total:
            ui.base.print_status(
                f"Skipping {original_total - new_total} already-processed questions"
            )

    # Limit questions per dataset if specified
    if max_questions is not None:
        for task in tasks:
            if len(task.questions) > max_questions:
                task.questions = task.questions[:max_questions]

    if not tasks:
        ui.base.print_error(
            "No valid CSV tasks found. Check questions and metadata files."
        )
        return 1

    # Cleanup any stale containers ONCE before starting (avoids race conditions in parallel mode)
    ui.base.print_status("Cleaning up old containers...")
    cleanup_csv_sandbox_containers()

    ui.base.print_section(f"Found {len(tasks)} CSV datasets to process")
    for task in tasks:
        ui.base.print_status(
            f"  • {task.dataset_name}: {len(task.questions)} questions"
        )

    all_episodes = []

    # Always use container pool for efficiency
    max_concurrent = config.max_concurrent_containers
    if config.dynamic_triangulation and config.triangulation_by_difficulty:
        from src.datagen.teacher import resolve_n_consistency

        max_consistency = n_consistency
        for task in tasks:
            for q in task.questions:
                max_consistency = max(
                    max_consistency,
                    resolve_n_consistency(
                        q, n_consistency, config.triangulation_by_difficulty
                    ),
                )
    else:
        max_consistency = n_consistency

    # Pre-flight resource check - auto-adjust to safe container count
    resource_status = check_resource_availability(max_concurrent)
    if resource_status.recommended_max_containers < 1:
        # No room at all - must wait
        ui.base.print_error(
            f"❌ Insufficient resources: {resource_status.available_memory_gb:.1f}GB available, "
            f"{resource_status.existing_containers} containers already running."
        )
        ui.base.print_error(
            "   Wait for other scripts to finish, or run: "
            'uv run python -c "from src.utils.docker import cleanup_csv_sandbox_containers; cleanup_csv_sandbox_containers()"'
        )
        return 1
    elif resource_status.recommended_max_containers < max_concurrent:
        # Reduce parallelism to safe level
        original = max_concurrent
        max_concurrent = resource_status.recommended_max_containers
        ui.base.print_warning(
            f"⚠️  Reducing parallelism: {original} → {max_concurrent} containers "
            f"({resource_status.available_memory_gb:.1f}GB available, "
            f"{resource_status.existing_containers} containers from other sessions)"
        )
    elif resource_status.existing_containers > 0:
        ui.base.print_status(
            f"Note: {resource_status.existing_containers} containers from other sessions, "
            f"{resource_status.available_memory_gb:.1f}GB available"
        )

    ui.base.print_section(
        f"Processing {len(tasks)} CSVs ({max_concurrent} containers pooled)"
    )

    # Create container pool (containers created once, reused across CSVs)
    pool = ContainerPool(
        max_containers=max_concurrent,
        n_question_slots=config.n_question_slots,
        n_consistency=max_consistency,
        session_id=session_id,
    )
    await pool.start(initial_csv_path=tasks[0].csv_path)

    async def process_task_wrapper(
        task: CSVTask,
    ) -> tuple[CSVTask, list[EpisodeJSONL], list[dict]]:
        # Acquire container from pool (blocks until one is available)
        container = await pool.acquire(task.csv_path)
        # Get dataset hash if manifest is active
        task_dataset_hash = dataset_hashes.get(task.csv_path) if manifest else None
        try:
            episodes, failures = await process_csv_task(
                task=task,
                teacher_model=teacher_model,
                n_consistency=n_consistency,
                max_turns=max_turns,
                sampling_args=sampling_args,
                float_tol=float_tol,
                verified_only=verified_only,
                ui=ui,
                external_container=container,
                manifest=manifest,
                dataset_hash=task_dataset_hash,
            )
            return (task, episodes, failures)
        finally:
            # Release container back to pool for reuse
            await pool.release(container)

    # Create tasks for asyncio.as_completed
    pending_tasks = [asyncio.create_task(process_task_wrapper(task)) for task in tasks]

    # Process as each completes (provides real-time progress)
    completed_count = 0
    all_failures = []
    had_error = False
    try:
        for coro in asyncio.as_completed(pending_tasks):
            task_result, episodes, failures = await coro
            completed_count += 1

            n_verified = sum(1 for ep in episodes if ep.verified)
            ui.base.print_success(
                f"✓ [{completed_count}/{len(tasks)}] {task_result.dataset_name}: "
                f"{len(episodes)} episodes ({n_verified} verified)"
            )
            all_episodes.extend(episodes)
            # Tag failures with dataset for analysis
            for f in failures:
                f["dataset"] = task_result.dataset_name
            all_failures.extend(failures)
    except Exception as e:
        had_error = True
        ui.base.print_error(f"ERROR: Episode generation failed: {e}")
    finally:
        # Cancel remaining tasks before stopping pool to avoid tearing down live workers
        for task in pending_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)

        # Stop the pool (destroys all containers)
        await pool.stop()

    if had_error:
        return 1

    # Write all episodes to output file
    total_verified = sum(1 for ep in all_episodes if ep.verified)

    write_mode = "a" if append_mode else "w"
    with open(output_jsonl, write_mode) as f:
        for episode in all_episodes:
            f.write(json.dumps(episode.model_dump(), default=str) + "\n")

    # Display final summary
    ui.base.print_section("PIPELINE COMPLETE")
    ui.base.print_key_value("Output file", str(output_jsonl))
    ui.base.print_key_value("Total sources", len(tasks))
    ui.base.print_key_value("Total episodes saved", len(all_episodes))
    ui.base.print_key_value("Total verified", total_verified)
    ui.base.print_key_value("Total unverified", len(all_failures))

    # Write failures to log file for later investigation
    if all_failures:
        failures_log = output_jsonl.parent / "failures_llm.jsonl"
        with open(failures_log, "w") as f:
            for failure in all_failures:
                f.write(json.dumps(failure, default=str) + "\n")
        ui.base.print_status(f"Failures logged to: {failures_log}")

    ui.base.print_empty_line()

    return 0
