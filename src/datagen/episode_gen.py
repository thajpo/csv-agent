"""Unified episode generation pipeline.

This module generates episodes for all question sources:
- llm_gen: consistency triangulation
- template: single-trace ground-truth verification
- procedural: single-trace ground-truth verification
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from csv_spec import EpisodeJSONL, TimingMetadataDict

from src.core.config import config
from src.core.prompts import generate_data_overview
from src.datagen.manifest import (
    DatagenManifest,
    compute_dataset_hash,
    compute_llm_fingerprint,
    compute_synthetic_fingerprint_from_question,
)
from src.datagen.pipeline_ui import EpisodeGenUI
from src.datagen.shared.dataset_meta import (
    generate_description_from_overview,
    load_dataset_meta,
)
from src.datagen.shared.episode_factory import create_episode
from src.datagen.shared.questions_io import load_questions
from src.datagen.shared.verification import (
    VerificationResult,
    resolve_question_prompt,
    verify_question,
)
from src.datagen.teacher import batch_triangulate
from src.envs.container_pool import ContainerPool
from src.utils.docker import (
    check_resource_availability,
    cleanup_csv_sandbox_containers,
    cleanup_session,
    generate_session_id,
)


SourceMode = Literal["llm_gen", "template", "procedural"]
ALLOWED_SOURCES: tuple[SourceMode, ...] = ("llm_gen", "template", "procedural")


@dataclass
class CSVTask:
    """Represents a single dataset task for episode generation."""

    csv_path: str
    dataset_name: str
    dataset_description: str
    questions: list[dict]
    questions_file: Path
    source: SourceMode


def make_signal_handler(session_id: str):
    """Create a signal handler that cleans up only this session's containers."""

    def handler(signum, frame):
        print(f"\n\nInterrupted. Cleaning up session {session_id} containers...")
        cleanup_session(session_id)
        print("Cleanup complete")
        sys.exit(0)

    return handler


def infer_source_from_questions_dir(questions_dir: Path) -> SourceMode | None:
    """Infer source mode from questions dir name when possible."""
    inferred = questions_dir.name
    if inferred in ALLOWED_SOURCES:
        return inferred  # type: ignore[return-value]
    return None


def build_csv_source_map() -> dict[str, str]:
    """Build dataset_name -> csv_path map from configured sources."""
    csv_sources = config.csv_sources
    if isinstance(csv_sources, str):
        csv_sources = [csv_sources]

    csv_by_name: dict[str, str] = {}
    for csv_path in csv_sources:
        csv_path_obj = Path(csv_path)
        dataset_name = (
            csv_path_obj.parent.name
            if csv_path_obj.name == "data.csv"
            else csv_path_obj.stem
        )
        csv_by_name[dataset_name] = str(csv_path)

    return csv_by_name


def filter_by_difficulty(
    questions: list[dict],
    distribution: dict[str, float],
    total_target: int,
) -> tuple[list[dict], bool]:
    """Select questions matching target difficulty distribution."""
    result = []
    allocations = []
    base_total = 0

    for idx, (difficulty, fraction) in enumerate(distribution.items()):
        exact = total_target * fraction
        base = int(exact)
        fractional_part = exact - base
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
            return [], False
        result.extend(matching[:count_needed])

    return result, True


def gather_csv_tasks(
    source: SourceMode,
    questions_dir: Path,
    ui: EpisodeGenUI,
    skip_difficulty_filter: bool,
) -> list[CSVTask]:
    """Gather valid dataset tasks from questions directory."""
    csv_by_name = build_csv_source_map()
    question_files = sorted(questions_dir.glob("*/questions.json"))

    tasks: list[CSVTask] = []
    for questions_file in question_files:
        dataset_name = questions_file.parent.name
        csv_path = csv_by_name.get(dataset_name)
        if not csv_path:
            ui.base.print_warning(f"Skipping {dataset_name}: no matching CSV found")
            continue

        questions = load_questions(str(questions_file))
        questions = [q for q in questions if q.get("source") == source]
        if not questions:
            continue

        _meta_dataset_name, dataset_description = load_dataset_meta(csv_path)
        if not dataset_description or not dataset_description.strip():
            data_overview = generate_data_overview(str(csv_path))
            dataset_description = generate_description_from_overview(data_overview)
            ui.base.print_warning(
                f"{dataset_name}: no description found, synthesized from data overview"
            )

        if source == "llm_gen" and not skip_difficulty_filter:
            filtered_questions, filter_ok = filter_by_difficulty(
                questions,
                config.question_difficulty_distribution,
                config.num_questions_to_generate,
            )
            if filter_ok:
                questions = filtered_questions
            else:
                ui.base.print_warning(
                    f"{dataset_name}: insufficient questions for target distribution "
                    f"(need {config.num_questions_to_generate} with "
                    f"{config.question_difficulty_distribution}). Using all "
                    f"{len(questions)} available questions instead."
                )

        tasks.append(
            CSVTask(
                csv_path=str(csv_path),
                dataset_name=dataset_name,
                dataset_description=dataset_description,
                questions=questions,
                questions_file=questions_file,
                source=source,
            )
        )

    return tasks


def compute_question_fingerprint(
    question: dict,
    source: SourceMode,
    dataset_hash: str,
) -> str | None:
    """Compute manifest fingerprint for a question by source."""
    if source == "llm_gen":
        question_text = resolve_question_prompt(question)
        if not question_text:
            return None
        return compute_llm_fingerprint(question_text, dataset_hash)
    return compute_synthetic_fingerprint_from_question(question, dataset_hash)


def filter_cached_questions(
    tasks: list[CSVTask],
    source: SourceMode,
    manifest: DatagenManifest,
    retry_failed: bool,
    ui: EpisodeGenUI,
) -> tuple[list[CSVTask], dict[str, str]]:
    """Filter tasks by manifest cache and return dataset hash cache."""
    dataset_hashes: dict[str, str] = {}
    original_total = sum(len(task.questions) for task in tasks)

    for task in tasks:
        if task.csv_path not in dataset_hashes:
            dataset_hashes[task.csv_path] = compute_dataset_hash(task.csv_path)
        dataset_hash = dataset_hashes[task.csv_path]

        filtered_questions = []
        for question in task.questions:
            fingerprint = compute_question_fingerprint(
                question=question,
                source=source,
                dataset_hash=dataset_hash,
            )
            if fingerprint is None:
                filtered_questions.append(question)
                continue

            if source == "llm_gen":
                exists = manifest.has_llm(
                    fingerprint,
                    include_failures=not retry_failed,
                )
            else:
                exists = manifest.has_synthetic(
                    fingerprint,
                    include_failures=not retry_failed,
                )

            if not exists:
                filtered_questions.append(question)

        task.questions = filtered_questions

    tasks = [task for task in tasks if task.questions]
    new_total = sum(len(task.questions) for task in tasks)
    if original_total > new_total:
        ui.base.print_status(f"Skipping {original_total - new_total} cached questions")

    return tasks, dataset_hashes


def print_manifest_stats(
    source: SourceMode, manifest: DatagenManifest, ui: EpisodeGenUI
) -> None:
    """Print source-scoped manifest summary."""
    stats = manifest.stats()
    if source == "llm_gen":
        if stats["llm_total"] > 0:
            ui.base.print_status(
                f"Manifest loaded: {stats['llm_success']} success, {stats['llm_failure']} failures"
            )
        return

    if stats["synthetic_total"] > 0:
        ui.base.print_status(
            f"Manifest loaded: {stats['synthetic_success']} success, {stats['synthetic_failure']} failures"
        )


async def process_llm_task(
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
) -> tuple[list[EpisodeJSONL], list[dict]]:
    """Process one dataset in LLM triangulation mode."""
    data_overview = generate_data_overview(task.csv_path)

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

    episodes: list[EpisodeJSONL] = []
    failures: list[dict] = []
    for result in results:
        question_text = resolve_question_prompt(result.question)

        fingerprint = None
        if manifest is not None and dataset_hash is not None:
            fingerprint = compute_question_fingerprint(
                question=result.question,
                source="llm_gen",
                dataset_hash=dataset_hash,
            )

        if not result.verified:
            failures.append(
                {
                    "question": question_text[:100],
                    "gold_answer": result.gold_trace.get("final_answer")
                    if result.gold_trace
                    else None,
                    "gold_success": result.gold_trace.get("success")
                    if result.gold_trace
                    else False,
                    "majority_answer": result.majority_answer_hash,
                    "majority_count": result.majority_count,
                    "n_consistency": len(result.consistency_results),
                }
            )

            if manifest is not None and fingerprint is not None:
                manifest.record_llm(
                    fingerprint=fingerprint,
                    status="failure",
                    dataset=task.dataset_name,
                    question_text=question_text,
                    model=teacher_model,
                    n_consistency=len(result.consistency_results),
                    elapsed_seconds=result.timing_metadata.get("avg_elapsed"),
                )

            if verified_only:
                continue

        consistency_traces = [trace for trace, _ in result.consistency_results]
        verification_result = VerificationResult(
            success=result.verified,
            match=result.verified,
            trace=result.gold_trace,
            traces=consistency_traces,
            majority_answer_hash=result.majority_answer_hash,
            error=None,
        )

        episode = await create_episode(
            question=result.question,
            verification_result=verification_result,
            source="llm_gen",
            csv_path=task.csv_path,
        )
        episode.timing = TimingMetadataDict(
            gold_elapsed=result.timing_metadata["gold_elapsed"],
            consistency_elapsed=result.timing_metadata["consistency_elapsed"],
            total_elapsed=result.timing_metadata["total_elapsed"],
            avg_elapsed=result.timing_metadata["avg_elapsed"],
        )
        episodes.append(episode)

        if result.verified and manifest is not None and fingerprint is not None:
            manifest.record_llm(
                fingerprint=fingerprint,
                status="success",
                dataset=task.dataset_name,
                question_text=question_text,
                episode_id=episode.episode_id,
                model=teacher_model,
                n_consistency=len(result.consistency_results),
                elapsed_seconds=result.timing_metadata.get("avg_elapsed"),
            )

    return episodes, failures


async def process_ground_truth_task(
    task: CSVTask,
    teacher_model: str,
    max_turns: int,
    sampling_args: dict,
    float_tol: float,
    ui: EpisodeGenUI,
    session_id: str,
    manifest: DatagenManifest | None = None,
    dataset_hash: str | None = None,
) -> tuple[list[EpisodeJSONL], list[dict]]:
    """Process one dataset in ground-truth verification mode."""
    data_overview = generate_data_overview(task.csv_path)
    episodes: list[EpisodeJSONL] = []
    failures: list[dict] = []

    for index, question in enumerate(task.questions, 1):
        question_preview = resolve_question_prompt(question)[:60]
        ui.base.print_status(f"  [{index}/{len(task.questions)}] {question_preview}...")

        start = time.time()
        verification_result = await verify_question(
            question=question,
            csv_path=task.csv_path,
            strategy="ground_truth",
            model=teacher_model,
            max_turns=max_turns,
            sampling_args=sampling_args,
            dataset_description=task.dataset_description,
            data_overview=data_overview,
            ui=ui,
            session_id=session_id,
            float_tol=float_tol,
        )
        elapsed = time.time() - start

        fingerprint = None
        if manifest is not None and dataset_hash is not None:
            fingerprint = compute_question_fingerprint(
                question=question,
                source=task.source,
                dataset_hash=dataset_hash,
            )

        success = (
            verification_result.success
            and verification_result.match is True
            and verification_result.trace is not None
        )
        if success:
            episode = await create_episode(
                question=question,
                verification_result=verification_result,
                source=task.source,
                csv_path=task.csv_path,
            )
            episode.timing = TimingMetadataDict(
                gold_elapsed=elapsed,
                consistency_elapsed=[],
                total_elapsed=elapsed,
                avg_elapsed=elapsed,
            )
            episodes.append(episode)
            ui.base.print_success(f"    Validated ({elapsed:.1f}s)")

            if manifest is not None and fingerprint is not None:
                manifest.record_synthetic(
                    fingerprint=fingerprint,
                    status="success",
                    dataset=task.dataset_name,
                    template_name=question.get("template_name", "unknown"),
                    template_params=question.get("template_params"),
                    episode_id=episode.episode_id,
                    model=teacher_model,
                    elapsed_seconds=elapsed,
                )
            continue

        trace = verification_result.trace
        failure_record = {
            "question": resolve_question_prompt(question)[:100],
            "template_name": question.get("template_name"),
            "variant_index": question.get("variant_index"),
            "error": verification_result.error,
            "elapsed": elapsed,
        }
        if trace is not None and verification_result.match is False:
            failure_record["expected"] = question.get("ground_truth")
            failure_record["actual"] = trace.get("final_answer")

        failures.append(failure_record)
        ui.base.print_warning(f"    Failed: {verification_result.error}")

        if manifest is not None and fingerprint is not None:
            manifest.record_synthetic(
                fingerprint=fingerprint,
                status="failure",
                dataset=task.dataset_name,
                template_name=question.get("template_name", "unknown"),
                template_params=question.get("template_params"),
                model=teacher_model,
                elapsed_seconds=elapsed,
            )

    return episodes, failures


async def run_llm_pipeline(
    tasks: list[CSVTask],
    n_consistency: int,
    teacher_model: str,
    max_turns: int,
    sampling_args: dict,
    float_tol: float,
    verified_only: bool,
    ui: EpisodeGenUI,
    session_id: str,
    manifest: DatagenManifest,
    dataset_hashes: dict[str, str],
) -> tuple[list[EpisodeJSONL], list[dict], bool]:
    """Run pooled LLM triangulation pipeline across tasks."""
    ui.base.print_status("Cleaning up old containers...")
    cleanup_csv_sandbox_containers()

    max_concurrent = config.max_concurrent_containers
    if config.dynamic_triangulation and config.triangulation_by_difficulty:
        from src.datagen.teacher import resolve_n_consistency

        max_consistency = n_consistency
        for task in tasks:
            for question in task.questions:
                max_consistency = max(
                    max_consistency,
                    resolve_n_consistency(
                        question,
                        n_consistency,
                        config.triangulation_by_difficulty,
                    ),
                )
    else:
        max_consistency = n_consistency

    resource_status = check_resource_availability(max_concurrent)
    if resource_status.recommended_max_containers < 1:
        ui.base.print_error(
            f"Insufficient resources: {resource_status.available_memory_gb:.1f}GB available, "
            f"{resource_status.existing_containers} containers already running."
        )
        ui.base.print_error(
            "Wait for other scripts to finish, or run: "
            'uv run python -c "from src.utils.docker import cleanup_csv_sandbox_containers; cleanup_csv_sandbox_containers()"'
        )
        return [], [], True

    if resource_status.recommended_max_containers < max_concurrent:
        original = max_concurrent
        max_concurrent = resource_status.recommended_max_containers
        ui.base.print_warning(
            f"Reducing parallelism: {original} -> {max_concurrent} containers "
            f"({resource_status.available_memory_gb:.1f}GB available, "
            f"{resource_status.existing_containers} containers from other sessions)"
        )
    elif resource_status.existing_containers > 0:
        ui.base.print_status(
            f"Note: {resource_status.existing_containers} containers from other sessions, "
            f"{resource_status.available_memory_gb:.1f}GB available"
        )

    ui.base.print_section(
        f"Processing {len(tasks)} datasets ({max_concurrent} containers pooled)"
    )

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
        container = await pool.acquire(task.csv_path)
        task_dataset_hash = dataset_hashes.get(task.csv_path)
        try:
            episodes, failures = await process_llm_task(
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
            return task, episodes, failures
        finally:
            await pool.release(container)

    pending_tasks = [asyncio.create_task(process_task_wrapper(task)) for task in tasks]

    completed_count = 0
    all_episodes: list[EpisodeJSONL] = []
    all_failures: list[dict] = []
    had_error = False

    try:
        for coro in asyncio.as_completed(pending_tasks):
            task_result, episodes, failures = await coro
            completed_count += 1

            n_verified = sum(1 for episode in episodes if episode.verified)
            ui.base.print_success(
                f"[{completed_count}/{len(tasks)}] {task_result.dataset_name}: "
                f"{len(episodes)} episodes ({n_verified} verified)"
            )

            all_episodes.extend(episodes)
            for failure in failures:
                failure["dataset"] = task_result.dataset_name
            all_failures.extend(failures)
    except Exception as error:
        had_error = True
        ui.base.print_error(f"Episode generation failed: {error}")
    finally:
        for task in pending_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)
        await pool.stop()

    return all_episodes, all_failures, had_error


async def run_ground_truth_pipeline(
    tasks: list[CSVTask],
    teacher_model: str,
    max_turns: int,
    sampling_args: dict,
    float_tol: float,
    ui: EpisodeGenUI,
    session_id: str,
    manifest: DatagenManifest,
    dataset_hashes: dict[str, str],
    parallel: bool,
    n_workers: int,
) -> tuple[list[EpisodeJSONL], list[dict], dict[str, int], bool]:
    """Run ground-truth validation pipeline across tasks."""
    all_episodes: list[EpisodeJSONL] = []
    all_failures: list[dict] = []
    failure_by_template: dict[str, int] = defaultdict(int)
    had_error = False

    async def process_one_task(
        task: CSVTask,
    ) -> tuple[CSVTask, list[EpisodeJSONL], list[dict]]:
        ui.base.print_section(
            f"Processing {task.dataset_name}: {len(task.questions)} questions"
        )
        episodes, failures = await process_ground_truth_task(
            task=task,
            teacher_model=teacher_model,
            max_turns=max_turns,
            sampling_args=sampling_args,
            float_tol=float_tol,
            ui=ui,
            session_id=session_id,
            manifest=manifest,
            dataset_hash=dataset_hashes.get(task.csv_path),
        )
        return task, episodes, failures

    if parallel and len(tasks) > 1:
        semaphore = asyncio.Semaphore(max(1, n_workers))

        async def process_with_limit(
            task: CSVTask,
        ) -> tuple[CSVTask, list[EpisodeJSONL], list[dict]]:
            async with semaphore:
                return await process_one_task(task)

        results = await asyncio.gather(
            *(process_with_limit(task) for task in tasks),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                had_error = True
                ui.base.print_error(f"Dataset failed with exception: {result}")
                continue
            task_result, episodes, failures = result
            all_episodes.extend(episodes)
            all_failures.extend(failures)
            for failure in failures:
                failure_by_template[failure.get("template_name", "unknown")] += 1
            ui.base.print_success(
                f"{task_result.dataset_name}: {len(episodes)} verified, {len(failures)} failed"
            )
    else:
        for task in tasks:
            try:
                task_result, episodes, failures = await process_one_task(task)
            except Exception as error:
                had_error = True
                ui.base.print_error(
                    f"Dataset {task.dataset_name} failed with exception: {error}"
                )
                continue

            all_episodes.extend(episodes)
            all_failures.extend(failures)
            for failure in failures:
                failure_by_template[failure.get("template_name", "unknown")] += 1
            ui.base.print_success(
                f"{task_result.dataset_name}: {len(episodes)} verified, {len(failures)} failed"
            )

    return all_episodes, all_failures, failure_by_template, had_error


def failure_log_name(source: SourceMode) -> str:
    """Return failures sidecar filename for the source."""
    if source == "llm_gen":
        return "failures_llm.jsonl"
    return "failures_synthetic.jsonl"


async def main(
    questions_dir: str,
    output_path: str,
    source: str | None = None,
    max_questions: int | None = None,
    n_consistency: int | None = None,
    skip_difficulty_filter: bool = False,
    difficulties: list[str] | None = None,
    retry_failed: bool = False,
    parallel: bool = False,
    n_workers: int = 4,
) -> int:
    """Generate episodes for a single source mode."""
    ui = EpisodeGenUI()

    questions_dir_path = Path(questions_dir)
    if source is None:
        source = infer_source_from_questions_dir(questions_dir_path)
    if source not in ALLOWED_SOURCES:
        ui.base.print_error(
            "Invalid or missing source. Use --source with one of: "
            "template, procedural, llm_gen."
        )
        return 2
    source_mode = cast(SourceMode, source)

    if not questions_dir_path.exists():
        ui.base.print_error(f"Questions directory not found: {questions_dir}")
        return 2

    output_jsonl = Path(output_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.unlink()

    session_id = generate_session_id()
    ui.base.print_status(f"Session ID: {session_id}")

    signal_handler = make_signal_handler(session_id)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    teacher_model = config.teacher_model
    n_consistency = n_consistency if n_consistency is not None else config.n_consistency
    max_turns = config.max_turns
    float_tol = config.float_tolerance
    verified_only = config.verified_only
    sampling_args = {
        "temperature": config.sampling_args.temperature,
        "max_tokens": config.sampling_args.max_tokens,
    }

    tasks = gather_csv_tasks(
        source=source_mode,
        questions_dir=questions_dir_path,
        ui=ui,
        skip_difficulty_filter=skip_difficulty_filter,
    )

    if difficulties:
        allowed = {difficulty.upper() for difficulty in difficulties}
        for task in tasks:
            task.questions = [
                question
                for question in task.questions
                if str(question.get("difficulty", "")).upper() in allowed
            ]
        tasks = [task for task in tasks if task.questions]

    manifest = DatagenManifest()
    manifest.load()
    print_manifest_stats(source=source_mode, manifest=manifest, ui=ui)

    tasks, dataset_hashes = filter_cached_questions(
        tasks=tasks,
        source=source_mode,
        manifest=manifest,
        retry_failed=retry_failed,
        ui=ui,
    )

    if max_questions is not None:
        for task in tasks:
            if len(task.questions) > max_questions:
                task.questions = task.questions[:max_questions]

    tasks = [task for task in tasks if task.questions]
    if not tasks:
        ui.base.print_error("No valid dataset tasks found.")
        return 2

    ui.base.print_section(f"Found {len(tasks)} datasets to process")
    for task in tasks:
        ui.base.print_status(f"  {task.dataset_name}: {len(task.questions)} questions")

    if source_mode == "llm_gen":
        all_episodes, all_failures, had_error = await run_llm_pipeline(
            tasks=tasks,
            n_consistency=n_consistency,
            teacher_model=teacher_model,
            max_turns=max_turns,
            sampling_args=sampling_args,
            float_tol=float_tol,
            verified_only=verified_only,
            ui=ui,
            session_id=session_id,
            manifest=manifest,
            dataset_hashes=dataset_hashes,
        )
        failure_by_template: dict[str, int] = {}
    else:
        (
            all_episodes,
            all_failures,
            failure_by_template,
            had_error,
        ) = await run_ground_truth_pipeline(
            tasks=tasks,
            teacher_model=teacher_model,
            max_turns=max_turns,
            sampling_args=sampling_args,
            float_tol=float_tol,
            ui=ui,
            session_id=session_id,
            manifest=manifest,
            dataset_hashes=dataset_hashes,
            parallel=parallel,
            n_workers=n_workers,
        )

    if had_error:
        return 2

    with open(output_jsonl, "w") as output_file:
        for episode in all_episodes:
            output_file.write(json.dumps(episode.model_dump(), default=str) + "\n")

    ui.base.print_section("PIPELINE COMPLETE")
    ui.base.print_key_value("Output file", str(output_jsonl))
    ui.base.print_key_value("Total datasets", len(tasks))
    ui.base.print_key_value("Total episodes", len(all_episodes))
    ui.base.print_key_value("Total failures", len(all_failures))

    if all_failures and source_mode != "llm_gen":
        ui.base.print_empty_line()
        ui.base.print_status("Failures by template:")
        for template, count in sorted(
            failure_by_template.items(), key=lambda item: -item[1]
        ):
            ui.base.print_status(f"  {template}: {count}")

    if all_failures:
        failures_log = output_jsonl.parent / failure_log_name(source_mode)
        with open(failures_log, "w") as failure_file:
            for failure in all_failures:
                failure_file.write(json.dumps(failure, default=str) + "\n")
        ui.base.print_status(f"Failures logged to: {failures_log}")

    ui.base.print_empty_line()

    if not all_episodes:
        return 2
    if all_failures:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build argparse parser for module execution."""
    parser = argparse.ArgumentParser(description="Unified episode generation pipeline")
    parser.add_argument(
        "--questions-dir",
        type=str,
        required=True,
        help="Directory containing question files (e.g. data/questions/template)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--source",
        choices=list(ALLOWED_SOURCES),
        default=None,
        help="Question source mode (template, procedural, llm_gen)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Max questions per dataset",
    )
    parser.add_argument(
        "--n-consistency",
        type=int,
        default=None,
        help="Consistency traces for llm_gen",
    )
    parser.add_argument(
        "--skip-difficulty-filter",
        action="store_true",
        help="Skip LLM difficulty distribution filter",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=None,
        help="Filter to specific difficulty values",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry questions that previously failed",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process datasets in parallel for ground-truth modes",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Parallel workers for ground-truth modes",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        raise SystemExit(
            asyncio.run(
                main(
                    questions_dir=args.questions_dir,
                    output_path=args.output,
                    source=args.source,
                    max_questions=args.max_questions,
                    n_consistency=args.n_consistency,
                    skip_difficulty_filter=args.skip_difficulty_filter,
                    difficulties=args.difficulties,
                    retry_failed=args.retry_failed,
                    parallel=args.parallel,
                    n_workers=args.n_workers,
                )
            )
        )
    except KeyboardInterrupt:
        raise SystemExit(0)
