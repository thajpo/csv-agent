"""Episode factory for creating training episodes from verification results.

This module centralizes episode creation for all question sources:
- template: Template-based questions with ground truth
- llm_gen: LLM-generated questions with consistency verification
- procedural: Procedurally generated questions

Usage:
    # Direct creation from verification result
    episode = await create_episode(
        question=question_dict,
        verification_result=result,
        source="template",
        csv_path="/path/to/data.csv",
    )

    # Convenience wrappers
    episode = await create_episode_from_ground_truth(...)
    episode = await create_episode_from_consistency(...)
"""

import uuid
from datetime import datetime
from typing import Any, Literal

from csv_spec import (
    EpisodeJSONL,
    QADict,
    TraceDict,
    TriangulationMetadataDict,
    TimingMetadataDict,
)
from src.datagen.shared.verification import VerificationResult, verify_question


ALLOWED_SOURCES = ("llm_gen", "template", "procedural")


async def create_episode(
    question: dict,
    verification_result: VerificationResult,
    source: Literal["llm_gen", "template", "procedural"],
    csv_path: str,
) -> EpisodeJSONL:
    """Create episode from verification result.

    Args:
        question: Question metadata (must include id, question_text, hint, etc.)
        verification_result: Output from verify_question()
        source: Origin of question ("llm_gen", "template", or "procedural")
        csv_path: Path to source CSV

    Returns:
        EpisodeJSONL with verification metadata embedded (success flag, error info, etc.)
    """
    # Generate unique episode ID
    episode_id = str(uuid.uuid4())
    timestamp = datetime.now()

    if source not in ALLOWED_SOURCES:
        raise ValueError(f"Invalid episode source: {source}")

    question_source = question.get("source")
    if question_source not in ALLOWED_SOURCES:
        raise ValueError(f"Invalid question source: {question_source}")
    if source != question_source:
        raise ValueError(
            f"Episode source mismatch: source={source} question.source={question_source}"
        )

    # Build QADict from question
    qa_dict: QADict = {
        "id": question.get("id"),
        "question_text": question.get("question_text") or question.get("question", ""),
        "hint": question.get("hint"),
        "difficulty": question.get("difficulty"),
        "n_steps": question.get("n_steps"),
        "category": question.get("category"),
        "tags": question.get("tags"),
        "template_name": question.get("template_name"),
        "template_params": question.get("template_params"),
        "output_type": question.get("output_type"),
        "output_schema": question.get("output_schema"),
        "ground_truth_hash": question.get("ground_truth_hash"),
        "ground_truth_hashes": question.get("ground_truth_hashes"),
        "ground_truth": question.get("ground_truth"),
    }

    # Extract traces from verification result
    gold_trace: TraceDict = verification_result.trace or {
        "turns": [],
        "final_answer": None,
        "final_answer_hash": None,
        "success": False,
    }

    consistency_traces: list[TraceDict] = verification_result.traces or []

    # Calculate triangulation metadata
    n_succeeded = sum(1 for t in consistency_traces if t.get("success", False))

    triangulation = TriangulationMetadataDict(
        n_consistency_runs=len(consistency_traces),
        n_consistency_succeeded=n_succeeded,
        majority_answer_hash=verification_result.majority_answer_hash,
        majority_count=_calculate_majority_count(
            consistency_traces, verification_result.majority_answer_hash
        ),
        gold_matches_majority=verification_result.success
        and verification_result.match is True,
    )

    # Build timing metadata (defaults if not available)
    timing = TimingMetadataDict(
        gold_elapsed=0.0,
        consistency_elapsed=[0.0] * len(consistency_traces),
        total_elapsed=0.0,
        avg_elapsed=0.0,
    )

    # Create the episode
    episode = EpisodeJSONL(
        episode_id=episode_id,
        timestamp=timestamp,
        csv_source=csv_path,
        question=qa_dict,
        gold_trace=gold_trace,
        consistency_traces=consistency_traces,
        verified=verification_result.success and verification_result.match is True,
        triangulation=triangulation,
        timing=timing,
        source=source,
    )

    return episode


def _calculate_majority_count(
    traces: list[TraceDict], majority_hash: str | None
) -> int:
    """Count how many traces match the majority answer hash."""
    if majority_hash is None:
        return 0
    return sum(1 for t in traces if t.get("final_answer_hash") == majority_hash)


async def create_episode_from_ground_truth(
    question: dict, csv_path: str, model: str, **kwargs
) -> EpisodeJSONL:
    """Convenience wrapper for ground-truth verification (template/procedural).

    Runs verification using the "ground_truth" strategy and creates an episode.

    Args:
        question: Question record dict
        csv_path: Path to CSV file
        model: Model identifier for teacher
        **kwargs: Additional arguments passed to verify_question and create_episode
            - source: Override source (default: question["source"])
            - float_tol: Float tolerance for answer matching
            - ui: UI instance for progress display

    Returns:
        EpisodeJSONL created from ground-truth verification
    """
    source = kwargs.pop("source", question.get("source"))

    # Run ground-truth verification
    verification_result = await verify_question(
        question=question,
        csv_path=csv_path,
        strategy="ground_truth",
        model=model,
        **kwargs,
    )

    # Create episode from result
    return await create_episode(
        question=question,
        verification_result=verification_result,
        source=source,
        csv_path=csv_path,
    )


async def create_episode_from_consistency(
    question: dict, csv_path: str, model: str, n_consistency: int = 5, **kwargs
) -> EpisodeJSONL:
    """Convenience wrapper for consistency verification (LLM).

    Runs verification using the "consistency" strategy and creates an episode.

    Args:
        question: Question record dict
        csv_path: Path to CSV file
        model: Model identifier for teacher
        n_consistency: Number of consistency traces to run
        **kwargs: Additional arguments passed to verify_question and create_episode
            - float_tol: Float tolerance for answer matching
            - ui: UI instance for progress display

    Returns:
        EpisodeJSONL created from consistency verification
    """
    # Run consistency verification
    verification_result = await verify_question(
        question=question,
        csv_path=csv_path,
        strategy="consistency",
        n_traces=n_consistency,
        model=model,
        **kwargs,
    )

    # Create episode from result
    return await create_episode(
        question=question,
        verification_result=verification_result,
        source=question.get("source", "llm_gen"),
        csv_path=csv_path,
    )
