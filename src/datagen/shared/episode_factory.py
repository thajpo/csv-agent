"""Episode factory for creating training episodes from verification results."""

import uuid
from datetime import datetime
from typing import Literal

from csv_spec import (
    EpisodeJSONL,
    QADict,
    TraceDict,
    TriangulationMetadataDict,
    TimingMetadataDict,
)
from src.datagen.shared.verification import (
    VerificationResult,
    derive_ground_truth_verification,
    derive_llm_verification,
    resolve_question_prompt,
    validate_float_tolerance,
)
from src.datagen.process_report import build_process_report


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
        "question_text": resolve_question_prompt(question),
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
    float_tolerance = validate_float_tolerance(verification_result.float_tolerance)

    n_succeeded = sum(1 for t in consistency_traces if t.get("success", False))
    if source == "llm_gen":
        evidence = derive_llm_verification(
            gold_trace=gold_trace,
            consistency_traces=consistency_traces,
            float_tolerance=float_tolerance,
        )
    else:
        evidence = derive_ground_truth_verification(
            question=question,
            gold_trace=gold_trace,
            float_tolerance=float_tolerance,
        )
    if (
        verification_result.match is not evidence.verdict
        or verification_result.majority_answer_hash
        != evidence.majority_answer_hash
        or verification_result.success != (evidence.verdict is True)
    ):
        raise ValueError("Verification result disagrees with trace provenance")

    triangulation = TriangulationMetadataDict(
        n_consistency_runs=len(consistency_traces),
        n_consistency_succeeded=n_succeeded,
        majority_answer_hash=evidence.majority_answer_hash,
        majority_count=evidence.majority_count,
        gold_matches_majority=evidence.verdict is True,
        float_tolerance=float_tolerance,
    )
    verified = evidence.verdict is True
    process_report = build_process_report(
        source=source,
        gold_trace=gold_trace,
        consistency_traces=consistency_traces,
        verifier_verdict=evidence.verdict,
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
        verified=verified,
        triangulation=triangulation,
        timing=timing,
        process_report=process_report,
        source=source,
    )

    return episode
