"""Verification utilities (correctness gate).

Centralizes correctness checking for both LLM and synthetic questions.
Provides two evidence strategies: ground_truth and consistency.
"""

import logging
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, Mapping

from csv_spec import TraceDict, hash_artifact


logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    success: bool
    match: bool | None
    trace: TraceDict | None
    traces: list[TraceDict]
    majority_answer_hash: str | None
    float_tolerance: float
    error: str | None


@dataclass(frozen=True)
class VerificationEvidence:
    verdict: bool | None
    majority_answer_hash: str | None
    majority_count: int


def validate_float_tolerance(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError("float tolerance is invalid")
    return float(value)


def trace_answer_hash(trace: TraceDict) -> str | None:
    answer = trace.get("final_answer")
    stored_hash = trace.get("final_answer_hash")
    if answer is None:
        if stored_hash is not None:
            raise ValueError("final_answer_hash is present without a final_answer")
        return None

    computed_hash = hash_artifact(answer)
    if stored_hash is not None and stored_hash != computed_hash:
        raise ValueError("final_answer_hash does not match final_answer")
    return stored_hash or computed_hash


def derive_llm_verification(
    *,
    gold_trace: TraceDict,
    consistency_traces: list[TraceDict],
    float_tolerance: float,
) -> VerificationEvidence:
    from src.datagen.teacher import answers_match, get_majority_answer

    float_tolerance = validate_float_tolerance(float_tolerance)
    successful_answers = [
        trace["final_answer"]
        for trace in consistency_traces
        if trace.get("success") and trace.get("final_answer") is not None
    ]
    if not successful_answers:
        return VerificationEvidence(None, None, 0)

    majority_answer, majority_count = get_majority_answer(
        successful_answers, float_tol=float_tolerance
    )
    majority_answer_hash = hash_artifact(majority_answer)
    verdict = answers_match(
        trace_answer_hash(gold_trace),
        majority_answer_hash,
        gold_trace.get("final_answer"),
        majority_answer,
        float_tol=float_tolerance,
    )
    return VerificationEvidence(verdict, majority_answer_hash, majority_count)


def derive_ground_truth_verification(
    *,
    question: Mapping[str, Any],
    gold_trace: TraceDict,
    float_tolerance: float,
) -> VerificationEvidence:
    from src.datagen.teacher import answers_match

    float_tolerance = validate_float_tolerance(float_tolerance)
    raw_expected_hashes = question.get("ground_truth_hashes")
    expected_hashes = (
        raw_expected_hashes
        if raw_expected_hashes
        else [question.get("ground_truth_hash")]
    )
    if not isinstance(expected_hashes, list) or not expected_hashes or any(
        not isinstance(value, str) or not value for value in expected_hashes
    ):
        raise ValueError("ground-truth hash provenance is unavailable")

    if not gold_trace.get("success") or gold_trace.get("final_answer") is None:
        return VerificationEvidence(None, None, 0)

    actual_hash = trace_answer_hash(gold_trace)
    actual_answer = gold_trace["final_answer"]
    expected_answer = question.get("ground_truth")
    verdict = actual_hash in expected_hashes
    if not verdict and expected_answer is not None:
        verdict = any(
            answers_match(
                expected_hash,
                actual_hash,
                expected_answer,
                actual_answer,
                float_tol=float_tolerance,
            )
            for expected_hash in expected_hashes
        )
    return VerificationEvidence(verdict, actual_hash, 1)


def resolve_question_prompt(question: Mapping[str, Any]) -> str:
    """Resolve canonical runtime prompt text from a question record.

    Resolution order is strict:
    1) question_text
    2) question_mechanical
    """
    question_text = question.get("question_text")
    if isinstance(question_text, str) and question_text.strip():
        return question_text

    mechanical = question.get("question_mechanical")
    if isinstance(mechanical, str) and mechanical.strip():
        return mechanical

    return ""


async def verify_question(
    question: dict,
    csv_path: str,
    strategy: Literal["ground_truth", "consistency"],
    n_traces: int = 1,
    **kwargs,
) -> VerificationResult:
    """Run verification for a question.

    - ground_truth: run one (or n) teacher trace(s) and compare to ground truth
    - consistency: run n teacher traces and require majority agreement

    Hints are optional. In consistency mode, use hint for the gold trace when present.

    Args:
        question: Question record dict.
        csv_path: Path to CSV file.
        strategy: Either "ground_truth" or "consistency".
        n_traces: Number of teacher traces to run.
        **kwargs: Additional arguments (model, ui, float_tol, etc.).

    Returns:
        VerificationResult with match status and traces.
    """
    if strategy == "ground_truth":
        return await verify_synthetic(question, csv_path, **kwargs)
    else:
        return await verify_llm(question, csv_path, n_traces=n_traces, **kwargs)


async def verify_synthetic(
    question: dict,
    csv_path: str,
    **kwargs,
) -> VerificationResult:
    """Convenience wrapper for ground-truth verification (synthetic).

    Runs one teacher trace (optionally with hint) and compares to ground truth.

    Args:
        question: Question record dict.
        csv_path: Path to CSV file.
        **kwargs: Additional arguments (model, ui, float_tol, etc.).

    Returns:
        VerificationResult with match status and full trace.
    """
    from src.core.config import config
    from src.datagen.teacher import execute_teacher_trace

    float_tolerance = kwargs.get("float_tol", config.float_tolerance)

    try:
        question_text = resolve_question_prompt(question)
        if not question_text:
            return VerificationResult(
                success=False,
                match=None,
                trace=None,
                traces=[],
                majority_answer_hash=None,
                float_tolerance=float_tolerance,
                error="Missing question_text/question_mechanical",
            )

        hint = question.get("hint")

        trace, _conversation, _system, elapsed = await execute_teacher_trace(
            csv_path=csv_path,
            question=question_text,
            hint=hint,
            **kwargs,
        )

        # Check if trace succeeded
        if not trace.get("success", False):
            return VerificationResult(
                success=False,
                match=None,
                trace=trace,
                traces=[],
                majority_answer_hash=None,
                float_tolerance=float_tolerance,
                error=trace.get("error", "Unknown"),
            )

        try:
            evidence = derive_ground_truth_verification(
                question=question,
                gold_trace=trace,
                float_tolerance=float_tolerance,
            )
        except ValueError as exc:
            return VerificationResult(
                success=False,
                match=None,
                trace=trace,
                traces=[],
                majority_answer_hash=None,
                float_tolerance=float_tolerance,
                error=str(exc),
            )

        if evidence.verdict is True:
            return VerificationResult(
                success=True,
                match=True,
                trace=trace,
                traces=[],
                majority_answer_hash=evidence.majority_answer_hash,
                float_tolerance=float_tolerance,
                error=None,
            )

        return VerificationResult(
            success=False,
            match=False,
            trace=trace,
            traces=[],
            majority_answer_hash=evidence.majority_answer_hash,
            float_tolerance=float_tolerance,
            error=f"Answer mismatch: got {trace.get('final_answer')}",
        )
    except Exception as e:
        logger.error(f"verify_synthetic error: {e}")
        return VerificationResult(
            success=False,
            match=None,
            trace=None,
            traces=[],
            majority_answer_hash=None,
            float_tolerance=float_tolerance,
            error=str(e),
        )


async def verify_llm(
    question: dict,
    csv_path: str,
    n_traces: int = 3,
    **kwargs,
) -> VerificationResult:
    """Convenience wrapper for consistency verification (LLM).

    Runs gold trace with hint (if present) + N no-hint traces.
    Checks for majority agreement across traces.

    Args:
        question: Question record dict.
        csv_path: Path to CSV file.
        n_traces: Number of consistency traces to run.
        **kwargs: Additional arguments (model, ui, float_tol, etc.).

    Returns:
        VerificationResult with match status and all traces.
    """
    from src.core.config import config
    from src.datagen.teacher import triangulate_teacher

    float_tolerance = kwargs.get("float_tol", config.float_tolerance)

    try:
        question_text = resolve_question_prompt(question)
        if not question_text:
            return VerificationResult(
                success=False,
                match=None,
                trace=None,
                traces=[],
                majority_answer_hash=None,
                float_tolerance=float_tolerance,
                error="Missing question_text/question_mechanical",
            )

        result = await triangulate_teacher(
            csv_path=csv_path,
            question=question_text,
            hint=question.get("hint") or "",
            n_consistency=n_traces,
            **kwargs,
        )

        consistency_traces = [trace for trace, _conv in result.consistency_results]
        evidence = derive_llm_verification(
            gold_trace=result.gold_trace,
            consistency_traces=consistency_traces,
            float_tolerance=float_tolerance,
        )

        return VerificationResult(
            success=evidence.verdict is True,
            match=evidence.verdict,
            trace=result.gold_trace,
            traces=consistency_traces,
            majority_answer_hash=evidence.majority_answer_hash,
            float_tolerance=float_tolerance,
            error=None,
        )
    except Exception as e:
        logger.error(f"verify_llm error: {e}")
        return VerificationResult(
            success=False,
            match=None,
            trace=None,
            traces=[],
            majority_answer_hash=None,
            float_tolerance=float_tolerance,
            error=str(e),
        )
