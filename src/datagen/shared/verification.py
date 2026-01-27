"""Verification utilities (correctness gate).

Centralizes correctness checking for both LLM and synthetic questions.
Provides two evidence strategies: ground_truth and consistency.
"""

from dataclasses import dataclass
from typing import Literal, Any
import logging


logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    success: bool
    match: bool | None
    trace: dict | None
    traces: list[dict]
    majority_answer_hash: str | None
    error: str | None


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
    from src.datagen.teacher import execute_teacher_trace, answers_match
    import time

    start_time = time.time()

    try:
        # Use mechanical question or question_text if available
        question_text = (
            question.get("question_mechanical") or question.get("question_text") or ""
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
                error=trace.get("error", "Unknown"),
            )

        # Get expected answer hashes
        expected_hashes = question.get("ground_truth_hashes") or [
            question.get("ground_truth_hash")
        ]
        expected_hashes = [h for h in expected_hashes if h is not None]

        if not expected_hashes:
            return VerificationResult(
                success=False,
                match=None,
                trace=trace,
                traces=[],
                majority_answer_hash=None,
                error="No ground_truth_hash in question",
            )

        # Fast path: exact hash match
        actual_hash = trace.get("final_answer_hash")
        if actual_hash in expected_hashes:
            return VerificationResult(
                success=True,
                match=True,
                trace=trace,
                traces=[],
                majority_answer_hash=actual_hash,
                error=None,
            )

        # Tolerant comparison
        expected_answers = question.get("_ground_truths") or [
            question.get("_ground_truth")
        ]
        expected_answers = [a for a in expected_answers if a is not None]
        actual_answer = trace.get("final_answer")

        from src.core.config import config

        float_tol = kwargs.get("float_tol", config.float_tolerance)

        for exp_hash, exp_answer in zip(expected_hashes, expected_answers):
            if answers_match(
                exp_hash, actual_hash, exp_answer, actual_answer, float_tol=float_tol
            ):
                return VerificationResult(
                    success=True,
                    match=True,
                    trace=trace,
                    traces=[],
                    majority_answer_hash=actual_hash,
                    error=None,
                )

        return VerificationResult(
            success=False,
            match=False,
            trace=trace,
            traces=[],
            majority_answer_hash=actual_hash,
            error=f"Answer mismatch: expected {expected_answers}, got {actual_answer}",
        )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"verify_synthetic error: {e}")
        return VerificationResult(
            success=False,
            match=None,
            trace=None,
            traces=[],
            majority_answer_hash=None,
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
    from src.datagen.teacher import triangulate_teacher

    try:
        result = await triangulate_teacher(
            csv_path=csv_path,
            question=question.get("question_text", ""),
            hint=question.get("hint") or None,
            n_consistency=n_traces,
            **kwargs,
        )

        return VerificationResult(
            success=result.get("verified", False),
            match=result.get("gold_matches_majority"),
            trace=result.get("gold_trace"),
            traces=result.get("consistency_traces", []),
            majority_answer_hash=result.get("majority_answer_hash"),
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
            error=str(e),
        )
