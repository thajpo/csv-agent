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
    match: bool | None  # None if execution failed
    trace: dict | None  # single trace for ground-truth verification
    traces: list[dict]  # multiple traces for consistency verification
    majority_answer_hash: str | None
    error: str | None


def verify_question(
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
        return verify_synthetic(question, csv_path, **kwargs)
    else:
        return verify_llm(question, csv_path, n_traces=n_traces, **kwargs)


def verify_synthetic(
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
    # TODO: Import and call teacher.execute_teacher_trace
    # from src.datagen.teacher import execute_teacher_trace
    # trace, _, _, _ = await execute_teacher_trace(
    #     csv_path=csv_path,
    #     question=question.get("question_text") or question.get("question_mechanical"),
    #     hint=question.get("hint"),
    #     **kwargs,
    # )
    # Compare trace answer to ground_truth with hash and tolerance logic
    logger.warning("verify_synthetic: not yet implemented, placeholder")
    return VerificationResult(
        success=False,
        match=None,
        trace=None,
        traces=[],
        majority_answer_hash=None,
        error="Not implemented",
    )


def verify_llm(
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
    # TODO: Import and call teacher.triangulate_teacher
    # from src.datagen.teacher import triangulate_teacher
    # result = await triangulate_teacher(
    #     csv_path=csv_path,
    #     question=question["question_text"],
    #     hint=question.get("hint"),
    #     n_consistency=n_traces,
    #     **kwargs,
    # )
    logger.warning("verify_llm: not yet implemented, placeholder")
    return VerificationResult(
        success=False,
        match=None,
        trace=None,
        traces=[],
        majority_answer_hash=None,
        error="Not implemented",
    )
