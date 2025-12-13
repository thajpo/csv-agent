"""
Teacher execution and triangulation verification.

This module implements the teacher triangulation protocol:
1. Run teacher WITH hint (gold trace)
2. Run teacher WITHOUT hint N times (consistency traces)
3. Compare final answers using majority voting
4. Only keep episodes where gold matches majority

This filters out questions where:
- The hint is misleading
- The question is ambiguous
- The dataset doesn't support the question
"""

import logging
from collections import Counter
from typing import List, Tuple

from src.core.environment import Environment
from src.core.types import ExecutionTrace, Question
from src.core.config import DataConfig, ModelConfig, ExecutionConfig, TaskConfig
from src.utils.hashing import hash_artifact
from src.core.kernel import JupyterKernel
from src.utils.logger import create_logger


def execute_teacher_trace(
    csv_path: str,
    question: str,
    hint: str | None = None,
    mode: str = "teacher-tutor",
    model: str = "openai/gpt-oss-120b",
    dataset_description: str = "",
    data_overview: str = "",
    max_turns: int = 10,
    sampling_args: dict | None = None,
    logger: logging.Logger | None = None,
) -> ExecutionTrace:
    """
    Execute a single teacher trace (with or without hint).

    Logger will be created silently if not provided.

    Args:
        csv_path: Path to CSV file
        question: Question to solve
        hint: Optional hint (None for consistency mode)
        mode: "teacher-tutor" (with hint) or "teacher-consistency" (without)
        model: Model identifier
        dataset_description: Dataset description for prompt
        data_overview: Data overview for prompt
        max_turns: Max conversation turns
        sampling_args: Sampling parameters for model
        logger: Optional logger

    Returns:
        ExecutionTrace with code, artifacts, and final answer
    """
    # Ensure logger exists
    if logger is None:
        logger = create_logger(silent=True)

    # Build question object
    question_obj = Question(question_text=question, hint=hint) if question else None

    # Build Pydantic nested configs
    data_config = DataConfig(
        csv_path=csv_path,
        dataset_description=dataset_description,
        data_overview=data_overview,
    )

    model_config = ModelConfig(
        model_name=model,
        **(sampling_args or {})  # Unpack temperature, max_tokens, top_p
    )

    execution_config = ExecutionConfig(
        max_turns=max_turns,
    )

    task_config = TaskConfig(
        mode=mode,
        question=question_obj,
    )

    # Create fresh kernel for this trace
    kernel = JupyterKernel(timeout=120.0, csv_path=csv_path)

    try:
        # Create environment with nested configs
        env = Environment(
            data=data_config,
            model=model_config,
            execution=execution_config,
            task=task_config,
            kernel=kernel,
            logger=logger,
        )

        # Execute rollout
        final_state = env.rollout()

        # Extract code cells from all turns
        code_cells = []
        for turn in final_state.conversation_manager.active_turns:
            code_cells.extend(turn.code_cells)

        # Snapshot artifacts from kernel
        artifacts = kernel.snapshot_artifacts()

        # Get final answer
        final_answer = kernel.get_final_answer()
        final_answer_hash = hash_artifact(final_answer) if final_answer is not None else None

        # Check if execution was successful (submitted an answer)
        execution_success = final_answer is not None

        return ExecutionTrace(
            code_cells=code_cells,
            artifacts=artifacts,
            final_answer=final_answer,
            final_answer_hash=final_answer_hash,
            execution_success=execution_success,
        )

    finally:
        # Always cleanup kernel
        kernel.shutdown()


def triangulate_teacher(
    csv_path: str,
    question: str,
    hint: str,
    n_consistency: int = 3,
    model: str = "openai/gpt-oss-120b",
    dataset_description: str = "",
    data_overview: str = "",
    max_turns: int = 10,
    sampling_args: dict | None = None,
    logger: logging.Logger | None = None,
) -> Tuple[ExecutionTrace, List[ExecutionTrace], bool]:
    """
    Run teacher triangulation: gold trace + consistency traces.

    Args:
        csv_path: Path to CSV file
        question: Question to solve
        hint: Hint for gold trace
        n_consistency: Number of consistency traces (default 3)
        model: Model identifier
        dataset_description: Dataset description
        data_overview: Data overview
        max_turns: Max turns per trace
        sampling_args: Model sampling args
        logger: Optional logger

    Returns:
        Tuple of:
        - gold_trace: ExecutionTrace WITH hint
        - consistency_traces: List of ExecutionTrace WITHOUT hint
        - verified: True if gold matches majority of consistency traces
    """
    # Ensure logger exists
    if logger is None:
        logger = create_logger(silent=True)

    logger.info("triangulation_start", extra={
        "question": question,
        "n_consistency": n_consistency
    })

    # 1. Run gold trace (with hint)
    logger.info("executing_gold_trace", extra={"hint": hint})

    gold_trace = execute_teacher_trace(
        csv_path=csv_path,
        question=question,
        hint=hint,
        mode="teacher-tutor",
        model=model,
        dataset_description=dataset_description,
        data_overview=data_overview,
        max_turns=max_turns,
        sampling_args=sampling_args,
        logger=logger,
    )

    # 2. Run consistency traces (without hint)
    consistency_traces = []
    for i in range(n_consistency):
        logger.info("executing_consistency_trace", extra={"trace_num": i + 1})

        trace = execute_teacher_trace(
            csv_path=csv_path,
            question=question,
            hint=None,
            mode="teacher-consistency",
            model=model,
            dataset_description=dataset_description,
            data_overview=data_overview,
            max_turns=max_turns,
            sampling_args=sampling_args,
            logger=logger,
        )
        consistency_traces.append(trace)

    # 3. Majority voting on final answer hashes
    consistency_hashes = [
        trace.final_answer_hash
        for trace in consistency_traces
        if trace.final_answer_hash is not None
    ]

    if not consistency_hashes:
        # No consistency traces succeeded
        logger.info("triangulation_failed", extra={
            "reason": "No consistency traces produced answers"
        })
        return gold_trace, consistency_traces, False

    # Find majority answer
    hash_counts = Counter(consistency_hashes)
    majority_hash, majority_count = hash_counts.most_common(1)[0]

    # Check if gold matches majority
    verified = (gold_trace.final_answer_hash == majority_hash)

    logger.info("triangulation_complete", extra={
        "verified": verified,
        "gold_hash": gold_trace.final_answer_hash,
        "majority_hash": majority_hash,
        "majority_count": majority_count,
        "total_consistency": len(consistency_traces),
    })

    return gold_trace, consistency_traces, verified


def batch_triangulate(
    csv_path: str,
    questions: List[dict],
    n_consistency: int = 3,
    model: str = "openai/gpt-oss-120b",
    dataset_description: str = "",
    data_overview: str = "",
    max_turns: int = 10,
    sampling_args: dict | None = None,
    logger: logging.Logger | None = None,
) -> List[Tuple[dict, ExecutionTrace, List[ExecutionTrace], bool]]:
    """
    Run triangulation on a batch of questions.

    Args:
        csv_path: Path to CSV file
        questions: List of question dicts (from question_gen.py)
        n_consistency: Number of consistency traces per question
        model: Model identifier
        dataset_description: Dataset description
        data_overview: Data overview
        max_turns: Max turns per trace
        sampling_args: Model sampling args
        logger: Optional logger

    Returns:
        List of tuples: (question_dict, gold_trace, consistency_traces, verified)
    """
    # Ensure logger exists
    if logger is None:
        logger = create_logger(silent=True)

    results = []

    for i, q_dict in enumerate(questions, 1):
        logger.info("batch_progress", extra={
            "current": i,
            "total": len(questions),
            "question": q_dict["question"]
        })

        gold_trace, consistency_traces, verified = triangulate_teacher(
            csv_path=csv_path,
            question=q_dict["question"],
            hint=q_dict.get("hint", ""),
            n_consistency=n_consistency,
            model=model,
            dataset_description=dataset_description,
            data_overview=data_overview,
            max_turns=max_turns,
            sampling_args=sampling_args,
            logger=logger,
        )

        results.append((q_dict, gold_trace, consistency_traces, verified))

    # Summary stats
    n_verified = sum(1 for _, _, _, verified in results if verified)
    logger.info("batch_complete", extra={
        "total": len(questions),
        "verified": n_verified,
        "verification_rate": n_verified / len(questions) if questions else 0.0,
    })

    return results
