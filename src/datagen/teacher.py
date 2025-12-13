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
from typing import Any, List, Tuple

from src.core.environment import Environment
from src.core.types import ExecutionTrace, Question
from src.core.config import DataConfig, ModelConfig, ExecutionConfig, TaskConfig
from src.utils.hashing import hash_artifact
from src.core.kernel import JupyterKernel
from src.utils.logger import create_logger


def answers_match(
    hash1: str | None,
    hash2: str | None,
    val1: Any = None,
    val2: Any = None,
    float_tol: float = 0.1
) -> bool:
    """
    Check if two answers match, with tolerance for floats.

    Args:
        hash1, hash2: Answer hashes (for exact match)
        val1, val2: Raw answer values (for tolerant comparison)
        float_tol: Absolute tolerance for float comparison (default ±0.1)

    Returns:
        True if answers match within tolerance
    """
    # Exact hash match
    if hash1 is not None and hash2 is not None and hash1 == hash2:
        return True

    # If we have raw values, try tolerant comparison
    if val1 is not None and val2 is not None:
        # Both floats: compare with tolerance
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(float(val1) - float(val2)) <= float_tol

        # Both tuples/lists: compare element-wise
        if isinstance(val1, (tuple, list)) and isinstance(val2, (tuple, list)):
            if len(val1) != len(val2):
                return False
            for a, b in zip(val1, val2):
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if abs(float(a) - float(b)) > float_tol:
                        return False
                elif a != b:
                    return False
            return True

        # Exact equality for other types
        if val1 == val2:
            return True

    return False


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
) -> tuple[ExecutionTrace, list[dict], str]:
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
        Tuple of (ExecutionTrace, conversation_messages, system_prompt)
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

        # Extract conversation for SFT training
        conversation_messages = final_state.conversation.to_openai_messages()
        system_prompt = conversation_messages[0]["content"] if conversation_messages else ""

        # Extract code cells from all assistant messages
        import re
        code_cells = []
        pattern = r"```python\n(.*?)```"
        for msg in conversation_messages:
            if msg.get("role") == "assistant":
                cells = re.findall(pattern, msg["content"], re.DOTALL)
                code_cells.extend(cells)

        # Snapshot artifacts from kernel
        artifacts = kernel.snapshot_artifacts()

        # Get final answer
        final_answer = kernel.get_final_answer()
        final_answer_hash = hash_artifact(final_answer) if final_answer is not None else None

        # Check if execution was successful (submitted an answer)
        execution_success = final_answer is not None

        trace = ExecutionTrace(
            code_cells=code_cells,
            artifacts=artifacts,
            final_answer=final_answer,
            final_answer_hash=final_answer_hash,
            execution_success=execution_success,
        )

        return trace, conversation_messages, system_prompt

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
) -> tuple[ExecutionTrace, list[dict], str, list[tuple[ExecutionTrace, list[dict]]], bool]:
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
        - gold_conversation: list[dict] of messages
        - system_prompt: str
        - consistency_results: list of (ExecutionTrace, conversation) tuples
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

    gold_trace, gold_conversation, system_prompt = execute_teacher_trace(
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
    consistency_results = []
    for i in range(n_consistency):
        logger.info("executing_consistency_trace", extra={"trace_num": i + 1})

        trace, conversation, _ = execute_teacher_trace(
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
        consistency_results.append((trace, conversation))

    # 3. Majority voting on final answer hashes
    consistency_answers = [
        (trace.final_answer_hash, trace.final_answer)
        for trace, _ in consistency_results
        if trace.final_answer_hash is not None
    ]

    if not consistency_answers:
        # No consistency traces succeeded
        logger.info("triangulation_failed", extra={
            "reason": "No consistency traces produced answers"
        })
        return gold_trace, gold_conversation, system_prompt, consistency_results, False

    # Find majority answer by hash
    consistency_hashes = [h for h, _ in consistency_answers]
    hash_counts = Counter(consistency_hashes)
    majority_hash, majority_count = hash_counts.most_common(1)[0]

    # Get the actual value for the majority hash
    majority_value = next(
        (val for h, val in consistency_answers if h == majority_hash),
        None
    )

    # Check if gold matches majority (with tolerance for floats)
    # Default tolerance is 0.1, can be configured via config
    float_tol = 0.1  # TODO: Make configurable via function parameter
    verified = answers_match(
        gold_trace.final_answer_hash,
        majority_hash,
        gold_trace.final_answer,
        majority_value,
        float_tol=float_tol
    )

    logger.info("triangulation_complete", extra={
        "verified": verified,
        "gold_hash": gold_trace.final_answer_hash,
        "gold_value": str(gold_trace.final_answer)[:50],
        "majority_hash": majority_hash,
        "majority_value": str(majority_value)[:50],
        "majority_count": majority_count,
        "total_consistency": len(consistency_results),
        "float_tolerance": float_tol,
    })

    return gold_trace, gold_conversation, system_prompt, consistency_results, verified


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
) -> list[tuple[dict, ExecutionTrace, list[dict], str, list[tuple[ExecutionTrace, list[dict]]], bool]]:
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
        List of tuples: (question_dict, gold_trace, gold_conversation, system_prompt, consistency_results, verified)
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

        gold_trace, gold_conversation, system_prompt, consistency_results, verified = triangulate_teacher(
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

        results.append((q_dict, gold_trace, gold_conversation, system_prompt, consistency_results, verified))

    # Summary stats
    n_verified = sum(1 for _, _, _, _, _, verified in results if verified)
    logger.info("batch_complete", extra={
        "total": len(questions),
        "verified": n_verified,
        "verification_rate": n_verified / len(questions) if questions else 0.0,
    })

    return results
