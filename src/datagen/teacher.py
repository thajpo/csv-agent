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

import asyncio
import json
import re
import time
import pandas as pd
from typing import Any, List

from src.core.environment import Environment
from src.core.types import (
    TraceDict,
    TurnDict,
    ExecutionResultDict,
    HookDict,
)
from src.utils.hashing import hash_artifact
from src.utils.normalization import normalize_value


def parse_hooks_from_stdout(stdout: str) -> list[HookDict]:
    """Extract hook JSON objects from execution stdout."""
    hooks = []
    for line in stdout.split("\n"):
        if "📍 Hook:" in line:
            json_start = line.find("{")
            if json_start != -1:
                try:
                    hook_data = json.loads(line[json_start:])
                    if hook_data.get("__csv_agent_hook__"):
                        hooks.append(
                            HookDict(
                                variable_name=hook_data.get("variable_name"),
                                code_line=hook_data.get("code_line", ""),
                                value=hook_data.get("value"),
                                value_hash=hook_data.get("value_hash", ""),
                                depends_on=hook_data.get("depends_on", []),
                                description=hook_data.get("description"),
                            )
                        )
                except json.JSONDecodeError:
                    continue
    return hooks


def extract_reasoning_from_response(response: str) -> str:
    """Extract reasoning text from assistant response (everything before code block)."""
    code_pattern = r"```python\n.*?```"
    parts = re.split(code_pattern, response, flags=re.DOTALL)
    return parts[0].strip() if parts else ""


def build_trace_dict(
    final_state,
    conversation_messages: list[dict],
) -> TraceDict:
    """Build TraceDict from environment final state and conversation."""
    turns: list[TurnDict] = []

    assistant_messages = [
        m for m in conversation_messages if m.get("role") == "assistant"
    ]
    execution_results_per_turn = final_state.execution_results_per_turn

    for turn_idx, assistant_msg in enumerate(assistant_messages):
        response = assistant_msg["content"]
        reasoning = extract_reasoning_from_response(response)

        code_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
        code = code_blocks[0] if code_blocks else ""

        exec_results = (
            execution_results_per_turn[turn_idx]
            if turn_idx < len(execution_results_per_turn)
            else []
        )

        if exec_results:
            result = exec_results[0]
            hooks = parse_hooks_from_stdout(result.stdout)
            execution = ExecutionResultDict(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                hooks=hooks,
                submitted_answer=result.submitted_answer,
            )
        else:
            execution = ExecutionResultDict(
                success=True,
                stdout="",
                stderr="",
                hooks=[],
                submitted_answer=None,
            )

        turns.append(
            TurnDict(
                turn_index=turn_idx,
                reasoning=reasoning,
                code=code,
                execution=execution,
            )
        )

    return TraceDict(
        turns=turns,
        final_answer=final_state.submitted_answer,
        final_answer_hash=hash_artifact(final_state.submitted_answer)
        if final_state.submitted_answer
        else None,
        success=final_state.submitted_answer is not None,
    )


def answers_match(
    hash1: str | None,
    hash2: str | None,
    val1: Any = None,
    val2: Any = None,
    float_tol: float = 0.1,
    p_value_tol: float = 0.002,
) -> bool:
    """
    Check if two answers match, with tolerance for floats and flexible types.

    Args:
        hash1, hash2: Answer hashes (for exact match)
        val1, val2: Raw answer values (for tolerant comparison)
        float_tol: Absolute tolerance for float comparison (default ±0.1)
        p_value_tol: Tighter tolerance for p-value comparison (default ±0.002)

    Returns:
        True if answers match within tolerance
    """
    # 1. Exact hash match (fast path)
    if hash1 is not None and hash2 is not None and hash1 == hash2:
        return True

    # 2. Tolerant comparison if we have raw values
    if val1 is not None and val2 is not None:
        # A. Direct DataFrame vs DataFrame (preserves sorting/testing logic)
        if isinstance(val1, pd.DataFrame) and isinstance(val2, pd.DataFrame):
            try:
                # 1. Sort columns
                df1 = val1.sort_index(axis=1)
                df2 = val2.sort_index(axis=1)

                if df1.shape != df2.shape:
                    return False

                # 2. Sort rows by values to handle reordering
                # We sort by all columns to be completely order-invariant
                df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
                df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

                pd.testing.assert_frame_equal(
                    df1,
                    df2,
                    check_dtype=False,
                    check_like=True,
                    atol=float_tol,
                    rtol=float_tol,
                )
                return True
            except (AssertionError, Exception):
                pass  # DataFrame comparison failed, fall through to normalization

        # B. Normalize and compare
        v1 = normalize_value(val1)
        v2 = normalize_value(val2)

        # Both floats
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return abs(float(v1) - float(v2)) <= float_tol

        # Both dicts: compare keys exactly, values with float tolerance
        if isinstance(v1, dict) and isinstance(v2, dict):

            def _values_match(a: Any, b: Any) -> bool:
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return abs(float(a) - float(b)) <= float_tol
                return answers_match(
                    None,
                    None,
                    a,
                    b,
                    float_tol=float_tol,
                    p_value_tol=p_value_tol,
                )

            special_keys = {"answer", "p_value"}
            has_special = any(k in v1 or k in v2 for k in special_keys)

            # Structured statistical answers: compare required keys AND all other fields
            if has_special:
                if set(v1.keys()) != set(v2.keys()):
                    return False

                if "answer" in v1:
                    if (
                        str(v1["answer"]).lower().strip()
                        != str(v2["answer"]).lower().strip()
                    ):
                        return False

                if "p_value" in v1:
                    try:
                        p1 = float(v1["p_value"])
                        p2 = float(v2["p_value"])
                        if abs(p1 - p2) > p_value_tol:
                            return False
                    except (ValueError, TypeError):
                        if str(v1["p_value"]) != str(v2["p_value"]):
                            return False

                for k in v1.keys():
                    if k in special_keys:
                        continue
                    if not _values_match(v1[k], v2[k]):
                        return False
                return True

            # Standard exact key match for other dicts
            if set(v1.keys()) != set(v2.keys()):
                return False
            for k in v1.keys():
                if not _values_match(v1[k], v2[k]):
                    return False
            return True

        # Both tuples/lists: compare element-wise
        if isinstance(v1, (tuple, list)) and isinstance(v2, (tuple, list)):
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if abs(float(a) - float(b)) > float_tol:
                        return False
                elif a != b:
                    return False
            return True

        # Exact equality for others
        try:
            if v1 == v2:
                return True
        except Exception:
            return False

    return False


def get_majority_answer(answers: list[Any], float_tol: float = 0.1) -> tuple[Any, int]:
    """
    Find the majority answer by clustering values using answers_match.

    Returns:
        Tuple of (majority_value, vote_count)
    """
    if not answers:
        return None, 0

    clusters = []  # list of [representative_value, count]
    for ans in answers:
        found = False
        for cluster in clusters:
            if answers_match(None, None, cluster[0], ans, float_tol=float_tol):
                cluster[1] += 1
                found = True
                break
        if not found:
            clusters.append([ans, 1])

    # Sort by count descending
    clusters.sort(key=lambda x: x[1], reverse=True)
    return clusters[0][0], clusters[0][1]


async def execute_teacher_trace(
    csv_path: str,
    question: str,
    model: str,
    *,
    hint: str | None = None,
    n_steps: int | None = None,
    difficulty: str | None = None,
    mode: str = "teacher-tutor",
    dataset_description: str = "",
    data_overview: str = "",
    max_turns: int = 10,
    sampling_args: dict | None = None,
    env=None,
    state: dict | None = None,
    reuse_env: bool = False,
    ui: Any,
    trace_mode: str = "gold",
    llm=None,
) -> tuple[TraceDict, list[dict], str, float]:
    """
    Execute a single teacher trace (with or without hint).

    Returns:
        Tuple of (TraceDict, conversation_messages, system_prompt, elapsed_seconds)
    """
    # Track timing
    start_time = time.time()

    ui.print_trace_start(trace_mode)

    env_instance = await Environment.from_params(
        csv_path=csv_path,
        model=model,
        question=question,
        hint=hint,
        n_steps=n_steps,
        difficulty=difficulty,
        mode=mode,
        dataset_description=dataset_description,
        data_overview=data_overview,
        max_turns=max_turns,
        sampling_args=sampling_args or {},
        env=env,
        state=state,
        reuse_env=reuse_env,
        llm=llm,
    )
    final_state = await env_instance.rollout()

    # Extract conversation for SFT training
    conversation_messages = final_state.conversation.to_openai_messages()
    system_prompt = conversation_messages[0]["content"] if conversation_messages else ""

    # Get code cells from environment (already extracted during execution)
    code_cells = final_state.code_cells

    # Extract assistant messages (used for turn counting and UI display)
    assistant_messages = [
        msg for msg in conversation_messages if msg.get("role") == "assistant"
    ]

    # Show full details for gold trace and consistency trace 1 (for visibility)
    # Other consistency traces just show summary to avoid clutter
    show_turns = trace_mode == "gold" or trace_mode.startswith("1/")

    if show_turns:
        import re

        code_pattern = r"```python\n(.*?)```"

        # Get execution results from final_state (stored during rollout)
        stored_results = final_state.execution_results_per_turn

        for i, msg in enumerate(assistant_messages, 1):
            response = msg["content"]
            # Extract code cells from this message
            turn_code_cells = re.findall(code_pattern, response, re.DOTALL)

            # Get execution results unless we are at the end of conversation
            if i - 1 < len(stored_results):
                turn_results = stored_results[i - 1]
                # Convert CodeCellResult objects to dicts for UI
                execution_results = [
                    {
                        "success": r.success,
                        "stdout": r.stdout,
                        "stderr": r.stderr,
                    }
                    for r in turn_results
                ]
            else:
                execution_results = []

            ui.print_turn(
                turn_num=i,
                max_turns=max_turns,
                response=response,
                code_cells=turn_code_cells,
                execution_results=execution_results,
            )
    else:
        # For consistency traces 2-5, just show summary
        ui.console.print(f"[dim]    Executed {len(assistant_messages)} turn(s)[/dim]")

    # Get final answer from environment's tracked submission
    final_answer = final_state.submitted_answer
    final_answer_hash = (
        hash_artifact(final_answer) if final_answer is not None else None
    )

    # Check if execution was successful (submitted an answer)
    execution_success = final_answer is not None

    # Calculate elapsed time
    elapsed_seconds = time.time() - start_time

    # Display trace completion
    ui.print_trace_complete(
        success=execution_success,
        final_answer=final_answer,
        turns=len(assistant_messages),
        elapsed_seconds=elapsed_seconds,
    )

    trace = build_trace_dict(final_state, conversation_messages)

    return trace, conversation_messages, system_prompt, elapsed_seconds


async def triangulate_teacher(
    csv_path: str,
    question: str,
    hint: str,
    model: str,
    *,
    n_steps: int | None = None,
    difficulty: str | None = None,
    n_consistency: int = 3,
    dataset_description: str = "",
    data_overview: str = "",
    max_turns: int = 10,
    sampling_args: dict | None = None,
    container_pool: list[tuple] | None = None,
    ui: Any,
    float_tol: float = 0.1,
    llm=None,
) -> tuple[
    TraceDict,
    list[dict],
    str,
    list[tuple[TraceDict, list[dict]]],
    bool,
    dict,
    str | None,
    int,
]:
    """
    Run teacher triangulation: gold trace + consistency traces.

    Args:
        container_pool: Optional list of (env, state) tuples for container reuse.
                       Should have 1 + n_consistency entries.
                       If None, creates new containers for each trace.

    Returns:
        Tuple of:
        - gold_trace: TraceDict WITH hint
        - gold_conversation: list[dict] of messages
        - system_prompt: str
        - consistency_results: list of (TraceDict, conversation) tuples
        - verified: True if gold matches majority of consistency traces
        - timing_metadata: dict with gold_elapsed, consistency_elapsed, total_elapsed, avg_elapsed
        - majority_answer_hash: hash of tolerance-based majority answer (if any)
        - majority_count: number of consistency traces in majority cluster
    """
    use_pool = container_pool is not None

    # 1. Run gold trace (with hint)
    ui.print_trace_header(mode="gold", hint=hint)

    if use_pool:
        assert container_pool is not None
        gold_env, gold_state = container_pool[0]
    else:
        gold_env, gold_state = None, None
    (
        gold_trace,
        gold_conversation,
        system_prompt,
        gold_elapsed,
    ) = await execute_teacher_trace(
        csv_path=csv_path,
        question=question,
        model=model,
        hint=hint,
        n_steps=n_steps,
        difficulty=difficulty,
        mode="teacher-tutor",
        dataset_description=dataset_description,
        data_overview=data_overview,
        max_turns=max_turns,
        sampling_args=sampling_args,
        env=gold_env,
        state=gold_state,
        reuse_env=use_pool,
        ui=ui,
        trace_mode="gold",
        llm=llm,
    )

    pool = container_pool or []

    async def run_consistency_trace(i: int):
        """Helper to run a single consistency trace."""
        ui.print_trace_header(mode=f"{i + 1}/{n_consistency}", hint=None)

        if use_pool:
            c_env, c_state = pool[i + 1]
        else:
            c_env, c_state = None, None
        trace, conversation, _, elapsed = await execute_teacher_trace(
            csv_path=csv_path,
            question=question,
            model=model,
            hint=None,
            n_steps=n_steps,
            difficulty=difficulty,
            mode="teacher-consistency",
            dataset_description=dataset_description,
            data_overview=data_overview,
            max_turns=max_turns,
            sampling_args=sampling_args,
            env=c_env,
            state=c_state,
            reuse_env=use_pool,
            ui=ui,
            trace_mode=f"{i + 1}/{n_consistency}",
            llm=llm,
        )
        return (trace, conversation, elapsed)

    # Run all consistency traces concurrently
    consistency_results_with_timing = await asyncio.gather(
        *[run_consistency_trace(i) for i in range(n_consistency)]
    )

    # Separate timing data from results
    consistency_results = [
        (trace, conv) for trace, conv, _ in consistency_results_with_timing
    ]
    consistency_elapsed = [elapsed for _, _, elapsed in consistency_results_with_timing]

    # Build timing metadata
    total_elapsed = gold_elapsed + sum(consistency_elapsed)
    timing_metadata = {
        "gold_elapsed": gold_elapsed,
        "consistency_elapsed": consistency_elapsed,
        "total_elapsed": total_elapsed,
        "avg_elapsed": total_elapsed / (1 + n_consistency),
    }

    submitted_answers = [
        trace["final_answer"]
        for trace, _ in consistency_results
        if trace["final_answer"] is not None
    ]

    if not submitted_answers:
        return (
            gold_trace,
            gold_conversation,
            system_prompt,
            consistency_results,
            False,
            timing_metadata,
            None,
            0,
        )

    # Find majority answer by clustering (handles float tolerance and formatting differences)
    majority_value, majority_count = get_majority_answer(
        submitted_answers, float_tol=float_tol
    )
    majority_answer_hash = (
        hash_artifact(majority_value) if majority_value is not None else None
    )

    verified = answers_match(
        None, None, gold_trace["final_answer"], majority_value, float_tol=float_tol
    )

    # Display triangulation result
    consistency_traces = [trace for trace, _ in consistency_results]
    ui.print_triangulation_result(
        gold_trace=gold_trace,
        consistency_traces=consistency_traces,
        verified=verified,
        float_tol=float_tol,
    )

    return (
        gold_trace,
        gold_conversation,
        system_prompt,
        consistency_results,
        verified,
        timing_metadata,
        majority_answer_hash,
        majority_count,
    )


async def batch_triangulate(
    csv_path: str,
    questions: List[dict],
    model: str,  # Must come from src.core.config
    *,  # Force remaining args to be keyword-only
    n_consistency: int = 3,
    dataset_description: str = "",
    data_overview: str = "",
    max_turns: int = 10,
    sampling_args: dict | None = None,
    use_container_pool: bool = True,  # Enable container reuse optimization
    ui: Any,  # UI instance for Rich output (required)
    float_tol: float = 0.1,
) -> list[
    tuple[
        dict,
        TraceDict,
        list[dict],
        str,
        list[tuple[TraceDict, list[dict]]],
        bool,
        dict,
        str | None,
        int,
    ]
]:
    """
    Run triangulation on a batch of questions.

    Uses MultiTenantContainer for memory-efficient parallel execution.
    Workers share memory via fork() and copy-on-write semantics.

    Args:
        csv_path: Path to CSV file
        questions: List of question dicts (from question_gen.py)
        n_consistency: Number of consistency traces per question
        model: Model identifier
        dataset_description: Dataset description
        data_overview: Data overview
        max_turns: Max turns per trace
        sampling_args: Model sampling args
        use_container_pool: If True, use multi-tenant container (much faster)

    Returns:
        List of tuples: (question_dict, gold_trace, gold_conversation, system_prompt, consistency_results, verified, timing_metadata, majority_answer_hash, majority_count)
    """
    from src.envs.container_pool import MultiTenantContainer, WorkerAdapter

    results = []
    verified_count = 0
    container = None
    container_pool = None

    # Create multi-tenant container if enabled
    if use_container_pool:
        n_workers = 1 + n_consistency  # 1 gold + N consistency traces

        # Cleanup existing containers first
        ui.base.print_status("Cleaning up old containers...")
        from src.utils.docker import cleanup_csv_sandbox_containers

        cleanup_csv_sandbox_containers()

        ui.base.print_status(
            f"Creating multi-tenant container ({n_workers} workers)..."
        )

        # Create single container with multiple workers (fork-based, memory shared)
        container = MultiTenantContainer(csv_path=csv_path, n_workers=n_workers)
        await container.start()

        # Create WorkerAdapters for triangulate_teacher compatibility
        container_pool = [
            (WorkerAdapter(container, i), WorkerAdapter.create_state(i))
            for i in range(n_workers)
        ]

        ui.base.print_success(f"✓ Container ready ({n_workers} workers)")

    try:
        for i, q_dict in enumerate(questions, 1):
            ui.print_question_header(q_num=i, total=len(questions), question=q_dict)

            (
                gold_trace,
                gold_conversation,
                system_prompt,
                consistency_results,
                verified,
                timing_metadata,
                majority_answer_hash,
                majority_count,
            ) = await triangulate_teacher(
                csv_path=csv_path,
                question=q_dict["question"],
                hint=q_dict.get("hint", ""),
                n_steps=q_dict.get("n_steps"),
                difficulty=q_dict.get("difficulty"),
                n_consistency=n_consistency,
                model=model,
                dataset_description=dataset_description,
                data_overview=data_overview,
                max_turns=max_turns,
                sampling_args=sampling_args,
                container_pool=container_pool,
                ui=ui,
                float_tol=float_tol,
            )

            results.append(
                (
                    q_dict,
                    gold_trace,
                    gold_conversation,
                    system_prompt,
                    consistency_results,
                    verified,
                    timing_metadata,
                    majority_answer_hash,
                    majority_count,
                )
            )

            if verified:
                verified_count += 1

            ui.print_progress_summary(
                current=i, total=len(questions), verified_count=verified_count
            )

            # Reset all workers between questions (faster than recreate)
            if container:
                await container.reset_all_workers()

    finally:
        # Clean up container
        if container:
            ui.base.print_status("Cleaning up container...")
            await container.stop()

    return results
