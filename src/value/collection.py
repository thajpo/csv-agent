"""Collect verifier-grounded future-success values for trajectory prefixes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from csv_spec import (
    ContinuationPolicy,
    PrefixContinuation,
    PrefixValueRecord,
    TraceDict,
    TrajectoryPrefix,
    hash_artifact,
)

from src.core.environment import Environment
from src.core.model import APILLM
from src.datagen.shared.verification import derive_ground_truth_verification
from src.datagen.teacher import build_trace_dict

ContinuationRunner = Callable[
    [TrajectoryPrefix, ContinuationPolicy, int, int | None],
    Awaitable[TraceDict],
]


@dataclass(frozen=True)
class InitialModelTrace:
    trace: TraceDict
    system_prompt: str
    turn_responses: list[str]
    turn_completed: list[bool]
    boundary_messages: list[list[dict[str, str]]]


def build_trajectory_prefix(
    *,
    episode_id: str,
    csv_source: str,
    system_prompt: str,
    question_text: str,
    trace: TraceDict,
    turn_responses: Sequence[str],
    turn_completed: Sequence[bool],
    conversation_messages: Sequence[Mapping[str, str]],
    turn_count: int,
    max_turns: int,
) -> TrajectoryPrefix:
    """Create a deterministic, oracle-free prefix from a recorded boundary."""
    turns = trace.get("turns", [])
    if turn_count < 0 or turn_count > len(turns):
        raise ValueError("turn_count is outside the recorded trace")
    if len(turn_responses) < turn_count:
        raise ValueError("turn responses are unavailable at the selected boundary")
    if len(turn_completed) < turn_count:
        raise ValueError("turn completion states are unavailable at the boundary")
    selected_turns = deepcopy(turns[:turn_count])
    selected_responses = list(turn_responses[:turn_count])
    selected_completion = list(turn_completed[:turn_count])
    selected_messages = [dict(message) for message in conversation_messages]
    identity = hash_artifact(
        {
            "episode_id": episode_id,
            "csv_source": csv_source,
            "system_prompt": system_prompt,
            "question_text": question_text,
            "turns": selected_turns,
            "turn_responses": selected_responses,
            "turn_completed": selected_completion,
            "conversation_messages": selected_messages,
            "max_turns": max_turns,
        }
    )
    return TrajectoryPrefix(
        prefix_id=f"{episode_id}:{turn_count}:{identity}",
        episode_id=episode_id,
        csv_source=csv_source,
        system_prompt=system_prompt,
        question_text=question_text,
        turns=selected_turns,
        turn_responses=selected_responses,
        turn_completed=selected_completion,
        conversation_messages=selected_messages,
        max_turns=max_turns,
    )


def verify_terminal_trace(
    question: Mapping[str, Any], trace: TraceDict, *, float_tolerance: float
) -> bool:
    """Label a model continuation using only the existing terminal verifier."""
    if not trace.get("success") or trace.get("final_answer") is None:
        return False

    evidence = derive_ground_truth_verification(
        question=question,
        gold_trace=trace,
        float_tolerance=float_tolerance,
    )
    if evidence.verdict is None:
        raise ValueError("terminal verifier could not label a submitted answer")
    return evidence.verdict


async def run_initial_model_trace(
    *,
    csv_source: str,
    question_text: str,
    policy: ContinuationPolicy,
    max_turns: int,
    seed: int | None,
) -> InitialModelTrace:
    """Sample the actor once and retain each exact public turn boundary."""
    sampling_args = dict(policy.sampling_args)
    if seed is not None:
        sampling_args["seed"] = seed
    llm = APILLM(model=policy.model, sampling_args=sampling_args)
    try:
        environment = await Environment.from_params(
            csv_path=csv_source,
            model=policy.model,
            question=question_text,
            mode="student",
            max_turns=max_turns,
            sampling_args=policy.sampling_args,
            llm=llm,
            session_id="value-source",
        )
        final_state = await environment.rollout()
        trace = build_trace_dict(final_state)
        execution_turns = final_state.execution_turns
        return InitialModelTrace(
            trace=trace,
            system_prompt=final_state.conversation.system_prompt,
            turn_responses=[record["response"] for record in execution_turns],
            turn_completed=[record["completed"] for record in execution_turns],
            boundary_messages=[
                deepcopy(record["conversation_messages"]) for record in execution_turns
            ],
        )
    finally:
        await llm.aclose()


async def run_model_continuation(
    prefix: TrajectoryPrefix,
    policy: ContinuationPolicy,
    rollout_index: int,
    seed: int | None,
) -> TraceDict:
    """Replay a prefix and continue it once with the configured actor."""
    sampling_args = dict(policy.sampling_args)
    if seed is not None:
        sampling_args["seed"] = seed
    llm = APILLM(model=policy.model, sampling_args=sampling_args)
    try:
        environment = await Environment.from_params(
            csv_path=prefix.csv_source,
            model=policy.model,
            question=prefix.question_text,
            mode="student",
            max_turns=prefix.max_turns,
            sampling_args=policy.sampling_args,
            llm=llm,
            session_id=f"value-{prefix.prefix_id[-8:]}-{rollout_index}",
        )
        final_state = await environment.rollout_from_prefix(prefix)
        return build_trace_dict(final_state)
    finally:
        await llm.aclose()


async def collect_prefix_value(
    *,
    prefix: TrajectoryPrefix,
    question: Mapping[str, Any],
    policy: ContinuationPolicy,
    seeds: Sequence[int | None],
    float_tolerance: float,
    code_commit: str,
    dataset_revision: str | None = None,
    runner: ContinuationRunner = run_model_continuation,
) -> PrefixValueRecord:
    """Estimate success over all independently attempted continuations.

    Terminal submissions are labeled only by the ground-truth verifier.
    Replay, rollout, and verifier errors are retained without verdicts and
    contribute zero to the value's attempted-continuation denominator.
    """
    if not seeds:
        raise ValueError("at least one continuation seed is required")

    continuations: list[PrefixContinuation] = []
    for rollout_index, seed in enumerate(seeds):
        try:
            trace = await runner(prefix, policy, rollout_index, seed)
        except Exception as error:
            continuations.append(
                PrefixContinuation(
                    rollout_index=rollout_index,
                    seed=seed,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            continue

        try:
            verdict = verify_terminal_trace(
                question,
                trace,
                float_tolerance=float_tolerance,
            )
            continuations.append(
                PrefixContinuation(
                    rollout_index=rollout_index,
                    seed=seed,
                    trace=trace,
                    verifier_verdict=verdict,
                )
            )
        except Exception as error:
            continuations.append(
                PrefixContinuation(
                    rollout_index=rollout_index,
                    seed=seed,
                    trace=trace,
                    error=f"{type(error).__name__}: {error}",
                )
            )

    labeled = sum(item.verifier_verdict is not None for item in continuations)
    successes = sum(item.verifier_verdict is True for item in continuations)
    return PrefixValueRecord(
        prefix=prefix,
        policy=policy,
        continuations=continuations,
        attempted_continuations=len(continuations),
        labeled_continuations=labeled,
        successful_continuations=successes,
        value=successes / len(continuations),
        code_commit=code_commit,
        dataset_revision=dataset_revision,
    )
