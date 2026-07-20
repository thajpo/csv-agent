"""Tests for verifier-grounded trajectory-prefix value collection."""

import asyncio

import pytest
from csv_spec import ContinuationPolicy, hash_artifact

from src.value.collection import (
    build_trajectory_prefix,
    collect_prefix_value,
    run_initial_model_trace,
)


def _turn(index: int, code: str, *, submitted_answer=None) -> dict:
    return {
        "turn_index": index,
        "reasoning": "Perform the next required computation.",
        "code": code,
        "execution": {
            "success": True,
            "stdout": "ok",
            "stderr": "",
            "hooks": [],
            "submitted_answer": submitted_answer,
        },
    }


def _trace(answer=None) -> dict:
    return {
        "turns": [
            _turn(0, "print(df.columns)"),
            _turn(1, "submit(answer)", submitted_answer=answer),
        ],
        "final_answer": answer,
        "final_answer_hash": hash_artifact(answer) if answer is not None else None,
        "success": answer is not None,
    }


def _response(code: str) -> str:
    return f"Perform the next required computation.\n```python\n{code}\n```"


def _boundary_messages(code: str) -> list[dict[str, str]]:
    response = _response(code)
    return [
        {"role": "assistant", "content": response},
        {"role": "user", "content": "Execution completed."},
    ]


def test_build_prefix_is_deterministic_and_excludes_terminal_turn() -> None:
    first = build_trajectory_prefix(
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the task.",
        question_text="How many rows are present?",
        trace=_trace(3),
        turn_responses=[_response("print(df.columns)")],
        turn_completed=[False],
        conversation_messages=_boundary_messages("print(df.columns)"),
        turn_count=1,
        consumed_turns=1,
        max_turns=3,
    )
    second = build_trajectory_prefix(
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the task.",
        question_text="How many rows are present?",
        trace=_trace(3),
        turn_responses=[_response("print(df.columns)")],
        turn_completed=[False],
        conversation_messages=_boundary_messages("print(df.columns)"),
        turn_count=1,
        consumed_turns=1,
        max_turns=3,
    )

    assert first.prefix_id == second.prefix_id
    assert len(first.turns) == 1
    assert first.turns[0]["execution"]["submitted_answer"] is None
    assert "ground_truth" not in first.model_dump()


@pytest.mark.asyncio
async def test_collection_with_runner_error_has_no_value_estimate() -> None:
    expected_answer = 3
    question = {
        "question_text": "How many rows are present?",
        "ground_truth": expected_answer,
        "ground_truth_hash": hash_artifact(expected_answer),
    }
    prefix = build_trajectory_prefix(
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the task.",
        question_text=question["question_text"],
        trace=_trace(expected_answer),
        turn_responses=[_response("print(df.columns)")],
        turn_completed=[False],
        conversation_messages=_boundary_messages("print(df.columns)"),
        turn_count=1,
        consumed_turns=1,
        max_turns=3,
    )
    policy = ContinuationPolicy(
        model="test-model",
        sampling_args={"temperature": 0.7, "top_p": 1.0},
    )

    async def runner(_prefix, _policy, _rollout_index, seed):
        if seed == 10:
            return _trace(expected_answer)
        if seed == 11:
            return _trace(99)
        if seed == 12:
            return _trace()
        raise RuntimeError("sandbox unavailable")

    record = await collect_prefix_value(
        prefix=prefix,
        question=question,
        policy=policy,
        seeds=[10, 11, 12, 13],
        float_tolerance=0.1,
        code_commit="abc123",
        dataset_revision="hf-revision",
        runner=runner,
    )

    assert record.attempted_continuations == 4
    assert record.labeled_continuations == 3
    assert record.successful_continuations == 1
    assert record.value is None
    assert [item.verifier_verdict for item in record.continuations] == [
        True,
        False,
        False,
        None,
    ]
    assert record.continuations[-1].error == "RuntimeError: sandbox unavailable"
    assert record.dataset_revision == "hf-revision"


@pytest.mark.asyncio
async def test_independent_continuations_run_concurrently() -> None:
    active = 0
    peak = 0

    async def runner(_prefix, _policy, _rollout_index, _seed):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _trace(3)

    prefix = build_trajectory_prefix(
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the task.",
        question_text="How many rows are present?",
        trace=_trace(3),
        turn_responses=[],
        turn_completed=[],
        conversation_messages=[],
        turn_count=0,
        consumed_turns=0,
        max_turns=3,
    )
    record = await collect_prefix_value(
        prefix=prefix,
        question={
            "question_text": "How many rows are present?",
            "ground_truth": 3,
            "ground_truth_hash": hash_artifact(3),
        },
        policy=ContinuationPolicy(model="test-model", sampling_args={}),
        seeds=[1, 2, 3],
        float_tolerance=0.1,
        code_commit="abc123",
        runner=runner,
    )

    assert peak == 3
    assert record.value == 1.0


@pytest.mark.asyncio
async def test_initial_traces_use_isolated_source_sessions(monkeypatch) -> None:
    session_ids: list[str] = []

    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

        async def aclose(self):
            pass

    class FakeState:
        execution_turns = []

        class Conversation:
            system_prompt = "prompt"

        conversation = Conversation()

    class FakeEnvironment:
        @classmethod
        async def from_params(cls, **kwargs):
            session_ids.append(kwargs["session_id"])
            return cls()

        async def rollout(self):
            return FakeState()

    monkeypatch.setattr("src.value.collection.APILLM", FakeLLM)
    monkeypatch.setattr("src.value.collection.Environment", FakeEnvironment)
    monkeypatch.setattr(
        "src.value.collection.build_trace_dict", lambda _state: {"turns": []}
    )

    policy = ContinuationPolicy(model="test-model", sampling_args={})
    await asyncio.gather(
        run_initial_model_trace(
            csv_source="one.csv",
            question_text="Question one?",
            policy=policy,
            max_turns=1,
            seed=1,
        ),
        run_initial_model_trace(
            csv_source="two.csv",
            question_text="Question two?",
            policy=policy,
            max_turns=1,
            seed=2,
        ),
    )

    assert len(set(session_ids)) == 2
    assert all(session_id.startswith("value-source-") for session_id in session_ids)


@pytest.mark.asyncio
async def test_verifier_failure_is_recorded_without_a_value_label() -> None:
    prefix = build_trajectory_prefix(
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the task.",
        question_text="How many rows are present?",
        trace=_trace(3),
        turn_responses=[],
        turn_completed=[],
        conversation_messages=[],
        turn_count=0,
        consumed_turns=0,
        max_turns=3,
    )

    async def runner(_prefix, _policy, _rollout_index, _seed):
        return _trace(3)

    record = await collect_prefix_value(
        prefix=prefix,
        question={"question_text": "How many rows are present?"},
        policy=ContinuationPolicy(model="test-model", sampling_args={}),
        seeds=[1],
        float_tolerance=0.1,
        code_commit="abc123",
        runner=runner,
    )

    assert record.value is None
    assert record.labeled_continuations == 0
    assert record.continuations[0].trace == _trace(3)
    assert "ground-truth hash provenance is unavailable" in (
        record.continuations[0].error or ""
    )


@pytest.mark.asyncio
async def test_no_submission_still_requires_verifier_provenance() -> None:
    prefix = build_trajectory_prefix(
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the task.",
        question_text="How many rows are present?",
        trace=_trace(),
        turn_responses=[],
        turn_completed=[],
        conversation_messages=[],
        turn_count=0,
        consumed_turns=0,
        max_turns=3,
    )

    async def runner(_prefix, _policy, _rollout_index, _seed):
        return _trace()

    record = await collect_prefix_value(
        prefix=prefix,
        question={"question_text": "How many rows are present?"},
        policy=ContinuationPolicy(model="test-model", sampling_args={}),
        seeds=[1],
        float_tolerance=0.1,
        code_commit="abc123",
        runner=runner,
    )

    assert record.value is None
    assert record.continuations[0].verifier_verdict is None
    assert "ground-truth hash provenance is unavailable" in (
        record.continuations[0].error or ""
    )
