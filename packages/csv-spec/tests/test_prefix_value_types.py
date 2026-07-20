"""Contract tests for trajectory-prefix value records."""

import pytest
from pydantic import ValidationError

from csv_spec import (
    ContinuationPolicy,
    PrefixContinuation,
    PrefixValueRecord,
    TrajectoryPrefix,
)


def _turn(index: int, *, submitted_answer=None) -> dict:
    return {
        "turn_index": index,
        "reasoning": "Inspect the data.",
        "code": "print(df.shape)",
        "execution": {
            "success": True,
            "stdout": "(10, 2)",
            "stderr": "",
            "hooks": [],
            "submitted_answer": submitted_answer,
        },
    }


def _trace(success: bool) -> dict:
    return {
        "turns": [_turn(0)],
        "final_answer": 10 if success else None,
        "final_answer_hash": "answer-hash" if success else None,
        "success": success,
    }


def _response() -> str:
    return "Inspect the data.\n```python\nprint(df.shape)\n```"


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": _response()},
        {"role": "user", "content": "Execution completed."},
    ]


def test_prefix_excludes_private_verifier_fields() -> None:
    prefix = TrajectoryPrefix(
        prefix_id="episode-1:1",
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the CSV task using Python.",
        question_text="How many rows are present?",
        turns=[_turn(0)],
        turn_responses=[_response()],
        turn_completed=[False],
        conversation_messages=_messages(),
        consumed_turns=1,
        max_turns=3,
    )

    serialized = prefix.model_dump()

    assert "ground_truth" not in serialized
    assert "ground_truth_hash" not in serialized


def test_prefix_rejects_terminal_or_noncontiguous_turns() -> None:
    with pytest.raises(ValidationError, match="terminal turn"):
        TrajectoryPrefix(
            prefix_id="terminal",
            episode_id="episode-1",
            csv_source="dataset/data.csv",
            system_prompt="Solve the CSV task using Python.",
            question_text="How many rows are present?",
            turns=[_turn(0, submitted_answer=10)],
            turn_responses=[_response()],
            turn_completed=[True],
            conversation_messages=_messages(),
            consumed_turns=1,
            max_turns=3,
        )

    with pytest.raises(ValidationError, match="contiguous"):
        TrajectoryPrefix(
            prefix_id="gap",
            episode_id="episode-1",
            csv_source="dataset/data.csv",
            system_prompt="Solve the CSV task using Python.",
            question_text="How many rows are present?",
            turns=[_turn(1)],
            turn_responses=[_response()],
            turn_completed=[False],
            conversation_messages=_messages(),
            consumed_turns=1,
            max_turns=3,
        )


def test_value_record_requires_aggregate_to_match_verdicts() -> None:
    prefix = TrajectoryPrefix(
        prefix_id="episode-1:0",
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the CSV task using Python.",
        question_text="How many rows are present?",
        turns=[],
        turn_responses=[],
        turn_completed=[],
        conversation_messages=[],
        consumed_turns=0,
        max_turns=3,
    )
    policy = ContinuationPolicy(model="test-model", sampling_args={"temperature": 0.7})
    continuations = [
        PrefixContinuation(
            rollout_index=0,
            seed=10,
            trace=_trace(True),
            verifier_verdict=True,
        ),
        PrefixContinuation(
            rollout_index=1,
            seed=11,
            trace=_trace(False),
            verifier_verdict=False,
        ),
        PrefixContinuation(
            rollout_index=2,
            seed=12,
            error="sandbox unavailable",
        ),
    ]

    record = PrefixValueRecord(
        prefix=prefix,
        policy=policy,
        continuations=continuations,
        attempted_continuations=3,
        labeled_continuations=2,
        successful_continuations=1,
        value=None,
        code_commit="abc123",
        dataset_revision="hf-revision",
    )

    assert record.value is None
    assert record.model_validate_json(record.model_dump_json()) == record

    with pytest.raises(ValidationError, match="successful_continuations"):
        PrefixValueRecord(
            prefix=prefix,
            policy=policy,
            continuations=continuations,
            attempted_continuations=3,
            labeled_continuations=2,
            successful_continuations=2,
            value=None,
            code_commit="abc123",
        )


def test_prefix_requires_exact_turn_responses_and_boundary_messages() -> None:
    with pytest.raises(ValidationError, match="turn_responses"):
        TrajectoryPrefix(
            prefix_id="missing-response",
            episode_id="episode-1",
            csv_source="dataset/data.csv",
            system_prompt="Solve the CSV task using Python.",
            question_text="How many rows are present?",
            turns=[_turn(0)],
            turn_responses=[],
            turn_completed=[False],
            conversation_messages=_messages(),
            consumed_turns=1,
            max_turns=3,
        )

    with pytest.raises(ValidationError, match="turn boundary"):
        TrajectoryPrefix(
            prefix_id="wrong-boundary",
            episode_id="episode-1",
            csv_source="dataset/data.csv",
            system_prompt="Solve the CSV task using Python.",
            question_text="How many rows are present?",
            turns=[_turn(0)],
            turn_responses=[_response()],
            turn_completed=[False],
            conversation_messages=[
                {"role": "assistant", "content": "canonicalized response"},
                {"role": "user", "content": "Execution completed."},
            ],
            consumed_turns=1,
            max_turns=3,
        )


def test_prefix_allows_rejected_nonterminal_submission() -> None:
    prefix = TrajectoryPrefix(
        prefix_id="rejected-submission",
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the CSV task using Python.",
        question_text="How many rows are present?",
        turns=[_turn(0, submitted_answer={"rows": 10})],
        turn_responses=[_response()],
        turn_completed=[False],
        conversation_messages=_messages(),
        consumed_turns=1,
        max_turns=3,
    )

    assert prefix.turns[0]["execution"]["submitted_answer"] == {"rows": 10}


def test_prefix_tracks_consumed_turns_separately_from_executions() -> None:
    messages = [
        {"role": "assistant", "content": "Missing code block."},
        {"role": "user", "content": "Use one Python code block exactly."},
        *_messages(),
    ]
    prefix = TrajectoryPrefix(
        prefix_id="retried-prefix",
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the CSV task using Python.",
        question_text="How many rows are present?",
        turns=[_turn(0)],
        turn_responses=[_response()],
        turn_completed=[False],
        conversation_messages=messages,
        consumed_turns=2,
        max_turns=3,
    )

    assert len(prefix.turns) == 1
    assert prefix.consumed_turns == 2


def test_prefix_allows_pruned_conversation_history() -> None:
    messages = [
        {"role": "assistant", "content": f"Invalid response {index}"}
        for index in range(4)
        for _ in (0,)
    ]
    interleaved: list[dict[str, str]] = []
    for message in messages:
        interleaved.extend(
            [message, {"role": "user", "content": "Use one Python block."}]
        )
    interleaved.extend(_messages())

    prefix = TrajectoryPrefix(
        prefix_id="pruned-history",
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the CSV task using Python.",
        question_text="How many rows are present?",
        turns=[_turn(0)],
        turn_responses=[_response()],
        turn_completed=[False],
        conversation_messages=interleaved,
        consumed_turns=6,
        max_turns=8,
    )

    assert len(prefix.conversation_messages) == 10
    assert prefix.consumed_turns == 6
