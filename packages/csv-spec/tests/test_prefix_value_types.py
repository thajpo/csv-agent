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


def test_prefix_excludes_private_verifier_fields() -> None:
    prefix = TrajectoryPrefix(
        prefix_id="episode-1:1",
        episode_id="episode-1",
        csv_source="dataset/data.csv",
        system_prompt="Solve the CSV task using Python.",
        question_text="How many rows are present?",
        turns=[_turn(0)],
        max_turns=3,
    )

    serialized = prefix.model_dump()

    assert "ground_truth" not in serialized
    assert "ground_truth_hash" not in serialized


def test_prefix_rejects_terminal_or_noncontiguous_turns() -> None:
    with pytest.raises(ValidationError, match="terminal submission"):
        TrajectoryPrefix(
            prefix_id="terminal",
            episode_id="episode-1",
            csv_source="dataset/data.csv",
            system_prompt="Solve the CSV task using Python.",
            question_text="How many rows are present?",
            turns=[_turn(0, submitted_answer=10)],
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
        value=0.5,
        code_commit="abc123",
        dataset_revision="hf-revision",
    )

    assert record.value == 0.5
    assert record.model_validate_json(record.model_dump_json()) == record

    with pytest.raises(ValidationError, match="successful_continuations"):
        PrefixValueRecord(
            prefix=prefix,
            policy=policy,
            continuations=continuations,
            attempted_continuations=3,
            labeled_continuations=2,
            successful_continuations=2,
            value=1.0,
            code_commit="abc123",
        )
