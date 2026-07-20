"""Tests for the minimal trajectory value trainer and selection evaluation."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from csv_spec import (
    ContinuationPolicy,
    PrefixContinuation,
    PrefixValueRecord,
    TrajectoryPrefix,
)
from src.value.dataset import (
    ValueExample,
    assert_dataset_disjoint,
    example_from_record,
    render_prefix,
)
from src.value.evaluation import prediction_metrics, selection_metrics
from src.value.trainer import TrainedValueModel
from scripts.experiments.evaluate_value_selection import evaluate
from scripts.experiments.train_value_model import train


def _example(
    prefix_id: str,
    episode_id: str,
    text: str,
    successes: int,
    *,
    selection_verdict: bool,
    dataset_id: str = "dataset-a",
) -> ValueExample:
    return ValueExample(
        prefix_id=prefix_id,
        episode_id=episode_id,
        dataset_id=dataset_id,
        text=text,
        successes=successes,
        attempts=4,
        turns_consumed=1,
        turns_left=3,
        execution_failed="error" in text,
        output_chars=len(text),
        selection_verdict=selection_verdict,
    )


def _record() -> PrefixValueRecord:
    response = "Inspect rows.\n```python\nprint(len(df))\n```"
    trace = {
        "turns": [],
        "final_answer": 10,
        "final_answer_hash": "hash",
        "success": True,
    }
    prefix = TrajectoryPrefix(
        prefix_id="episode-1:candidate-1",
        episode_id="episode-1",
        csv_source="data/kaggle/dataset-a/data.csv",
        system_prompt="Question: count rows. Private answer is not shown.",
        question_text="Count the rows.",
        turns=[
            {
                "turn_index": 0,
                "reasoning": "Inspect rows.",
                "code": "print(len(df))",
                "execution": {
                    "success": True,
                    "stdout": "10",
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": None,
                },
            }
        ],
        turn_responses=[response],
        turn_completed=[False],
        conversation_messages=[
            {"role": "assistant", "content": response},
            {"role": "user", "content": "Output: 10"},
        ],
        consumed_turns=1,
        max_turns=4,
    )
    outcomes = [
        PrefixContinuation(
            rollout_index=index,
            seed=index,
            trace=trace,
            verifier_verdict=verdict,
        )
        for index, verdict in enumerate([True, True, False, True])
    ]
    return PrefixValueRecord(
        prefix=prefix,
        policy=ContinuationPolicy(model="actor", sampling_args={}),
        continuations=outcomes,
        attempted_continuations=4,
        labeled_continuations=4,
        successful_continuations=3,
        value=0.75,
        code_commit="abc123",
    )


def test_record_reserves_one_unseen_selection_outcome() -> None:
    example = example_from_record(_record(), holdout_continuations=1)

    assert example.successes == 2
    assert example.attempts == 3
    assert example.selection_verdict is True
    assert example.dataset_id == "dataset-a"
    assert "Output: 10" in example.text
    assert "verifier_verdict" not in example.text
    assert "ground_truth_hash" not in example.text


def test_prefix_renderer_contains_only_seen_state_and_turns_left() -> None:
    prefix = _record().prefix

    rendered = render_prefix(prefix)

    assert "[SYSTEM]" in rendered
    assert "[ASSISTANT]" in rendered
    assert "[USER]" in rendered
    assert rendered.endswith("[TURNS LEFT]\n3")


def test_split_guard_rejects_shared_csv_dataset() -> None:
    example = _example("p1", "e1", "good", 4, selection_verdict=True)
    with pytest.raises(ValueError, match="multiple splits"):
        assert_dataset_disjoint([example], [], [example])


def test_value_model_overfits_fixture_and_reloads(tmp_path: Path) -> None:
    train = [
        _example("good-1", "e1", "correct column useful output", 4, selection_verdict=True),
        _example("good-2", "e2", "correct filter useful output", 4, selection_verdict=True),
        _example("bad-1", "e3", "error wrong missing column", 0, selection_verdict=False),
        _example("bad-2", "e4", "error irrelevant output", 0, selection_verdict=False),
    ]
    model = TrainedValueModel.fit(train, seed=7)
    scores = model.predict(train)

    assert min(scores[:2]) > max(scores[2:])

    checkpoint = tmp_path / "value.joblib"
    model.save(checkpoint)
    restored = TrainedValueModel.load(checkpoint)
    assert np.allclose(scores, restored.predict(train))


def test_metrics_measure_ranking_and_equal_call_selection() -> None:
    examples = [
        _example("a-good", "episode-a", "good", 4, selection_verdict=True),
        _example("a-bad", "episode-a", "bad", 0, selection_verdict=False),
        _example("b-good", "episode-b", "good", 3, selection_verdict=True),
        _example("b-bad", "episode-b", "bad", 1, selection_verdict=False),
    ]
    scores = [0.9, 0.1, 0.8, 0.2]

    prediction = prediction_metrics(examples, scores)
    selection = selection_metrics(examples, scores, seed=42)

    assert prediction["pairwise_ranking_accuracy"] == 1.0
    assert selection["value_guided_accuracy"] == 1.0
    assert selection["deployment_model_calls_per_episode"]["random"] == (
        selection["deployment_model_calls_per_episode"]["value_guided"]
    )


def _write_records(path: Path, dataset_id: str, episodes: int = 1) -> None:
    records = []
    for episode_index in range(episodes):
        for candidate_index in range(2):
            record = _record()
            episode_id = f"{dataset_id}-episode-{episode_index}"
            prefix = record.prefix.model_copy(
                update={
                    "prefix_id": f"{episode_id}:candidate-{candidate_index}",
                    "episode_id": episode_id,
                    "csv_source": f"data/kaggle/{dataset_id}/data.csv",
                    "system_prompt": (
                        "Useful correct calculation"
                        if candidate_index == 0
                        else "Irrelevant exploratory calculation"
                    ),
                }
            )
            records.append(record.model_copy(update={"prefix": prefix}))
    path.write_text("".join(item.model_dump_json() + "\n" for item in records))


def test_training_and_selection_commands_run_end_to_end(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_records(train_path, "train-dataset", episodes=2)
    _write_records(validation_path, "validation-dataset")
    _write_records(test_path, "test-dataset")
    output_dir = tmp_path / "model"

    results = train(
        Namespace(
            train=train_path,
            validation=validation_path,
            test=test_path,
            output_dir=output_dir,
            holdout_continuations=1,
            seed=42,
            max_features=1_000,
        )
    )
    selection = evaluate(
        Namespace(
            test=test_path,
            checkpoint=output_dir / "value_model.joblib",
            output=tmp_path / "selection.json",
            holdout_continuations=1,
            seed=42,
        )
    )

    assert results["splits"]["train"]["datasets"] == ["train-dataset"]
    assert (output_dir / "metrics.json").is_file()
    assert selection["episodes"] == 1
    assert (tmp_path / "selection.json").is_file()
