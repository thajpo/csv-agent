"""Tests for restoring the pinned value dataset into trainer JSONL files."""

from pathlib import Path

import pytest
from csv_spec import PrefixValueCollectionContract, PrefixValueRecord

from test_value_trainer import _record
from scripts.experiments.download_value_snapshot import write_snapshot


def _snapshot_record(
    *,
    episode_id: str = "episode-1",
    candidate_id: str = "candidate-1",
    code: str = "print(len(df))",
    candidates_per_episode: int = 1,
    seed: int = 42,
):
    record = _record()
    response = f"Inspect rows.\n```python\n{code}\n```"
    turn = dict(record.prefix.turns[0])
    turn["code"] = code
    messages = [dict(message) for message in record.prefix.conversation_messages]
    messages[-2]["content"] = response
    prefix = record.prefix.model_copy(
        update={
            "prefix_id": f"{episode_id}:{candidate_id}",
            "episode_id": episode_id,
            "turns": [turn],
            "turn_responses": [response],
            "conversation_messages": messages,
        }
    )
    contract = PrefixValueCollectionContract(
        code_commit=record.code_commit,
        dataset_revision="revision",
        policy=record.policy,
        source_system_prompt_suffix="Take one intermediate action.",
        source_initial_user_message="Begin.",
        episode_inputs_hash="inputs-hash",
        turn_count=1,
        max_turns=record.prefix.max_turns,
        continuations=record.attempted_continuations,
        candidates_per_episode=candidates_per_episode,
        seed=seed,
        float_tolerance=0.1,
    )
    return record.model_copy(
        update={
            "prefix": prefix,
            "dataset_revision": contract.dataset_revision,
            "collection_contract": contract,
        }
    )


def test_snapshot_download_restores_validated_records(tmp_path: Path) -> None:
    row = {"record_json": _snapshot_record().model_dump_json()}
    dataset = {"train": [row], "validation": [row], "test": [row]}

    manifest = write_snapshot(
        dataset,
        output_dir=tmp_path,
        repo="owner/value-data",
        revision="abc123",
    )

    assert manifest["records"] == {"train": 1, "validation": 1, "test": 1}
    assert (tmp_path / "train-values.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "snapshot.json").is_file()


def test_snapshot_download_rejects_missing_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing the test split"):
        write_snapshot(
            {"train": [], "validation": []},
            output_dir=tmp_path,
            repo="owner/value-data",
            revision="abc123",
        )


def test_snapshot_download_rejects_unexpected_split_counts(tmp_path: Path) -> None:
    row = {"record_json": _snapshot_record().model_dump_json()}

    with pytest.raises(ValueError, match="do not match expected"):
        write_snapshot(
            {"train": [row], "validation": [row], "test": [row]},
            output_dir=tmp_path,
            repo="owner/value-data",
            revision="abc123",
            expected_records={"train": 2, "validation": 1, "test": 1},
        )

    assert list(tmp_path.iterdir()) == []


def test_snapshot_download_rejects_incomplete_candidates(tmp_path: Path) -> None:
    payload = _snapshot_record().model_dump(mode="json")
    payload["continuations"][0]["verifier_verdict"] = None
    payload["continuations"][0]["error"] = "verifier failed"
    payload["labeled_continuations"] = 3
    payload["successful_continuations"] = 2
    payload["value"] = None
    record = PrefixValueRecord.model_validate(payload)

    with pytest.raises(ValueError, match="not a complete labeled candidate"):
        write_snapshot(
            {
                "train": [{"record_json": record.model_dump_json()}],
                "validation": [],
                "test": [],
            },
            output_dir=tmp_path,
            repo="owner/value-data",
            revision="abc123",
        )

    assert list(tmp_path.iterdir()) == []


def test_snapshot_download_rejects_missing_collection_contract(tmp_path: Path) -> None:
    row = {"record_json": _record().model_dump_json()}

    with pytest.raises(ValueError, match="has no collection contract"):
        write_snapshot(
            {"train": [row], "validation": [], "test": []},
            output_dir=tmp_path,
            repo="owner/value-data",
            revision="abc123",
        )

    assert list(tmp_path.iterdir()) == []


def test_snapshot_download_rejects_wrong_source_revision(tmp_path: Path) -> None:
    row = {"record_json": _snapshot_record().model_dump_json()}

    with pytest.raises(ValueError, match="does not match expected source revision"):
        write_snapshot(
            {"train": [row], "validation": [row], "test": [row]},
            output_dir=tmp_path,
            repo="owner/value-data",
            revision="abc123",
            expected_source_revision="different-revision",
        )

    assert list(tmp_path.iterdir()) == []


def test_snapshot_download_requires_complete_candidate_groups(tmp_path: Path) -> None:
    row = {"record_json": _snapshot_record(candidates_per_episode=2).model_dump_json()}

    with pytest.raises(
        ValueError, match="episode episode-1 has 1 candidates; expected 2"
    ):
        write_snapshot(
            {"train": [row], "validation": [], "test": []},
            output_dir=tmp_path,
            repo="owner/value-data",
            revision="abc123",
        )

    assert list(tmp_path.iterdir()) == []


def test_snapshot_download_requires_one_contract_per_split(tmp_path: Path) -> None:
    rows = [
        {"record_json": _snapshot_record(episode_id="episode-1").model_dump_json()},
        {
            "record_json": _snapshot_record(
                episode_id="episode-2", seed=99
            ).model_dump_json()
        },
    ]

    with pytest.raises(ValueError, match="does not share one collection contract"):
        write_snapshot(
            {"train": rows, "validation": [], "test": []},
            output_dir=tmp_path,
            repo="owner/value-data",
            revision="abc123",
        )

    assert list(tmp_path.iterdir()) == []


def test_snapshot_download_rejects_normalized_duplicate_actions(
    tmp_path: Path,
) -> None:
    first = _snapshot_record(
        candidates_per_episode=2,
        code="missing_percentages = df.isna().mean()\nprint(missing_percentages)",
    )
    repeated = _snapshot_record(
        candidate_id="candidate-2",
        candidates_per_episode=2,
        code="missing_percent = df.isna().mean()\nprint(missing_percent)",
    )
    rows = [
        {"record_json": first.model_dump_json()},
        {"record_json": repeated.model_dump_json()},
    ]

    with pytest.raises(
        ValueError, match="repeats a first action for episode episode-1"
    ):
        write_snapshot(
            {"train": rows, "validation": [], "test": []},
            output_dir=tmp_path,
            repo="owner/value-data",
            revision="abc123",
        )

    assert list(tmp_path.iterdir()) == []
