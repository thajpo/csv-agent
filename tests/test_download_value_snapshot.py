"""Tests for restoring the pinned value dataset into trainer JSONL files."""

from pathlib import Path

import pytest

from test_value_trainer import _record
from scripts.experiments.download_value_snapshot import write_snapshot


def test_snapshot_download_restores_validated_records(tmp_path: Path) -> None:
    row = {"record_json": _record().model_dump_json()}
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


def test_snapshot_download_rejects_normalized_duplicate_actions(
    tmp_path: Path,
) -> None:
    first = _record()
    first_turn = dict(first.prefix.turns[0])
    first_turn["code"] = (
        "missing_percentages = df.isna().mean()\nprint(missing_percentages)"
    )
    first_prefix = first.prefix.model_copy(
        update={"prefix_id": "episode-1:candidate-1", "turns": [first_turn]}
    )
    repeated = _record()
    repeated_turn = dict(repeated.prefix.turns[0])
    repeated_turn["code"] = "missing_percent = df.isna().mean()\nprint(missing_percent)"
    repeated_prefix = repeated.prefix.model_copy(
        update={"prefix_id": "episode-1:candidate-2", "turns": [repeated_turn]}
    )
    rows = [
        {
            "record_json": first.model_copy(
                update={"prefix": first_prefix}
            ).model_dump_json()
        },
        {
            "record_json": repeated.model_copy(
                update={"prefix": repeated_prefix}
            ).model_dump_json()
        },
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
