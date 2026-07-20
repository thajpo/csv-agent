"""Tests for restoring the pinned value dataset into trainer JSONL files."""

from pathlib import Path

import pytest

from tests.test_value_trainer import _record
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
