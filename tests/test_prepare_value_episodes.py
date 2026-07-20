"""Tests for CSV-disjoint value experiment preparation."""

import json
from pathlib import Path

import pytest

from scripts.experiments.prepare_value_episodes import (
    normalize_snapshot_episode,
    prepare_episode_splits,
)


def _episode(serial: int, dataset_id: str, csv_source: str) -> str:
    payload = json.loads(Path("tests/fixtures/expected_episode.json").read_text())
    payload["episode_id"] = f"{dataset_id}-{serial}"
    payload["csv_source"] = csv_source
    return json.dumps(payload)


def test_prepared_splits_hold_out_complete_csv_datasets(tmp_path: Path) -> None:
    rows = []
    for dataset_index in range(4):
        dataset_id = f"dataset-{dataset_index}"
        csv = tmp_path / dataset_id / "data.csv"
        csv.parent.mkdir(parents=True)
        csv.write_text("a\n1\n")
        rows.extend(_episode(index, dataset_id, str(csv)) for index in range(3))

    splits, manifest = prepare_episode_splits(
        rows,
        local_data=tmp_path,
        dataset_counts={"train": 2, "validation": 1, "test": 1},
        episodes_per_dataset={"train": 2, "validation": 2, "test": 2},
        seed=42,
    )

    dataset_sets = [set(manifest["datasets"][split]) for split in splits]
    assert all(left.isdisjoint(right) for index, left in enumerate(dataset_sets) for right in dataset_sets[index + 1 :])
    assert {split: len(rows) for split, rows in splits.items()} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    assert all(Path(item.csv_source).is_file() for rows in splits.values() for item in rows)


def test_preparation_fails_when_too_few_datasets_are_available(tmp_path: Path) -> None:
    csv = tmp_path / "only" / "data.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("a\n1\n")
    rows = [_episode(index, "only", str(csv)) for index in range(2)]

    with pytest.raises(ValueError, match="need 3 datasets"):
        prepare_episode_splits(
            rows,
            local_data=tmp_path,
            dataset_counts={"train": 1, "validation": 1, "test": 1},
            episodes_per_dataset={"train": 1, "validation": 1, "test": 1},
            seed=42,
        )


def test_old_snapshot_diagnostics_are_regenerated_for_current_contract() -> None:
    payload = json.loads(Path("tests/fixtures/expected_episode.json").read_text())
    payload["triangulation"].pop("float_tolerance")
    payload.pop("process_report")

    episode = normalize_snapshot_episode(json.dumps(payload))

    assert episode.triangulation["float_tolerance"] == 0.1
    assert episode.process_report["steps"] == []
