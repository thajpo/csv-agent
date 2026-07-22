"""Tests for CSV-disjoint value experiment preparation."""

import json
from pathlib import Path

import pytest

from scripts.experiments.prepare_value_episodes import (
    normalize_snapshot_episode,
    prepare_episode_splits,
)


def _episode(
    serial: int,
    dataset_id: str,
    csv_source: str,
    *,
    template_name: str | None = None,
) -> str:
    payload = json.loads(Path("tests/fixtures/expected_episode.json").read_text())
    payload["episode_id"] = f"{dataset_id}-{serial}"
    payload["csv_source"] = csv_source
    if template_name is not None:
        payload["question"]["template_name"] = template_name
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
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(dataset_sets)
        for right in dataset_sets[index + 1 :]
    )
    assert {split: len(rows) for split, rows in splits.items()} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    assert all(
        Path(item.csv_source).is_file() for rows in splits.values() for item in rows
    )


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


def test_preparation_assigns_datasets_by_each_split_requirement(tmp_path: Path) -> None:
    rows = []
    for dataset_id, episode_count in (("train-ready", 6), ("test-ready", 8)):
        csv = tmp_path / dataset_id / "data.csv"
        csv.parent.mkdir(parents=True)
        csv.write_text("a\n1\n")
        rows.extend(
            _episode(index, dataset_id, str(csv)) for index in range(episode_count)
        )

    splits, manifest = prepare_episode_splits(
        rows,
        local_data=tmp_path,
        dataset_counts={"train": 1, "validation": 0, "test": 1},
        episodes_per_dataset={"train": 6, "validation": 4, "test": 8},
        seed=42,
    )

    assert manifest["datasets"] == {
        "train": ["train-ready"],
        "validation": [],
        "test": ["test-ready"],
    }
    assert {split: len(episodes) for split, episodes in splits.items()} == {
        "train": 6,
        "validation": 0,
        "test": 8,
    }


def test_preparation_excludes_datasets_and_covers_eval_templates(
    tmp_path: Path,
) -> None:
    rows = []
    for dataset_id in ("dataset-a", "dataset-b", "excluded"):
        csv = tmp_path / dataset_id / "data.csv"
        csv.parent.mkdir(parents=True)
        csv.write_text("a\n1\n")
        rows.extend(
            _episode(
                index,
                dataset_id,
                str(csv),
                template_name=("shared" if index == 0 else f"{dataset_id}-{index}"),
            )
            for index in range(3)
        )

    splits, manifest = prepare_episode_splits(
        rows,
        local_data=tmp_path,
        dataset_counts={"train": 1, "validation": 0, "test": 1},
        episodes_per_dataset={"train": 2, "validation": 1, "test": 1},
        seed=7,
        excluded_datasets={"excluded"},
        require_evaluation_templates_in_train=True,
    )

    train_templates = {episode.question["template_name"] for episode in splits["train"]}
    test_templates = {episode.question["template_name"] for episode in splits["test"]}
    assert manifest["excluded_datasets"] == ["excluded"]
    assert test_templates <= train_templates
    assert "excluded" not in {
        dataset_id
        for dataset_ids in manifest["datasets"].values()
        for dataset_id in dataset_ids
    }


def test_old_snapshot_diagnostics_are_regenerated_for_current_contract() -> None:
    payload = json.loads(Path("tests/fixtures/expected_episode.json").read_text())
    payload["triangulation"].pop("float_tolerance")
    payload.pop("process_report")

    episode = normalize_snapshot_episode(json.dumps(payload))

    assert episode.triangulation["float_tolerance"] == 0.1
    assert episode.process_report["steps"] == []
