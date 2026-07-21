"""Download and validate a pinned prefix-value dataset from Hugging Face."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Mapping, Sequence

from csv_spec import PrefixValueRecord
from scripts.experiments.collect_prefix_values import prefix_action_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download validated PrefixValueRecord files from Hugging Face."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/datasets/value-canary.toml"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def write_snapshot(
    dataset: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    output_dir: Path,
    repo: str,
    revision: str,
    expected_records: Mapping[str, int] | None = None,
) -> dict:
    counts: dict[str, int] = {}
    records_by_split: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        if split not in dataset:
            raise ValueError(f"value snapshot is missing the {split} split")
        records: list[str] = []
        actions_by_episode: dict[str, set[str]] = {}
        collection_contract = None
        for row_index, row in enumerate(dataset[split]):
            serialized = row.get("record_json")
            if not isinstance(serialized, str):
                raise ValueError(f"{split} row {row_index} has no record_json")
            record = PrefixValueRecord.model_validate_json(serialized)
            episode_id = record.prefix.episode_id
            if record.collection_contract is None:
                raise ValueError(f"{split} row {row_index} has no collection contract")
            if (
                record.value is None
                or record.labeled_continuations != record.attempted_continuations
                or record.attempted_continuations
                != record.collection_contract.continuations
            ):
                raise ValueError(
                    f"{split} row {row_index} is not a complete labeled candidate"
                )
            if collection_contract is None:
                collection_contract = record.collection_contract
            elif record.collection_contract != collection_contract:
                raise ValueError(f"{split} does not share one collection contract")
            action_id = prefix_action_identity(record.prefix.turns)
            episode_actions = actions_by_episode.setdefault(episode_id, set())
            if action_id in episode_actions:
                raise ValueError(
                    f"{split} repeats a first action for episode {episode_id}"
                )
            episode_actions.add(action_id)
            records.append(record.model_dump_json())
        if collection_contract is not None:
            expected_candidates = collection_contract.candidates_per_episode
            for episode_id, actions in actions_by_episode.items():
                if len(actions) != expected_candidates:
                    raise ValueError(
                        f"{split} episode {episode_id} has {len(actions)} candidates; "
                        f"expected {expected_candidates}"
                    )
        records_by_split[split] = records
        counts[split] = len(records)

    if expected_records is not None:
        expected_counts = {
            split: int(expected_records[split])
            for split in ("train", "validation", "test")
        }
        if counts != expected_counts:
            raise ValueError(
                f"value snapshot record counts {counts} do not match "
                f"expected {expected_counts}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in records_by_split.items():
        (output_dir / f"{split}-values.jsonl").write_text(
            "".join(record + "\n" for record in records)
        )
    manifest = {"repo": repo, "revision": revision, "records": counts}
    (output_dir / "snapshot.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def download(args: argparse.Namespace) -> dict:
    from datasets import load_dataset

    with args.dataset_config.open("rb") as handle:
        config = tomllib.load(handle)
    expected_records = config.get("expected_records")
    if not isinstance(expected_records, dict):
        raise ValueError("value snapshot config must declare expected_records")
    dataset = load_dataset(config["repo"], revision=config["revision"], token=True)
    return write_snapshot(
        dataset,
        output_dir=args.output_dir,
        repo=config["repo"],
        revision=config["revision"],
        expected_records=expected_records,
    )


def main() -> None:
    print(json.dumps(download(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
