"""Create CSV-disjoint episode files from the pinned Hugging Face snapshot."""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from csv_spec import EpisodeJSONL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train, validation, and test episodes for value research."
    )
    parser.add_argument(
        "--dataset-config", type=Path, default=Path("configs/datasets/template.toml")
    )
    parser.add_argument("--local-data", type=Path, default=Path("data/kaggle"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-datasets", type=int, default=6)
    parser.add_argument("--validation-datasets", type=int, default=2)
    parser.add_argument("--test-datasets", type=int, default=2)
    parser.add_argument("--train-episodes-per-dataset", type=int, default=6)
    parser.add_argument("--validation-episodes-per-dataset", type=int, default=4)
    parser.add_argument("--test-episodes-per-dataset", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_episode_splits(
    episode_json: Iterable[str],
    *,
    local_data: Path,
    dataset_counts: dict[str, int],
    episodes_per_dataset: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[EpisodeJSONL]], dict]:
    grouped: dict[str, list[EpisodeJSONL]] = defaultdict(list)
    seen_episode_ids: set[str] = set()
    minimum_examples = max(episodes_per_dataset.values())
    for serialized in episode_json:
        episode = EpisodeJSONL.model_validate_json(serialized)
        if episode.episode_id in seen_episode_ids:
            continue
        seen_episode_ids.add(episode.episode_id)
        dataset_id = Path(episode.csv_source).parent.name
        local_csv = (local_data / dataset_id / "data.csv").resolve()
        if not local_csv.is_file():
            continue
        episode = episode.model_copy(update={"csv_source": str(local_csv)})
        grouped[dataset_id].append(episode)

    eligible = sorted(
        dataset_id
        for dataset_id, episodes in grouped.items()
        if len(episodes) >= minimum_examples
    )
    needed = sum(dataset_counts.values())
    if len(eligible) < needed:
        raise ValueError(
            f"need {needed} datasets with at least {minimum_examples} episodes; "
            f"found {len(eligible)}"
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)

    assignments: dict[str, list[str]] = {}
    cursor = 0
    for split in ("train", "validation", "test"):
        count = dataset_counts[split]
        assignments[split] = eligible[cursor : cursor + count]
        cursor += count

    output: dict[str, list[EpisodeJSONL]] = {}
    for split, dataset_ids in assignments.items():
        selected: list[EpisodeJSONL] = []
        for dataset_id in dataset_ids:
            candidates = sorted(grouped[dataset_id], key=lambda item: item.episode_id)
            split_rng = random.Random(f"{seed}:{split}:{dataset_id}")
            split_rng.shuffle(candidates)
            selected.extend(candidates[: episodes_per_dataset[split]])
        output[split] = selected

    manifest = {
        "seed": seed,
        "datasets": assignments,
        "episodes": {split: len(rows) for split, rows in output.items()},
    }
    return output, manifest


def prepare(args: argparse.Namespace) -> dict:
    from datasets import load_dataset

    with args.dataset_config.open("rb") as handle:
        config = tomllib.load(handle)
    dataset = load_dataset(config["repo"], revision=config["revision"], token=True)
    serialized = [row["episode_json"] for split in dataset.values() for row in split]
    splits, manifest = prepare_episode_splits(
        serialized,
        local_data=args.local_data,
        dataset_counts={
            "train": args.train_datasets,
            "validation": args.validation_datasets,
            "test": args.test_datasets,
        },
        episodes_per_dataset={
            "train": args.train_episodes_per_dataset,
            "validation": args.validation_episodes_per_dataset,
            "test": args.test_episodes_per_dataset,
        },
        seed=args.seed,
    )
    manifest.update(
        {
            "hugging_face_repo": config["repo"],
            "hugging_face_revision": config["revision"],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, episodes in splits.items():
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(episode.model_dump_json() + "\n" for episode in episodes)
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    manifest = prepare(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
