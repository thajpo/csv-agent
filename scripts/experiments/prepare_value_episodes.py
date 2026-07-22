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


EMPTY_PROCESS_REPORT = {
    "summary": {
        "total_steps": 0,
        "labeled_steps": 0,
        "verified_steps": 0,
        "heuristic_steps": 0,
        "unlabeled_steps": 0,
        "positive_steps": 0,
        "negative_steps": 0,
    },
    "steps": [],
}


def normalize_snapshot_episode(serialized: str) -> EpisodeJSONL:
    """Regenerate current diagnostic-only fields on the pinned old snapshot."""
    payload = json.loads(serialized)
    triangulation = dict(payload.get("triangulation", {}))
    triangulation.setdefault("float_tolerance", 0.1)
    payload["triangulation"] = triangulation
    payload.setdefault("process_report", EMPTY_PROCESS_REPORT)
    return EpisodeJSONL.model_validate(payload)


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
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        help="Dataset id to exclude; may be repeated.",
    )
    parser.add_argument(
        "--require-evaluation-templates-in-train",
        action="store_true",
        help="Select validation/test template names that are covered in train.",
    )
    return parser.parse_args()


def prepare_episode_splits(
    episode_json: Iterable[str],
    *,
    local_data: Path,
    dataset_counts: dict[str, int],
    episodes_per_dataset: dict[str, int],
    seed: int,
    excluded_datasets: set[str] | None = None,
    require_evaluation_templates_in_train: bool = False,
) -> tuple[dict[str, list[EpisodeJSONL]], dict]:
    excluded_datasets = excluded_datasets or set()
    grouped: dict[str, list[EpisodeJSONL]] = defaultdict(list)
    seen_episode_ids: set[str] = set()
    for serialized in episode_json:
        episode = normalize_snapshot_episode(serialized)
        if episode.episode_id in seen_episode_ids:
            continue
        seen_episode_ids.add(episode.episode_id)
        dataset_id = Path(episode.csv_source).parent.name
        if dataset_id in excluded_datasets:
            continue
        local_csv = (local_data / dataset_id / "data.csv").resolve()
        if not local_csv.is_file():
            continue
        episode = episode.model_copy(update={"csv_source": str(local_csv)})
        grouped[dataset_id].append(episode)

    needed = sum(dataset_counts.values())
    active_splits = [split for split, count in dataset_counts.items() if count]
    minimum_examples = (
        min(episodes_per_dataset[split] for split in active_splits)
        if active_splits
        else 0
    )
    eligible = sorted(
        dataset_id
        for dataset_id, episodes in grouped.items()
        if len(episodes) >= minimum_examples
    )
    if len(eligible) < needed:
        raise ValueError(
            f"need {needed} datasets with at least {minimum_examples} episodes; "
            f"found {len(eligible)}"
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)

    split_order = ("train", "validation", "test")
    assignments: dict[str, list[str]] = {split: [] for split in split_order}
    remaining = eligible
    allocation_order = sorted(
        split_order,
        key=lambda split: episodes_per_dataset[split],
        reverse=True,
    )
    for split in allocation_order:
        count = dataset_counts[split]
        required_episodes = episodes_per_dataset[split]
        candidates = [
            dataset_id
            for dataset_id in remaining
            if len(grouped[dataset_id]) >= required_episodes
        ]
        if len(candidates) < count:
            raise ValueError(
                f"need {count} {split} datasets with at least "
                f"{required_episodes} episodes; found {len(candidates)}"
            )
        assignments[split] = candidates[:count]
        selected = set(assignments[split])
        remaining = [
            dataset_id for dataset_id in remaining if dataset_id not in selected
        ]

    if require_evaluation_templates_in_train:
        output = _select_template_covered_episodes(
            grouped,
            assignments=assignments,
            episodes_per_dataset=episodes_per_dataset,
            seed=seed,
        )
    else:
        output = {}
        for split in split_order:
            dataset_ids = assignments[split]
            selected: list[EpisodeJSONL] = []
            for dataset_id in dataset_ids:
                candidates = sorted(
                    grouped[dataset_id], key=lambda item: item.episode_id
                )
                split_rng = random.Random(f"{seed}:{split}:{dataset_id}")
                split_rng.shuffle(candidates)
                selected.extend(candidates[: episodes_per_dataset[split]])
            output[split] = selected

    manifest = {
        "seed": seed,
        "excluded_datasets": sorted(excluded_datasets),
        "datasets": assignments,
        "episodes": {split: len(rows) for split, rows in output.items()},
        "templates": {
            split: sorted(
                {
                    episode.question.get("template_name")
                    for episode in rows
                    if episode.question.get("template_name")
                }
            )
            for split, rows in output.items()
        },
    }
    return output, manifest


def _select_template_covered_episodes(
    grouped: dict[str, list[EpisodeJSONL]],
    *,
    assignments: dict[str, list[str]],
    episodes_per_dataset: dict[str, int],
    seed: int,
) -> dict[str, list[EpisodeJSONL]]:
    """Select eval tasks whose named templates are represented in training."""
    train_candidates = [
        episode
        for dataset_id in assignments["train"]
        for episode in grouped[dataset_id]
    ]
    train_templates = {
        episode.question.get("template_name")
        for episode in train_candidates
        if episode.question.get("template_name")
    }
    output: dict[str, list[EpisodeJSONL]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for split in ("validation", "test"):
        for dataset_id in assignments[split]:
            candidates = [
                episode
                for episode in grouped[dataset_id]
                if episode.question.get("template_name") in train_templates
            ]
            split_rng = random.Random(f"{seed}:{split}:{dataset_id}")
            split_rng.shuffle(candidates)
            count = episodes_per_dataset[split]
            if len(candidates) < count:
                raise ValueError(
                    f"{dataset_id} has only {len(candidates)} evaluation episodes "
                    "whose templates occur in train; "
                    f"need {count}"
                )
            output[split].extend(candidates[:count])

    required_templates = {
        episode.question.get("template_name")
        for split in ("validation", "test")
        for episode in output[split]
    }
    selected_by_dataset: dict[str, list[EpisodeJSONL]] = {
        dataset_id: [] for dataset_id in assignments["train"]
    }
    selected_ids: set[str] = set()
    for template_name in sorted(required_templates):
        candidates = [
            episode
            for episode in train_candidates
            if episode.question.get("template_name") == template_name
            and episode.episode_id not in selected_ids
            and len(selected_by_dataset[Path(episode.csv_source).parent.name])
            < episodes_per_dataset["train"]
        ]
        if not candidates:
            raise ValueError(f"cannot cover evaluation template {template_name!r}")
        candidates.sort(
            key=lambda episode: (
                len(selected_by_dataset[Path(episode.csv_source).parent.name]),
                episode.episode_id,
            )
        )
        chosen = candidates[0]
        dataset_id = Path(chosen.csv_source).parent.name
        selected_by_dataset[dataset_id].append(chosen)
        selected_ids.add(chosen.episode_id)

    for dataset_id in assignments["train"]:
        candidates = [
            episode
            for episode in grouped[dataset_id]
            if episode.episode_id not in selected_ids
        ]
        split_rng = random.Random(f"{seed}:train:{dataset_id}")
        split_rng.shuffle(candidates)
        needed = episodes_per_dataset["train"] - len(selected_by_dataset[dataset_id])
        if len(candidates) < needed:
            raise ValueError(
                f"{dataset_id} cannot fill {episodes_per_dataset['train']} "
                "training episodes after template coverage"
            )
        selected_by_dataset[dataset_id].extend(candidates[:needed])

    output["train"] = [
        episode
        for dataset_id in assignments["train"]
        for episode in selected_by_dataset[dataset_id]
    ]
    selected_train_templates = {
        episode.question.get("template_name") for episode in output["train"]
    }
    if not required_templates <= selected_train_templates:
        raise RuntimeError(
            "selected training episodes do not cover evaluation templates"
        )
    return output


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
        excluded_datasets=set(args.exclude_dataset),
        require_evaluation_templates_in_train=(
            args.require_evaluation_templates_in_train
        ),
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
