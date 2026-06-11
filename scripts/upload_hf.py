#!/usr/bin/env python3
"""Upload csv-agent episode splits to Hugging Face Hub.

Usage:
    uv run python scripts/upload_hf.py --repo-id username/csv-agent-episodes --dry-run
    uv run python scripts/upload_hf.py --repo-id username/csv-agent-episodes --include-csvs
    uv run python scripts/upload_hf.py --repo-id username/csv-agent-episodes --input data/episodes/template.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

from src.training.split_episodes import load_episodes, save_split, split_episodes

REQUIRED_SPLITS = ("train", "val", "test")


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def episode_to_hf_row(episode: dict) -> dict:
    """Convert a canonical episode to a stable HF row."""
    question = episode.get("question") or {}
    triangulation = episode.get("triangulation") or {}
    return {
        "episode_id": episode.get("episode_id", ""),
        "source": episode.get("source") or "",
        "csv_source": episode.get("csv_source") or "",
        "verified": bool(episode.get("verified", False)),
        "question_id": question.get("id") or "",
        "question_dataset": question.get("dataset") or "",
        "difficulty": question.get("difficulty") or "UNKNOWN",
        "template_name": question.get("template_name") or "",
        "n_steps": int(question.get("n_steps") or 0),
        "majority_count": int(triangulation.get("majority_count") or 0),
        "episode_json": json.dumps(episode, default=str),
    }


def split_if_needed(
    *,
    input_path: Path | None,
    splits_dir: Path,
    include_unverified: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by: str,
) -> None:
    """Create split JSONL files when --input is provided."""
    if input_path is None:
        return

    episodes = load_episodes(str(input_path), verified_only=not include_unverified)
    train, val, test = split_episodes(
        episodes,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by=stratify_by,
        seed=seed,
    )
    save_split(train, str(splits_dir / "train.jsonl"))
    save_split(val, str(splits_dir / "val.jsonl"))
    save_split(test, str(splits_dir / "test.jsonl"))


def load_splits(
    splits_dir: Path, required_splits: tuple[str, ...] = REQUIRED_SPLITS
) -> DatasetDict:
    """Load train/val/test splits from a directory."""
    splits = {}

    missing = [
        name
        for name in required_splits
        if not (splits_dir / f"{name}.jsonl").exists()
    ]
    if missing:
        raise ValueError(f"Missing required split file(s): {', '.join(missing)}")

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.jsonl"
        if split_path.exists():
            episodes = load_jsonl(split_path)
            if split_name in required_splits and not episodes:
                raise ValueError(f"Required split is empty: {split_name}")
            if episodes:
                splits[split_name] = Dataset.from_list(
                    [episode_to_hf_row(episode) for episode in episodes]
                )
                print(f"  {split_name}: {len(episodes)} episodes")

    if not splits:
        raise ValueError(f"No valid splits found in {splits_dir}")

    return DatasetDict(splits)


def dataset_stats(dataset: DatasetDict) -> dict:
    stats: dict = {
        "splits": {},
        "total": 0,
        "difficulties": {},
        "sources": {},
        "question_datasets": {},
        "csv_sources": set(),
    }
    for split_name, split_dataset in dataset.items():
        split_count = len(split_dataset)
        stats["splits"][split_name] = split_count
        stats["total"] += split_count
        for row in split_dataset:
            difficulty = row.get("difficulty") or "UNKNOWN"
            stats["difficulties"][difficulty] = (
                stats["difficulties"].get(difficulty, 0) + 1
            )
            source = row.get("source") or "UNKNOWN"
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            question_dataset = row.get("question_dataset") or "UNKNOWN"
            stats["question_datasets"][question_dataset] = (
                stats["question_datasets"].get(question_dataset, 0) + 1
            )
            if row.get("csv_source"):
                stats["csv_sources"].add(row["csv_source"])
    stats["csv_sources"] = len(stats["csv_sources"])
    return stats


def _csv_candidates(csv_source: str, csv_root: Path) -> list[Path]:
    path = Path(csv_source)
    candidates = [path]
    parts = path.parts
    root_parts = csv_root.parts
    if len(parts) >= len(root_parts) and parts[: len(root_parts)] == root_parts:
        candidates.append(csv_root / Path(*parts[len(root_parts) :]))
    if "kaggle" in parts:
        kaggle_idx = parts.index("kaggle")
        rel_after_kaggle = Path(*parts[kaggle_idx + 1 :])
        candidates.append(csv_root / rel_after_kaggle)
    return candidates


def validate_csv_references(dataset: DatasetDict, csv_root: Path) -> None:
    """Fail fast if any uploaded episode references a CSV outside the upload tree."""
    missing = []
    refs = {
        row.get("csv_source")
        for split_dataset in dataset.values()
        for row in split_dataset
        if row.get("csv_source")
    }
    for ref in sorted(refs):
        candidates = _csv_candidates(ref, csv_root)
        if not any(candidate.exists() for candidate in candidates):
            missing.append(ref)
    if missing:
        sample = "\n".join(f"  - {ref}" for ref in missing[:10])
        extra = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} episode CSV reference(s) are missing from {csv_root}:\n"
            f"{sample}{extra}"
        )


def data_card(repo_id: str, stats: dict, include_csvs: bool) -> str:
    return f"""---
license: other
task_categories:
- text-generation
tags:
- csv-agent
- reinforcement-learning
- verifiers
- prime-rl
pretty_name: csv-agent Episodes
---

# csv-agent Episodes

Canonical `EpisodeJSONL` records generated by csv-agent. The dataset is intended
for deriving SFT, PRM, and RL environments at training time.

Each row stores scalar metadata plus `episode_json`, a JSON-encoded canonical
episode. This avoids Arrow schema instability from heterogeneous nested trace
objects while preserving the original episode contract.

## Splits

{json.dumps(stats["splits"], indent=2)}

Total episodes: `{stats["total"]}`

Difficulty distribution:

{json.dumps(stats["difficulties"], indent=2)}

Source distribution:

{json.dumps(stats.get("sources", {}), indent=2)}

Distinct CSV sources: `{stats["csv_sources"]}`

Raw CSV files included in this repo: `{include_csvs}`

## Prime-RL / Verifiers

Install csv-agent in the training environment, then load:

```python
import verifiers as vf

env = vf.load_environment(
    "csv-agent",
    dataset_name="{repo_id}",
    dataset_split="train",
)
```

If raw CSV files are included, the adapter snapshots `data/kaggle/**` from this
dataset repo and rebases episode `csv_source` paths automatically.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Upload csv-agent episodes to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        "--repo",
        dest="repo_id",
        default="ThaJpo/csv-agent-template-episodes",
        help="HuggingFace repo ID (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits/template"),
        help="Directory containing train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional source episodes JSONL to split before upload",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-by",
        default="difficulty,source,question.dataset",
        help=(
            "Comma-separated episode fields used when splitting --input "
            "(default: difficulty,source,question.dataset)"
        ),
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include unverified episodes when splitting --input",
    )
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=Path("data/kaggle"),
        help="Raw CSV root to upload when --include-csvs is set",
    )
    parser.add_argument(
        "--include-csvs",
        action="store_true",
        help="Upload raw CSV files under data/kaggle for self-contained RL runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print stats without uploading",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Create/update a private dataset repo (default)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create/update a public dataset repo",
    )
    args = parser.parse_args()
    private = not args.public

    split_if_needed(
        input_path=args.input,
        splits_dir=args.splits,
        include_unverified=args.include_unverified,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by=args.stratify_by,
    )
    print(f"Loading splits from {args.splits}...")
    dataset = load_splits(args.splits)
    stats = dataset_stats(dataset)
    print("\nStats:")
    print(json.dumps(stats, indent=2, default=str))

    if args.include_csvs:
        if not args.csv_root.exists():
            raise FileNotFoundError(f"CSV root not found: {args.csv_root}")
        validate_csv_references(dataset, args.csv_root)
        csv_file_count = sum(1 for p in args.csv_root.rglob("*") if p.is_file())
        print(f"CSV root: {args.csv_root} ({csv_file_count} files)")

    token = HfFolder.get_token()
    print(f"HF token: {'available' if token else 'missing'}")
    if args.dry_run:
        print(f"\n[dry-run] Would upload to {args.repo_id} (private={private})")
        print(f"[dry-run] Would include CSVs: {args.include_csvs}")
        return 0

    if not token:
        raise RuntimeError("No Hugging Face token found. Run `huggingface-cli login`.")

    api = HfApi()
    whoami = api.whoami(token=token)
    print(f"HF account: {whoami.get('name', '<unknown>')}")

    print(f"\nCreating/updating dataset repo {args.repo_id} (private={private})...")
    api.create_repo(args.repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

    print("Uploading episode splits...")
    dataset.push_to_hub(args.repo_id, private=private, token=token)

    if args.include_csvs:
        print("Uploading raw CSV tree...")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=args.csv_root,
            path_in_repo="data/kaggle",
            token=token,
            commit_message="Upload csv-agent raw CSV files",
        )

    print("Uploading dataset card...")
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=data_card(args.repo_id, stats, args.include_csvs).encode("utf-8"),
        path_in_repo="README.md",
        token=token,
        commit_message="Update dataset card",
    )

    print(f"\nDone. View at: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
