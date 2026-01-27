#!/usr/bin/env python3
"""
Push consolidated episodes to HuggingFace Hub.

Usage:
    uv run python scripts/push_to_hf.py
    uv run python scripts/push_to_hf.py --repo-id your-username/dataset-name
    uv run python scripts/push_to_hf.py --dry-run  # preview without uploading
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


DATA_DIR = Path(__file__).parent.parent / "data" / "questions_synthetic"
DEFAULT_REPO = "thajpo/csv-agent-synthetic-episodes"


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, return empty list if doesn't exist."""
    if not path.exists():
        return []
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: skipping malformed line in {path.name}: {e}")
    return episodes


def collect_episodes(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Collect all episodes from all dataset directories."""
    successful = []
    failed = []

    for dataset_dir in sorted(data_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # Load successful episodes
        eps = load_jsonl(dataset_dir / "episodes.jsonl")
        for ep in eps:
            ep["dataset"] = dataset_name  # Add dataset provenance
        successful.extend(eps)

        # Load failed episodes (for DPO)
        failed_eps = load_jsonl(dataset_dir / "episodes_failed.jsonl")
        for ep in failed_eps:
            ep["dataset"] = dataset_name
        failed.extend(failed_eps)

        if eps or failed_eps:
            print(f"  {dataset_name}: {len(eps)} ✓, {len(failed_eps)} ✗")

    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Push episodes to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without uploading",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make dataset private (default: True)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make dataset public",
    )
    args = parser.parse_args()

    private = not args.public

    print(f"Collecting episodes from: {DATA_DIR}")
    print()

    successful, failed = collect_episodes(DATA_DIR)

    print()
    print(f"Total: {len(successful)} successful, {len(failed)} failed")

    if not successful:
        print("No episodes found! Run synthetic generation first.")
        return 1

    # Compute stats
    datasets = set(ep.get("dataset", "unknown") for ep in successful)
    difficulties = {}
    for ep in successful:
        diff = ep.get("question", {}).get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1

    print()
    print(f"Datasets: {len(datasets)}")
    print(f"Difficulty distribution: {difficulties}")

    if args.dry_run:
        print()
        print("[dry-run] Would upload to:", args.repo_id)
        print("[dry-run] Private:", private)
        return 0

    # Create HF datasets
    print()
    print("Creating HuggingFace datasets...")

    ds_dict = DatasetDict({
        "train": Dataset.from_list(successful),
    })

    # Add failed split if we have any (for DPO)
    if failed:
        ds_dict["failed"] = Dataset.from_list(failed)

    print(f"  train: {len(ds_dict['train'])} episodes")
    if "failed" in ds_dict:
        print(f"  failed: {len(ds_dict['failed'])} episodes (for DPO)")

    # Push to hub
    print()
    print(f"Pushing to: {args.repo_id} (private={private})")
    ds_dict.push_to_hub(args.repo_id, private=private)

    print()
    print(f"✓ Uploaded to: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    exit(main())
