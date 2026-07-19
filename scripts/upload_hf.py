#!/usr/bin/env python3
"""
Upload episodes to HuggingFace Hub.

Usage:
    uv run python scripts/upload_hf.py --repo your-username/csv-agent-episodes
    uv run python scripts/upload_hf.py --repo your-username/csv-agent-episodes --splits data/fixtures/splits_40
    uv run python scripts/upload_hf.py --repo your-username/csv-agent-episodes --private
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_splits(splits_dir: Path) -> DatasetDict:
    """Load train/val/test splits from a directory."""
    splits = {}

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.jsonl"
        if split_path.exists():
            episodes = load_jsonl(split_path)
            if episodes:
                splits[split_name] = Dataset.from_list(episodes)
                print(f"  {split_name}: {len(episodes)} episodes")

    if not splits:
        raise ValueError(f"No valid splits found in {splits_dir}")

    return DatasetDict(splits)


def main():
    parser = argparse.ArgumentParser(description="Upload episodes to HuggingFace Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo ID (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/episodes/splits"),
        help="Directory containing train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private",
    )
    args = parser.parse_args()

    print(f"Loading splits from {args.splits}...")
    dataset = load_splits(args.splits)

    print(f"\nUploading to {args.repo}...")
    dataset.push_to_hub(
        args.repo,
        private=args.private,
    )

    print(f"\n✓ Done! View at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
