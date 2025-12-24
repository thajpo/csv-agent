#!/usr/bin/env python3
"""
SFT Training Pipeline Entrypoint.

Chains: episodes.jsonl → split → format → train

Usage:
    uv run python entrypoints/sft_train.py
    uv run python entrypoints/sft_train.py --episodes data/episodes/episodes.jsonl
    uv run python entrypoints/sft_train.py --skip-split  # If already split
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: str | None = None) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run full SFT pipeline: split → format → train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="data/episodes/episodes.jsonl",
        help="Path to episodes JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/episodes",
        help="Directory for split and formatted files"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="Training format provider"
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip splitting (use existing train.jsonl)"
    )
    parser.add_argument(
        "--skip-format",
        action="store_true",
        help="Skip formatting (use existing train_openai.jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    episodes_path = Path(args.episodes)
    output_dir = Path(args.output_dir)

    # Validate input
    if not episodes_path.exists():
        print(f"Error: Episodes file not found: {episodes_path}")
        sys.exit(1)

    # Step 1: Split episodes
    if not args.skip_split:
        split_cmd = [
            "uv", "run", "python", "-m", "src.training.split_episodes",
            "--input", str(episodes_path),
            "--output-dir", str(output_dir),
        ]
        if args.dry_run:
            print(f"Would run: {' '.join(split_cmd)}")
        else:
            if not run_cmd(split_cmd):
                print("Error: Split failed")
                sys.exit(1)

    train_jsonl = output_dir / "train.jsonl"
    if not train_jsonl.exists() and not args.dry_run:
        print(f"Error: train.jsonl not found at {train_jsonl}")
        sys.exit(1)

    # Step 2: Format for provider
    formatted_file = output_dir / f"train_{args.provider}.jsonl"
    if not args.skip_format:
        format_cmd = [
            "uv", "run", "python", "-m", "src.training.prepare_finetune_data",
            "--input", str(train_jsonl),
            "--provider", args.provider,
            "--output", str(formatted_file),
        ]
        if args.dry_run:
            print(f"Would run: {' '.join(format_cmd)}")
        else:
            if not run_cmd(format_cmd):
                print("Error: Format failed")
                sys.exit(1)

    if not formatted_file.exists() and not args.dry_run:
        print(f"Error: Formatted file not found at {formatted_file}")
        sys.exit(1)

    # Step 3: Train
    training_dir = Path("training")
    train_cmd = [
        "uv", "run", "python", "train_sft.py",
        "--model", args.model,
        "--data", str(Path("..") / formatted_file),
        "--epochs", str(args.epochs),
    ]

    if args.dry_run:
        print(f"Would run (in {training_dir}): {' '.join(train_cmd)}")
    else:
        print(f"\nStarting training in {training_dir}...")
        if not run_cmd(train_cmd, cwd=str(training_dir)):
            print("Error: Training failed")
            sys.exit(1)

    print("\n" + "="*60)
    print("SFT Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
