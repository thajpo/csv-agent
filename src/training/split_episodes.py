"""
Train/Test/Val Splitting for CSV Agent Episodes.

This module provides functionality to load episodes from JSONL,
split them into stratified train/validation/test sets, and save
the splits back to JSONL files.

Key features:
- Stratification by difficulty (EASY/MEDIUM/HARD/VERY_HARD)
- Deterministic splitting with fixed seed
- Graceful handling of small datasets
- Verified episodes only by default
"""

import json
import logging
import warnings
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import random

from csv_spec import EpisodeJSONL
from src.core.config import config

logger = logging.getLogger(__name__)


def _field_value(episode: EpisodeJSONL, field: str) -> str:
    if field == "difficulty":
        return str(episode.question.get("difficulty", "UNKNOWN"))
    if field == "source":
        return str(episode.source or episode.question.get("source") or "UNKNOWN")
    if field.startswith("question."):
        question_field = field.split(".", 1)[1]
        return str(episode.question.get(question_field) or "UNKNOWN")
    return str(getattr(episode, field, "UNKNOWN") or "UNKNOWN")


def _stratification_key(episode: EpisodeJSONL, stratify_by: str) -> str:
    fields = [field.strip() for field in stratify_by.split(",") if field.strip()]
    if not fields:
        raise ValueError("stratify_by must name at least one field")
    return "|".join(_field_value(episode, field) for field in fields)


def load_episodes(path: str, verified_only: bool = True) -> List[EpisodeJSONL]:
    """
    Load episodes from JSONL file.

    Args:
        path: Path to episodes JSONL file
        verified_only: If True, only load episodes where verified=True

    Returns:
        List of EpisodeJSONL objects

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If JSONL is malformed
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Episodes file not found: {path}")

    episodes = []
    with open(path_obj, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                episode = EpisodeJSONL(**data)

                # Filter by verification status
                if verified_only and not episode.verified:
                    logger.debug(f"Skipping unverified episode: {episode.episode_id}")
                    continue

                episodes.append(episode)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
            except Exception as e:
                raise ValueError(f"Failed to parse episode at line {line_num}: {e}")

    logger.info(f"Loaded {len(episodes)} episodes from {path}")
    if verified_only:
        logger.info("Filtered to verified episodes only")

    return episodes


def split_episodes(
    episodes: List[EpisodeJSONL],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_by: str = "difficulty",
    seed: int = 42,
) -> Tuple[List[EpisodeJSONL], List[EpisodeJSONL], List[EpisodeJSONL]]:
    """
    Split episodes into stratified train/validation/test sets.

    Stratification ensures balanced representation of difficulty levels
    across all splits. For small datasets (< 15 episodes), warns but
    still proceeds with splitting.

    Args:
        episodes: List of episodes to split
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        stratify_by: Field to stratify by (default: "difficulty")
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_episodes, val_episodes, test_episodes)

    Raises:
        ValueError: If ratios don't sum to 1.0 or episodes list is empty
    """
    # Validate inputs
    if not episodes:
        raise ValueError("Cannot split empty episode list")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    # Warn for small datasets
    if len(episodes) < 15:
        warnings.warn(
            f"Small dataset detected: {len(episodes)} episodes. "
            f"Splits may be very small (train: ~{int(len(episodes) * train_ratio)}, "
            f"val: ~{int(len(episodes) * val_ratio)}, "
            f"test: ~{int(len(episodes) * test_ratio)}). "
            f"Consider generating more episodes for robust evaluation.",
            UserWarning,
        )

    # Set random seed for reproducibility
    random.seed(seed)

    # Group episodes by the requested stratification key
    stratified_groups = defaultdict(list)
    for episode in episodes:
        key = _stratification_key(episode, stratify_by)
        stratified_groups[key].append(episode)

    # Log distribution
    stratified_counts = {k: len(v) for k, v in stratified_groups.items()}
    logger.info(f"Episode distribution by {stratify_by}: {stratified_counts}")

    # Split each difficulty group proportionally
    train_episodes = []
    val_episodes = []
    test_episodes = []

    for group_key, group_episodes in stratified_groups.items():
        # Shuffle episodes in this difficulty group
        shuffled = list(group_episodes)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(n_total * train_ratio))  # Ensure at least 1 if possible
        n_val = max(0, int(n_total * val_ratio))

        # Handle edge case: if group is tiny (1-2 episodes), put in train
        if n_total <= 2:
            train_episodes.extend(shuffled)
            logger.warning(
                f"Stratum '{group_key}' has only {n_total} episode(s), "
                f"placing all in training set"
            )
            continue

        # Split
        train_split = shuffled[:n_train]
        val_split = shuffled[n_train : n_train + n_val]
        test_split = shuffled[n_train + n_val :]

        train_episodes.extend(train_split)
        val_episodes.extend(val_split)
        test_episodes.extend(test_split)

        logger.debug(
            f"Stratum '{group_key}': {n_total} total -> "
            f"train={len(train_split)}, val={len(val_split)}, test={len(test_split)}"
        )

    # Final shuffle to mix difficulties within each split
    random.shuffle(train_episodes)
    random.shuffle(val_episodes)
    random.shuffle(test_episodes)

    logger.info(
        f"Split complete: train={len(train_episodes)}, "
        f"val={len(val_episodes)}, test={len(test_episodes)}"
    )

    return train_episodes, val_episodes, test_episodes


def save_split(episodes: List[EpisodeJSONL], output_path: str) -> None:
    """
    Save episodes to JSONL file.

    Args:
        episodes: List of episodes to save
        output_path: Path to output JSONL file
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_obj, "w") as f:
        for episode in episodes:
            json_str = json.dumps(episode.model_dump(), default=str)
            f.write(json_str + "\n")

    logger.info(f"Saved {len(episodes)} episodes to {output_path}")


def main():
    """
    CLI entry point for splitting episodes.

    Usage:
        python -m src.training.split_episodes
        python -m src.training.split_episodes --input episodes/episodes.jsonl
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Split episodes into train/val/test sets with stratification"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input episodes JSONL file (e.g., data/episodes/template.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="episodes",
        help="Output directory for splits (default: episodes/)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=config.train_ratio,
        help=f"Training set ratio (default: {config.train_ratio})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=config.val_ratio,
        help=f"Validation set ratio (default: {config.val_ratio})",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=config.test_ratio,
        help=f"Test set ratio (default: {config.test_ratio})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.split_seed,
        help=f"Random seed for reproducibility (default: {config.split_seed})",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="difficulty",
        help=(
            "Comma-separated episode fields for stratification "
            "(default: difficulty; e.g. difficulty,source,question.dataset)"
        ),
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include unverified episodes (default: verified only)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load episodes
    logger.info(f"Loading episodes from {args.input}")
    episodes = load_episodes(args.input, verified_only=not args.include_unverified)

    if not episodes:
        logger.error("No episodes to split!")
        return

    # Split episodes
    logger.info("Splitting episodes...")
    train, val, test = split_episodes(
        episodes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_by=args.stratify_by,
        seed=args.seed,
    )

    # Save splits
    output_dir = Path(args.output_dir)
    save_split(train, str(output_dir / "train.jsonl"))
    save_split(val, str(output_dir / "val.jsonl"))
    save_split(test, str(output_dir / "test.jsonl"))

    logger.info("Splitting complete!")
    logger.info(f"Train: {len(train)} episodes")
    logger.info(f"Val: {len(val)} episodes")
    logger.info(f"Test: {len(test)} episodes")


if __name__ == "__main__":
    main()
