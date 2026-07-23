"""Train frozen value selectors using train and validation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from src.value.dataset import assert_dataset_disjoint, load_value_examples
from src.value.evaluation import base_rate_scores, prediction_metrics, ranking_metrics
from src.value.trainer import (
    PairwiseValueRanker,
    SimpleSignalModel,
    TrainedValueModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train value selectors without opening held-out test records."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-continuations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-features", type=int, default=30_000)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def train(args: argparse.Namespace) -> dict:
    train_examples = load_value_examples(
        args.train, holdout_continuations=args.holdout_continuations
    )
    validation_examples = load_value_examples(
        args.validation, holdout_continuations=args.holdout_continuations
    )
    assert_dataset_disjoint(train_examples, validation_examples, [])

    pointwise_model = TrainedValueModel.fit(
        train_examples, seed=args.seed, max_features=args.max_features
    )
    simple_model = SimpleSignalModel.fit(train_examples, seed=args.seed)
    pairwise_ranker = PairwiseValueRanker.fit(
        train_examples, seed=args.seed, max_features=args.max_features
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        "pairwise_ranker": args.output_dir / "pairwise_ranker.joblib",
        "pointwise_value_model": args.output_dir / "pointwise_value_model.joblib",
        "simple_signal_model": args.output_dir / "simple_signal_model.joblib",
    }
    pairwise_ranker.save(checkpoint_paths["pairwise_ranker"])
    pointwise_model.save(checkpoint_paths["pointwise_value_model"])
    simple_model.save(checkpoint_paths["simple_signal_model"])

    pointwise_scores = pointwise_model.predict(validation_examples)
    simple_scores = simple_model.predict(validation_examples)
    pairwise_scores = pairwise_ranker.predict(validation_examples)
    results = {
        "seed": args.seed,
        "holdout_continuations": args.holdout_continuations,
        "primary_selector": "pairwise_ranker",
        "models": {
            "pairwise_ranker": {
                "type": "tfidf_pairwise_logistic_regression",
                "max_features": args.max_features,
                "training_pairs": pairwise_ranker.training_pairs,
            },
            "pointwise_value_model": {
                "type": "tfidf_logistic_regression",
                "max_features": args.max_features,
            },
            "simple_signal_model": {
                "type": "logistic_regression_over_execution_signals"
            },
        },
        "splits": {
            name: {
                "examples": len(examples),
                "episodes": len({item.episode_id for item in examples}),
                "datasets": sorted({item.dataset_id for item in examples}),
                "successes": sum(item.successes for item in examples),
                "attempts": sum(item.attempts for item in examples),
            }
            for name, examples in (
                ("train", train_examples),
                ("validation", validation_examples),
            )
        },
        "validation": {
            "base_rate": prediction_metrics(
                validation_examples,
                base_rate_scores(train_examples, validation_examples),
            ),
            "simple_signals": prediction_metrics(validation_examples, simple_scores),
            "pointwise_value_model": prediction_metrics(
                validation_examples, pointwise_scores
            ),
            "pairwise_ranker": ranking_metrics(validation_examples, pairwise_scores),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )

    freeze = {
        "code_commit": _current_commit(),
        "primary_selector": "pairwise_ranker",
        "seed": args.seed,
        "holdout_continuations": args.holdout_continuations,
        "max_features": args.max_features,
        "inputs": {
            "train": {"path": str(args.train), "sha256": _sha256(args.train)},
            "validation": {
                "path": str(args.validation),
                "sha256": _sha256(args.validation),
            },
        },
        "checkpoints": {
            name: {"path": path.name, "sha256": _sha256(path)}
            for name, path in checkpoint_paths.items()
        },
    }
    (args.output_dir / "model-freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    results["freeze"] = freeze
    return results


def main() -> None:
    results = train(parse_args())
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
