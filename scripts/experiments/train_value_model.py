"""Train the minimal local value model and report held-out metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.value.dataset import assert_dataset_disjoint, load_value_examples
from src.value.evaluation import base_rate_scores, prediction_metrics
from src.value.trainer import SimpleSignalModel, TrainedValueModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a local text model to predict continuation success."
    )
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-continuations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=30_000)
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict:
    train_examples = load_value_examples(
        args.train, holdout_continuations=args.holdout_continuations
    )
    validation_examples = load_value_examples(
        args.validation, holdout_continuations=args.holdout_continuations
    )
    test_examples = load_value_examples(
        args.test, holdout_continuations=args.holdout_continuations
    )
    assert_dataset_disjoint(train_examples, validation_examples, test_examples)

    text_model = TrainedValueModel.fit(
        train_examples, seed=args.seed, max_features=args.max_features
    )
    simple_model = SimpleSignalModel.fit(train_examples, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "value_model.joblib"
    text_model.save(checkpoint)

    results = {
        "seed": args.seed,
        "holdout_continuations": args.holdout_continuations,
        "model": {
            "type": "tfidf_logistic_regression",
            "max_features": args.max_features,
            "checkpoint": str(checkpoint),
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
                ("test", test_examples),
            )
        },
        "validation": {
            "base_rate": prediction_metrics(
                validation_examples,
                base_rate_scores(train_examples, validation_examples),
            ),
            "simple_signals": prediction_metrics(
                validation_examples, simple_model.predict(validation_examples)
            ),
            "text_model": prediction_metrics(
                validation_examples, text_model.predict(validation_examples)
            ),
        },
        "test": {
            "base_rate": prediction_metrics(
                test_examples, base_rate_scores(train_examples, test_examples)
            ),
            "simple_signals": prediction_metrics(
                test_examples, simple_model.predict(test_examples)
            ),
            "text_model": prediction_metrics(
                test_examples, text_model.predict(test_examples)
            ),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    return results


def main() -> None:
    results = train(parse_args())
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
