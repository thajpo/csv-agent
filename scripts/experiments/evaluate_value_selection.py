"""Run the phase-4 equal-call action-selection comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.value.dataset import load_value_examples
from src.value.evaluation import selection_metrics
from src.value.trainer import TrainedValueModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare random and value-guided partial-attempt selection."
    )
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holdout-continuations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    examples = load_value_examples(
        args.test, holdout_continuations=args.holdout_continuations
    )
    model = TrainedValueModel.load(args.checkpoint)
    results = selection_metrics(examples, model.predict(examples), seed=args.seed)
    results["checkpoint"] = str(args.checkpoint)
    results["test_records"] = str(args.test)
    results["holdout_continuations"] = args.holdout_continuations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def main() -> None:
    results = evaluate(parse_args())
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
