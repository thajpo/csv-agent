"""Evaluate frozen value selectors once on sealed held-out records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.value.dataset import load_value_examples
from src.value.evaluation import prediction_metrics, ranking_metrics, selection_metrics
from src.value.trainer import (
    PairwiseValueRanker,
    SimpleSignalModel,
    TrainedValueModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen selectors on sealed candidate outcomes."
    )
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--holdout-continuations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_models(model_dir: Path) -> tuple[dict, dict]:
    freeze_path = model_dir / "model-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    expected_names = {
        "pairwise_ranker",
        "pointwise_value_model",
        "simple_signal_model",
    }
    if set(freeze.get("checkpoints", {})) != expected_names:
        raise ValueError("model freeze does not declare the expected checkpoints")
    paths = {}
    for name in sorted(expected_names):
        metadata = freeze["checkpoints"][name]
        path = model_dir / metadata["path"]
        if _sha256(path) != metadata["sha256"]:
            raise ValueError(f"frozen checkpoint hash changed: {name}")
        paths[name] = path
    models = {
        "pairwise_ranker": PairwiseValueRanker.load(paths["pairwise_ranker"]),
        "pointwise_value_model": TrainedValueModel.load(paths["pointwise_value_model"]),
        "simple_signal_model": SimpleSignalModel.load(paths["simple_signal_model"]),
    }
    return freeze, models


def evaluate(args: argparse.Namespace) -> dict:
    freeze, models = _load_frozen_models(args.model_dir)
    if freeze["holdout_continuations"] != args.holdout_continuations:
        raise ValueError("evaluation holdout count differs from the frozen model")
    if freeze["seed"] != args.seed:
        raise ValueError("evaluation seed differs from the frozen model")

    examples = load_value_examples(
        args.test, holdout_continuations=args.holdout_continuations
    )
    score_sets = {name: model.predict(examples) for name, model in models.items()}
    selection = selection_metrics(examples, score_sets, seed=args.seed)
    primary = selection["selectors"][freeze["primary_selector"]]
    interval = primary["hierarchical_bootstrap_95"]
    decision_checks = {
        "effect_at_least_five_points": (
            primary["dataset_macro_difference_from_expected_random"] >= 0.05
        ),
        "bootstrap_interval_excludes_zero": interval[0] > 0,
        "positive_on_at_least_three_datasets": primary["positive_datasets"] >= 3,
    }
    results = {
        "test_records": {"path": str(args.test), "sha256": _sha256(args.test)},
        "model_freeze": freeze,
        "selection": selection,
        "test_label_metrics": {
            "pairwise_ranker": ranking_metrics(examples, score_sets["pairwise_ranker"]),
            "pointwise_value_model": prediction_metrics(
                examples, score_sets["pointwise_value_model"]
            ),
            "simple_signal_model": prediction_metrics(
                examples, score_sets["simple_signal_model"]
            ),
        },
        "preregistered_decision": {
            "checks": decision_checks,
            "improvement_demonstrated": all(decision_checks.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def main() -> None:
    results = evaluate(parse_args())
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
