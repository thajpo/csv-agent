"""Metrics for trajectory values and equal-call candidate selection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from src.value.dataset import ValueExample


def _ranking_summary(
    examples: Sequence[ValueExample], scores: Sequence[float]
) -> dict[str, float | int | None]:
    if len(examples) != len(scores) or not examples:
        raise ValueError("examples and scores must be nonempty and aligned")
    values = np.asarray(scores, dtype=float)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.episode_id].append(index)

    correct = 0.0
    pairs = 0
    selected_targets: list[float] = []
    varying_episodes = 0
    for indexes in grouped.values():
        selected = max(
            indexes, key=lambda index: (values[index], examples[index].prefix_id)
        )
        selected_targets.append(examples[selected].target)
        if len({examples[index].target for index in indexes}) > 1:
            varying_episodes += 1
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1 :]:
                target_delta = examples[left].target - examples[right].target
                if target_delta == 0:
                    continue
                pairs += 1
                score_delta = values[left] - values[right]
                if score_delta == 0:
                    correct += 0.5
                elif score_delta * target_delta > 0:
                    correct += 1.0

    return {
        "ranking_pairs": pairs,
        "pairwise_ranking_accuracy": float(correct / pairs) if pairs else None,
        "selected_mean_label": float(np.mean(selected_targets)),
        "episodes_with_label_variation": varying_episodes,
    }


def ranking_metrics(
    examples: Sequence[ValueExample], scores: Sequence[float]
) -> dict[str, float | int | None]:
    """Measure only within-question ordering, without probability semantics."""
    return {
        "examples": len(examples),
        "episodes": len({example.episode_id for example in examples}),
        **_ranking_summary(examples, scores),
    }


def prediction_metrics(
    examples: Sequence[ValueExample], scores: Sequence[float]
) -> dict[str, float | int | None]:
    """Measure pointwise success probabilities and their secondary ranking."""
    if len(examples) != len(scores) or not examples:
        raise ValueError("examples and scores must be nonempty and aligned")
    probabilities = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    targets = np.asarray([example.target for example in examples], dtype=float)
    weights = np.asarray([example.attempts for example in examples], dtype=float)
    log_loss = -np.average(
        targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities),
        weights=weights,
    )
    brier = np.average(
        targets * (1 - probabilities) ** 2 + (1 - targets) * probabilities**2,
        weights=weights,
    )

    calibration_error = 0.0
    total_weight = weights.sum()
    for lower in np.linspace(0.0, 0.8, 5):
        upper = lower + 0.2
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if mask.any():
            bin_weight = weights[mask].sum()
            calibration_error += (
                bin_weight
                / total_weight
                * abs(
                    np.average(probabilities[mask], weights=weights[mask])
                    - np.average(targets[mask], weights=weights[mask])
                )
            )

    return {
        "examples": len(examples),
        "episodes": len({example.episode_id for example in examples}),
        "log_loss": float(log_loss),
        "brier": float(brier),
        "calibration_error_5_bin": float(calibration_error),
        **_ranking_summary(examples, probabilities),
    }


def base_rate_scores(
    train: Sequence[ValueExample], examples: Sequence[ValueExample]
) -> np.ndarray:
    successes = sum(example.successes for example in train)
    attempts = sum(example.attempts for example in train)
    return np.full(len(examples), successes / attempts, dtype=float)


def _hierarchical_bootstrap(
    rows_by_dataset: Mapping[str, Sequence[dict]],
    *,
    selector: str,
    seed: int,
    iterations: int = 10_000,
) -> list[float]:
    dataset_ids = sorted(rows_by_dataset)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled_datasets = rng.choice(dataset_ids, size=len(dataset_ids), replace=True)
        dataset_differences = []
        for dataset_id in sampled_datasets:
            rows = rows_by_dataset[str(dataset_id)]
            sampled_indexes = rng.integers(0, len(rows), size=len(rows))
            dataset_differences.append(
                float(
                    np.mean(
                        [
                            rows[index]["selectors"][selector]
                            - rows[index]["expected_random_success"]
                            for index in sampled_indexes
                        ]
                    )
                )
            )
        estimates.append(float(np.mean(dataset_differences)))
    return estimates


def selection_metrics(
    examples: Sequence[ValueExample],
    score_sets: Mapping[str, Sequence[float]],
    *,
    seed: int,
) -> dict:
    """Compare selectors against exact expected random held-out success."""
    if not examples:
        raise ValueError("selection comparison requires examples")
    if not score_sets:
        raise ValueError("selection comparison requires at least one selector")
    for name, scores in score_sets.items():
        if len(scores) != len(examples):
            raise ValueError(f"selector {name!r} does not align with examples")
    if any(not example.selection_verdicts for example in examples):
        raise ValueError("selection comparison requires held-out verdicts")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.episode_id].append(index)

    rows: list[dict] = []
    for episode_id, indexes in sorted(grouped.items()):
        if len(indexes) < 2:
            continue
        dataset_ids = {examples[index].dataset_id for index in indexes}
        if len(dataset_ids) != 1:
            raise ValueError(f"episode {episode_id} spans multiple datasets")
        held_out = {index: examples[index].held_out_target for index in indexes}
        empirical = max(
            indexes,
            key=lambda index: (examples[index].target, examples[index].prefix_id),
        )
        selected = {
            name: max(
                indexes,
                key=lambda index: (scores[index], examples[index].prefix_id),
            )
            for name, scores in score_sets.items()
        }
        rows.append(
            {
                "episode_id": episode_id,
                "dataset_id": dataset_ids.pop(),
                "candidate_count": len(indexes),
                "held_out_continuations_per_candidate": len(
                    examples[indexes[0]].selection_verdicts
                ),
                "expected_random_success": float(
                    np.mean([held_out[index] for index in indexes])
                ),
                "empirical_rollout_success": held_out[empirical],
                "hindsight_realized_outcome_ceiling": max(held_out.values()),
                "empirical_prefix_id": examples[empirical].prefix_id,
                "selectors": {
                    name: held_out[index] for name, index in selected.items()
                },
                "selected_prefix_ids": {
                    name: examples[index].prefix_id for name, index in selected.items()
                },
            }
        )
    if not rows:
        raise ValueError("selection comparison has no multi-candidate episodes")

    rows_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[row["dataset_id"]].append(row)

    expected_random_episode = float(
        np.mean([row["expected_random_success"] for row in rows])
    )
    expected_random_dataset = float(
        np.mean(
            [
                np.mean([row["expected_random_success"] for row in dataset_rows])
                for dataset_rows in rows_by_dataset.values()
            ]
        )
    )
    selectors = {}
    for name in score_sets:
        per_dataset = {}
        for dataset_id, dataset_rows in sorted(rows_by_dataset.items()):
            guided = float(np.mean([row["selectors"][name] for row in dataset_rows]))
            expected = float(
                np.mean([row["expected_random_success"] for row in dataset_rows])
            )
            per_dataset[dataset_id] = {
                "episodes": len(dataset_rows),
                "guided_accuracy": guided,
                "expected_random_accuracy": expected,
                "difference": guided - expected,
            }
        dataset_macro_guided = float(
            np.mean([item["guided_accuracy"] for item in per_dataset.values()])
        )
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big"
        )
        bootstrap = _hierarchical_bootstrap(
            rows_by_dataset, selector=name, seed=stable_seed
        )
        selectors[name] = {
            "episode_weighted_accuracy": float(
                np.mean([row["selectors"][name] for row in rows])
            ),
            "dataset_macro_accuracy": dataset_macro_guided,
            "dataset_macro_difference_from_expected_random": (
                dataset_macro_guided - expected_random_dataset
            ),
            "hierarchical_bootstrap_95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "positive_datasets": sum(
                item["difference"] > 0 for item in per_dataset.values()
            ),
            "per_dataset": per_dataset,
        }

    empirical_dataset_macro = float(
        np.mean(
            [
                np.mean([row["empirical_rollout_success"] for row in dataset_rows])
                for dataset_rows in rows_by_dataset.values()
            ]
        )
    )
    hindsight_dataset_macro = float(
        np.mean(
            [
                np.mean(
                    [row["hindsight_realized_outcome_ceiling"] for row in dataset_rows]
                )
                for dataset_rows in rows_by_dataset.values()
            ]
        )
    )
    return {
        "episodes": len(rows),
        "datasets": len(rows_by_dataset),
        "candidate_actions_per_episode": sorted(
            {row["candidate_count"] for row in rows}
        ),
        "held_out_continuations_per_candidate": sorted(
            {row["held_out_continuations_per_candidate"] for row in rows}
        ),
        "deployment_model_calls_per_episode": {
            "expected_random": "candidate_count + 1 continuation",
            "learned_selector": "candidate_count + 1 continuation",
        },
        "measurement_note": (
            "Repeated held-out continuations estimate each deployment policy's "
            "one-continuation success probability; they are measurement calls, "
            "not extra deployed selector calls."
        ),
        "expected_random_episode_weighted_accuracy": expected_random_episode,
        "expected_random_dataset_macro_accuracy": expected_random_dataset,
        "empirical_rollout_dataset_macro_accuracy": empirical_dataset_macro,
        "hindsight_realized_outcome_dataset_macro_ceiling": (hindsight_dataset_macro),
        "selectors": selectors,
        "per_episode": rows,
    }
