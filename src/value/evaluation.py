"""Metrics and equal-call selection tests for trajectory values."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Sequence

import numpy as np

from src.value.dataset import ValueExample


def prediction_metrics(
    examples: Sequence[ValueExample], scores: Sequence[float]
) -> dict[str, float | int | None]:
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

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.episode_id].append(index)
    correct = 0.0
    pairs = 0
    selected_targets: list[float] = []
    for indexes in grouped.values():
        selected = max(
            indexes,
            key=lambda index: (probabilities[index], examples[index].prefix_id),
        )
        selected_targets.append(targets[selected])
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1 :]:
                if targets[left] == targets[right]:
                    continue
                pairs += 1
                score_delta = probabilities[left] - probabilities[right]
                target_delta = targets[left] - targets[right]
                if score_delta == 0:
                    correct += 0.5
                elif score_delta * target_delta > 0:
                    correct += 1.0

    return {
        "examples": len(examples),
        "episodes": len(grouped),
        "log_loss": float(log_loss),
        "brier": float(brier),
        "calibration_error_5_bin": float(calibration_error),
        "ranking_pairs": pairs,
        "pairwise_ranking_accuracy": float(correct / pairs) if pairs else None,
        "selected_mean_label": float(np.mean(selected_targets)),
    }


def base_rate_scores(
    train: Sequence[ValueExample], examples: Sequence[ValueExample]
) -> np.ndarray:
    successes = sum(example.successes for example in train)
    attempts = sum(example.attempts for example in train)
    return np.full(len(examples), successes / attempts, dtype=float)


def selection_metrics(
    examples: Sequence[ValueExample], scores: Sequence[float], *, seed: int
) -> dict:
    """Compare local-score and seeded-random choices using held-out outcomes."""
    if len(examples) != len(scores):
        raise ValueError("examples and scores must align")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        if example.selection_verdict is None:
            raise ValueError("selection comparison requires one held-out verdict")
        grouped[example.episode_id].append(index)

    rows = []
    for episode_id, indexes in sorted(grouped.items()):
        if len(indexes) < 2:
            continue
        guided = max(
            indexes, key=lambda index: (scores[index], examples[index].prefix_id)
        )
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()[:8], "big"
        )
        random_choice = random.Random(stable_seed).choice(indexes)
        rows.append(
            {
                "episode_id": episode_id,
                "candidate_count": len(indexes),
                "guided_prefix_id": examples[guided].prefix_id,
                "random_prefix_id": examples[random_choice].prefix_id,
                "guided_success": bool(examples[guided].selection_verdict),
                "random_success": bool(examples[random_choice].selection_verdict),
                "oracle_success": any(
                    bool(examples[index].selection_verdict) for index in indexes
                ),
            }
        )
    if not rows:
        raise ValueError("selection comparison has no multi-candidate episodes")

    guided = np.asarray([row["guided_success"] for row in rows], dtype=float)
    random_outcomes = np.asarray([row["random_success"] for row in rows], dtype=float)
    oracle = np.asarray([row["oracle_success"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(10_000):
        indexes = rng.integers(0, len(rows), size=len(rows))
        bootstrap.append(float(np.mean(guided[indexes] - random_outcomes[indexes])))
    return {
        "episodes": len(rows),
        "candidate_actions_per_episode": sorted(
            {row["candidate_count"] for row in rows}
        ),
        "deployment_model_calls_per_episode": {
            "random": "candidate_count + 1 continuation",
            "value_guided": "candidate_count + 1 continuation",
        },
        "random_accuracy": float(random_outcomes.mean()),
        "value_guided_accuracy": float(guided.mean()),
        "oracle_candidate_accuracy": float(oracle.mean()),
        "guided_minus_random": float((guided - random_outcomes).mean()),
        "guided_minus_random_bootstrap_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "per_episode": rows,
    }
