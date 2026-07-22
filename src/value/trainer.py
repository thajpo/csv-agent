"""Small CPU-friendly trainer for verifier-grounded trajectory values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Callable, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack

from src.value.dataset import ValueExample, simple_features


def _binomial_rows(
    examples: Sequence[ValueExample], feature: Callable[[ValueExample], object]
) -> tuple[list[object], list[int], list[float]]:
    rows: list[object] = []
    labels: list[int] = []
    weights: list[float] = []
    for example in examples:
        failures = example.attempts - example.successes
        if example.successes:
            rows.append(feature(example))
            labels.append(1)
            weights.append(float(example.successes))
        if failures:
            rows.append(feature(example))
            labels.append(0)
            weights.append(float(failures))
    return rows, labels, weights


@dataclass
class TrainedValueModel:
    """TF-IDF text model with a constant fallback for single-class data."""

    vectorizer: TfidfVectorizer | None
    classifier: LogisticRegression | None
    constant: float | None

    @classmethod
    def fit(
        cls,
        examples: Sequence[ValueExample],
        *,
        seed: int = 42,
        max_features: int = 30_000,
    ) -> "TrainedValueModel":
        if not examples:
            raise ValueError("at least one training example is required")
        rows, labels, weights = _binomial_rows(examples, lambda item: item.text)
        total_successes = sum(item.successes for item in examples)
        total_attempts = sum(item.attempts for item in examples)
        if len(set(labels)) < 2:
            return cls(None, None, total_successes / total_attempts)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            max_features=max_features,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(rows)
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1_000,
            random_state=seed,
            solver="liblinear",
        )
        classifier.fit(matrix, labels, sample_weight=weights)
        return cls(vectorizer, classifier, None)

    def predict(self, examples: Sequence[ValueExample]) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(examples), self.constant, dtype=float)
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("value model is not fitted")
        matrix = self.vectorizer.transform([example.text for example in examples])
        return self.classifier.predict_proba(matrix)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "TrainedValueModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("checkpoint does not contain a TrainedValueModel")
        return model


@dataclass
class SimpleSignalModel:
    """Logistic baseline using only length and execution-error signals."""

    classifier: LogisticRegression | None
    constant: float | None

    @classmethod
    def fit(
        cls, examples: Sequence[ValueExample], *, seed: int = 42
    ) -> "SimpleSignalModel":
        rows, labels, weights = _binomial_rows(examples, simple_features)
        successes = sum(item.successes for item in examples)
        attempts = sum(item.attempts for item in examples)
        if len(set(labels)) < 2:
            return cls(None, successes / attempts)
        classifier = LogisticRegression(max_iter=1_000, random_state=seed)
        classifier.fit(np.asarray(rows), labels, sample_weight=weights)
        return cls(classifier, None)

    def predict(self, examples: Sequence[ValueExample]) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(examples), self.constant, dtype=float)
        if self.classifier is None:
            raise RuntimeError("simple-signal model is not fitted")
        return self.classifier.predict_proba(
            np.asarray([simple_features(example) for example in examples])
        )[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "SimpleSignalModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("checkpoint does not contain a SimpleSignalModel")
        return model


@dataclass
class PairwiseValueRanker:
    """Linear text ranker trained only on within-question differences."""

    vectorizer: TfidfVectorizer | None
    classifier: LogisticRegression | None
    training_pairs: int

    @classmethod
    def fit(
        cls,
        examples: Sequence[ValueExample],
        *,
        seed: int = 42,
        max_features: int = 30_000,
    ) -> "PairwiseValueRanker":
        if not examples:
            raise ValueError("at least one training example is required")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            max_features=max_features,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([example.text for example in examples])
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            grouped[example.episode_id].append(index)

        differences = []
        labels: list[int] = []
        weights: list[float] = []
        pair_count = 0
        for indexes in grouped.values():
            for offset, left in enumerate(indexes):
                for right in indexes[offset + 1 :]:
                    target_delta = examples[left].target - examples[right].target
                    if target_delta == 0:
                        continue
                    preferred, rejected = (
                        (left, right) if target_delta > 0 else (right, left)
                    )
                    difference = matrix[preferred] - matrix[rejected]
                    weight = abs(target_delta)
                    differences.extend((difference, -difference))
                    labels.extend((1, 0))
                    weights.extend((weight, weight))
                    pair_count += 1

        if not differences:
            return cls(vectorizer, None, 0)

        classifier = LogisticRegression(
            C=1.0,
            fit_intercept=False,
            max_iter=1_000,
            random_state=seed,
            solver="liblinear",
        )
        classifier.fit(vstack(differences), labels, sample_weight=weights)
        return cls(vectorizer, classifier, pair_count)

    def predict(self, examples: Sequence[ValueExample]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("pairwise ranker is not fitted")
        if self.classifier is None:
            return np.zeros(len(examples), dtype=float)
        matrix = self.vectorizer.transform([example.text for example in examples])
        return np.asarray(self.classifier.decision_function(matrix), dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "PairwiseValueRanker":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("checkpoint does not contain a PairwiseValueRanker")
        return model
