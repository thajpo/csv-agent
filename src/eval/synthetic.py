"""
Evaluate teacher performance on synthetic (template-based) questions.

Synthetic questions have known ground truth from template execution.
This evaluator compares teacher answers against that ground truth.
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SyntheticEvalResult:
    """Result of evaluating a single synthetic episode."""

    question_text: str
    template: str
    difficulty: str

    # Match results
    matched: bool
    mismatch_reason: str | None = None

    # Raw values
    expected: Any = None
    actual: Any = None


@dataclass
class SyntheticEvalMetrics:
    """Aggregate metrics for synthetic evaluation."""

    accuracy: float
    total: int
    correct: int

    # By difficulty
    accuracy_by_difficulty: dict[str, float] = field(default_factory=dict)
    total_by_difficulty: dict[str, int] = field(default_factory=dict)
    correct_by_difficulty: dict[str, int] = field(default_factory=dict)

    # By template
    accuracy_by_template: dict[str, float] = field(default_factory=dict)

    # Mismatches for debugging
    mismatches: list[SyntheticEvalResult] = field(default_factory=list)


def normalize_key(k: str) -> str:
    """Normalize key for comparison."""
    return k.lower().replace("-", "_").replace("‑", "_")


def normalize_value(v: Any) -> Any:
    """Normalize value for comparison."""
    if isinstance(v, str):
        return v.lower().replace("‑", "-")
    if isinstance(v, list):
        return [normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {normalize_key(k): normalize_value(val) for k, val in v.items()}
    return v


def values_match(expected: Any, actual: Any, tol: float = 0.01) -> bool:
    """Check if two values match semantically."""
    expected = normalize_value(expected)
    actual = normalize_value(actual)

    if type(expected) != type(actual):
        if isinstance(expected, bool) and isinstance(actual, str):
            return str(expected).lower() == actual.lower()
        if isinstance(actual, bool) and isinstance(expected, str):
            return str(actual).lower() == expected.lower()
        return False

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isnan(expected) or math.isnan(actual):
            return math.isnan(expected) and math.isnan(actual)
        if expected == 0:
            return abs(actual) < tol
        return abs(expected - actual) / abs(expected) < tol

    if isinstance(expected, str):
        return expected == actual

    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False
        return all(values_match(e, a, tol) for e, a in zip(expected, actual))

    if isinstance(expected, dict):
        for k, v in expected.items():
            if k not in actual:
                return False
            if not values_match(v, actual[k], tol):
                return False
        return True

    return expected == actual


def dicts_match_semantic(
    expected: dict, actual: dict, tol: float = 0.01
) -> tuple[bool, str]:
    """
    Check if teacher answer matches ground truth semantically.

    Returns (match, reason)
    """
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        if values_match(expected, actual, tol):
            return True, "exact match"
        return False, f"value mismatch: expected {expected}, got {actual}"

    expected_norm = {normalize_key(k): v for k, v in expected.items()}
    actual_norm = {normalize_key(k): v for k, v in actual.items()}

    for k, v in expected_norm.items():
        if k not in actual_norm:
            aliases = {
                "iqr": ["interquartile_range", "iqr_range"],
                "distribution": ["normal", "is_normal"],
                "std": ["standard_deviation", "sd"],
            }
            found = False
            for alias in aliases.get(k, []):
                if alias in actual_norm:
                    if values_match(v, actual_norm[alias], tol):
                        found = True
                        break
            if not found:
                return False, f"missing key: {k}"
        else:
            if not values_match(v, actual_norm[k], tol):
                return False, f"value mismatch for {k}: expected {v}, got {actual_norm[k]}"

    return True, "semantic match"


class SyntheticEvaluator:
    """Evaluates teacher performance on synthetic questions."""

    def __init__(self, episodes_path: str, tolerance: float = 0.01):
        """
        Initialize evaluator.

        Args:
            episodes_path: Path to episodes.jsonl
            tolerance: Relative tolerance for float comparison (default 1%)
        """
        self.episodes_path = Path(episodes_path)
        self.tolerance = tolerance

    def evaluate(self) -> SyntheticEvalMetrics:
        """Run evaluation and return metrics."""
        results: list[SyntheticEvalResult] = []

        by_difficulty: dict[str, list[bool]] = defaultdict(list)
        by_template: dict[str, list[bool]] = defaultdict(list)

        with open(self.episodes_path) as f:
            for line in f:
                ep = json.loads(line)

                # Skip non-synthetic episodes
                question = ep.get("question", {})
                ground_truth = question.get("ground_truth")
                if ground_truth is None:
                    continue

                template = question.get("template", "unknown")
                difficulty = question.get("difficulty", "unknown")
                question_text = question.get("question_text", "")

                # Get teacher answer
                teacher_trace = ep.get("teacher_gold_trace", {})
                teacher_answer = teacher_trace.get("final_answer")

                # Compare
                matched, reason = dicts_match_semantic(
                    ground_truth, teacher_answer, self.tolerance
                )

                result = SyntheticEvalResult(
                    question_text=question_text[:100],
                    template=template,
                    difficulty=difficulty,
                    matched=matched,
                    mismatch_reason=None if matched else reason,
                    expected=ground_truth,
                    actual=teacher_answer,
                )
                results.append(result)

                by_difficulty[difficulty].append(matched)
                by_template[template].append(matched)

        # Compute metrics
        if not results:
            return SyntheticEvalMetrics(
                accuracy=0.0, total=0, correct=0, mismatches=[]
            )

        total = len(results)
        correct = sum(1 for r in results if r.matched)
        mismatches = [r for r in results if not r.matched]

        return SyntheticEvalMetrics(
            accuracy=correct / total,
            total=total,
            correct=correct,
            accuracy_by_difficulty={
                k: sum(v) / len(v) for k, v in by_difficulty.items()
            },
            total_by_difficulty={k: len(v) for k, v in by_difficulty.items()},
            correct_by_difficulty={k: sum(v) for k, v in by_difficulty.items()},
            accuracy_by_template={
                k: sum(v) / len(v) for k, v in by_template.items()
            },
            mismatches=mismatches,
        )

    def print_report(self, metrics: SyntheticEvalMetrics) -> None:
        """Print formatted evaluation report."""
        print("=" * 60)
        print("SYNTHETIC EVALUATION REPORT")
        print("=" * 60)
        print()
        print(f"Overall Accuracy: {metrics.accuracy:.1%} ({metrics.correct}/{metrics.total})")
        print()

        print("By Difficulty:")
        for diff in ["EASY", "MEDIUM", "HARD", "VERY_HARD"]:
            if diff in metrics.accuracy_by_difficulty:
                acc = metrics.accuracy_by_difficulty[diff]
                tot = metrics.total_by_difficulty[diff]
                cor = metrics.correct_by_difficulty[diff]
                print(f"  {diff:10} {acc:.1%} ({cor}/{tot})")
        print()

        print("By Template:")
        for template, acc in sorted(metrics.accuracy_by_template.items()):
            print(f"  {template:40} {acc:.1%}")
        print()

        if metrics.mismatches:
            print(f"Mismatches ({len(metrics.mismatches)}):")
            for m in metrics.mismatches[:5]:
                print(f"  - {m.template}: {m.mismatch_reason}")
                print(f"    Expected: {m.expected}")
                print(f"    Actual:   {m.actual}")
        print()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate synthetic questions")
    parser.add_argument(
        "--episodes",
        type=str,
        default="data/episodes/episodes.jsonl",
        help="Path to episodes.jsonl",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance for float comparison (default 1%%)",
    )
    args = parser.parse_args()

    evaluator = SyntheticEvaluator(args.episodes, args.tolerance)
    metrics = evaluator.evaluate()
    evaluator.print_report(metrics)


if __name__ == "__main__":
    main()
