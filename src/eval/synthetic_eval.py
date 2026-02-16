"""
Evaluate teacher performance on synthetic (template-based) questions.

Synthetic questions have known ground truth from template execution.
This evaluator compares teacher answers against that ground truth.
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.datagen.teacher import answers_match
from src.datagen.synthetic.profiler import DataProfiler
from src.datagen.synthetic.templates import ALL_TEMPLATES, CompositionTemplate
from src.envs.csv_env import LocalCSVAnalysisEnv
from csv_spec import hash_artifact
from src.datagen.shared.submission import parse_submission


_KEY_ALIASES: dict[str, list[str]] = {
    "target": ["target_column", "target_col"],
    "predictor": ["predictor_column", "best_predictor", "most_correlated_column"],
    "association_score": [
        "correlation",
        "absolute_correlation",
        "abs_correlation",
        "corr",
        "spearman_rho",
    ],
    "columns": ["pair"],
    "grouping_column": ["group_col", "group_column"],
    "test_statistic": [
        "t_statistic",
        "t_stat",
        "t",
        "u_statistic",
        "f_statistic",
        "f_stat",
        "f",
        "chi_squared",
        "ks_statistic",
        "levene_statistic",
    ],
    "significance_score": ["p_value", "p"],
    "fit_score": ["r_squared", "r2", "r2_score"],
    "fit_score_adj": ["adj_r_squared", "adj_r2"],
    "n_influential": ["n_significant"],
    "method_choice": ["test_used"],
    "effect_size_kind": ["effect_size_type"],
    "effect_size": ["eta_squared"],
    "df": ["degrees_of_freedom"],
    "original_association": ["original_correlation"],
    "clean_association": ["clean_correlation"],
    "log_association": ["log_correlation"],
    "mean1": ["group1_mean", "mean_low_group"],
    "mean2": ["group2_mean", "mean_high_group"],
    "significant": ["significant_at_0.05", "is_significant"],
}

_SORTED_LIST_KEYS = {"columns", "predictors", "pair"}


@dataclass
class TemplateExecutionResult:
    answer: Any
    answer_hash: str | None
    hooks: list[dict]


@dataclass
class SyntheticEvalResult:
    """Result of evaluating a single synthetic episode."""

    question_text: str
    template: str
    difficulty: str
    dataset: str

    # Match results
    matched: bool
    mismatch_reason: str | None = None

    # Hook metrics
    hook_matches: int = 0
    hook_expected: int = 0
    hook_teacher: int = 0
    hook_recall: float = 0.0
    hook_precision: float = 0.0

    # Raw values
    expected: Any = None
    actual: Any = None


@dataclass
class SyntheticEvalMetrics:
    """Aggregate metrics for synthetic evaluation."""

    accuracy: float
    total: int
    correct: int

    # Hook metrics
    avg_hook_recall: float
    avg_hook_precision: float

    # By difficulty
    accuracy_by_difficulty: dict[str, float] = field(default_factory=dict)
    total_by_difficulty: dict[str, int] = field(default_factory=dict)
    correct_by_difficulty: dict[str, int] = field(default_factory=dict)

    # By template
    accuracy_by_template: dict[str, float] = field(default_factory=dict)

    # Mismatches for debugging
    mismatches: list[SyntheticEvalResult] = field(default_factory=list)


class TemplateExecutionSession:
    """Executes templates in a shared sandbox for a single dataset."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path).resolve()
        self.profiler = DataProfiler()
        self.profile: dict[str, Any] | None = None
        self.env: LocalCSVAnalysisEnv | None = None
        self.state: dict | None = None
        self._cache: dict[str, TemplateExecutionResult] = {}

    async def setup(self) -> None:
        self.profile = self.profiler.analyze(str(self.csv_path))
        self.env = LocalCSVAnalysisEnv(csv_path=str(self.csv_path))
        self.state = await self.env.setup_state({})

    async def cleanup(self) -> None:
        if self.env and self.state:
            await self.env.destroy_sandbox(self.state["sandbox_id"])

    async def execute(
        self,
        template: CompositionTemplate,
        params: dict[str, Any] | None,
    ) -> TemplateExecutionResult | None:
        params = params or {}
        cache_key = json.dumps({"template": template.name, **params}, sort_keys=True)
        if cache_key in self._cache:
            return self._cache[cache_key]

        code = template.instantiate(self.profile or {}, params=params)

        await self.env.reset(
            self.state["sandbox_id"],
            self.state["python_state"],
        )

        output = await self.env.python(
            code=code,
            sandbox_id=self.state["sandbox_id"],
            python_state=self.state["python_state"],
        )

        submission, success = parse_submission(output)
        if not success or submission is None:
            return None

        answer = submission.get("__csv_agent_answer__")
        hooks = submission.get("hooks", [])
        answer_hash = hash_artifact(answer) if answer is not None else None

        result = TemplateExecutionResult(
            answer=answer,
            answer_hash=answer_hash,
            hooks=hooks,
        )
        self._cache[cache_key] = result
        return result


def _normalize_key(key: str) -> str:
    return key.lower().replace("-", "_").replace("‑", "_")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return stripped
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def _canonicalize_teacher_dict(
    teacher: dict,
    expected: dict,
) -> tuple[dict, dict, list[str]]:
    """Canonicalize teacher dict to match expected schema.

    Returns:
        - canonical: teacher dict with values aligned to expected keys
        - expected_norm: normalized expected dict (with sorted lists)
        - missing: list of keys missing from teacher
    """
    teacher_norm = {_normalize_key(k): _normalize_value(v) for k, v in teacher.items()}
    expected_norm = {
        _normalize_key(k): _normalize_value(v) for k, v in expected.items()
    }

    canonical: dict[str, Any] = {}
    missing: list[str] = []

    for key, expected_value in expected_norm.items():
        if key in teacher_norm:
            value = teacher_norm[key]
        else:
            value = None
            for alias in _KEY_ALIASES.get(key, []):
                alias_key = _normalize_key(alias)
                if alias_key in teacher_norm:
                    value = teacher_norm[alias_key]
                    break

        if value is None:
            missing.append(key)
            continue

        if key in _SORTED_LIST_KEYS and isinstance(value, list):
            value = sorted(value, key=lambda v: str(v))
            expected_value = sorted(expected_value, key=lambda v: str(v))
            expected_norm[key] = expected_value

        canonical[key] = value

    return canonical, expected_norm, missing


def _compare_answers(
    expected: Any,
    actual: Any,
    float_tol: float,
    p_value_tol: float,
) -> tuple[bool, str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, "actual answer is not a dict"

        canonical, expected_norm, missing = _canonicalize_teacher_dict(actual, expected)
        if missing:
            return False, f"missing keys: {', '.join(sorted(missing))}"

        matched = answers_match(
            None,
            None,
            expected_norm,
            canonical,
            float_tol=float_tol,
            p_value_tol=p_value_tol,
        )
        return matched, "match" if matched else "value mismatch"

    expected_norm = _normalize_value(expected)
    actual_norm = _normalize_value(actual)
    matched = answers_match(
        None,
        None,
        expected_norm,
        actual_norm,
        float_tol=float_tol,
        p_value_tol=p_value_tol,
    )
    return matched, "match" if matched else "value mismatch"


def _compute_hook_metrics(
    teacher_hooks: list[dict] | None,
    expected_hooks: list[dict] | None,
) -> tuple[int, int, int, float, float]:
    teacher_hooks = teacher_hooks or []
    expected_hooks = expected_hooks or []

    teacher_hashes = {h.get("value_hash") for h in teacher_hooks if h.get("value_hash")}
    expected_hashes = {
        h.get("value_hash") for h in expected_hooks if h.get("value_hash")
    }

    matches = len(teacher_hashes & expected_hashes)
    expected_count = len(expected_hashes)
    teacher_count = len(teacher_hashes)

    recall = matches / expected_count if expected_count else 0.0
    precision = matches / teacher_count if teacher_count else 0.0

    return matches, expected_count, teacher_count, recall, precision


def _get_template_by_name(name: str) -> CompositionTemplate | None:
    for template in ALL_TEMPLATES:
        if template.name == name:
            return template
    return None


def _template_label(template_name: str, params: dict[str, Any] | None) -> str:
    if not params:
        return template_name
    params_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{template_name} [{params_str}]"


class SyntheticEvaluator:
    """Evaluates teacher performance on synthetic questions."""

    def __init__(
        self,
        episodes_path: str,
        float_tol: float = 0.1,
        p_value_tol: float = 0.002,
    ):
        """
        Initialize evaluator.

        Args:
            episodes_path: Path to episodes.jsonl
            float_tol: Absolute tolerance for float comparison
            p_value_tol: Absolute tolerance for p-value comparison
        """
        self.episodes_path = Path(episodes_path)
        self.float_tol = float_tol
        self.p_value_tol = p_value_tol

    async def evaluate_async(self) -> SyntheticEvalMetrics:
        results: list[SyntheticEvalResult] = []

        by_difficulty: dict[str, list[bool]] = defaultdict(list)
        by_template: dict[str, list[bool]] = defaultdict(list)
        hook_recalls: list[float] = []
        hook_precisions: list[float] = []

        episodes_by_dataset: dict[str, list[dict]] = defaultdict(list)
        dataset_csv_map: dict[str, str] = {}

        with open(self.episodes_path) as f:
            for line in f:
                ep = json.loads(line)
                csv_source = ep.get("csv_source")
                if not csv_source:
                    continue
                dataset = Path(csv_source).parent.name
                episodes_by_dataset[dataset].append(ep)
                dataset_csv_map[dataset] = csv_source

        for dataset, episodes in episodes_by_dataset.items():
            session = TemplateExecutionSession(dataset_csv_map[dataset])
            await session.setup()

            try:
                for ep in episodes:
                    question = ep.get("question", {})
                    template_name = question.get("template_name") or question.get(
                        "_template"
                    )
                    template_params = question.get("template_params")
                    difficulty = question.get("difficulty", "unknown")
                    question_text = question.get("question_text", "")

                    teacher_trace = ep.get("gold_trace") or ep.get(
                        "teacher_gold_trace", {}
                    )
                    teacher_answer = teacher_trace.get("final_answer")
                    all_hooks = []
                    for turn in teacher_trace.get("turns", []):
                        exec_result = turn.get("execution", {})
                        all_hooks.extend(exec_result.get("hooks", []))
                    teacher_hooks = all_hooks or teacher_trace.get("hooks", [])

                    expected_answer = None
                    expected_hooks = []

                    if template_name:
                        template = _get_template_by_name(template_name)
                        if template is None:
                            matched = False
                            reason = "template not found"
                        else:
                            exec_result = await session.execute(
                                template, template_params
                            )
                            if exec_result is None:
                                matched = False
                                reason = "template execution failed"
                            else:
                                expected_answer = exec_result.answer
                                expected_hooks = exec_result.hooks
                                matched, reason = _compare_answers(
                                    expected_answer,
                                    teacher_answer,
                                    self.float_tol,
                                    self.p_value_tol,
                                )
                    else:
                        continue

                    (matches, expected_count, teacher_count, recall, precision) = (
                        _compute_hook_metrics(teacher_hooks, expected_hooks)
                    )

                    hook_recalls.append(recall)
                    hook_precisions.append(precision)

                    template_label = _template_label(
                        template_name or "unknown", template_params
                    )
                    result = SyntheticEvalResult(
                        question_text=question_text[:100],
                        template=template_label,
                        difficulty=difficulty,
                        dataset=dataset,
                        matched=matched,
                        mismatch_reason=None if matched else reason,
                        hook_matches=matches,
                        hook_expected=expected_count,
                        hook_teacher=teacher_count,
                        hook_recall=recall,
                        hook_precision=precision,
                        expected=expected_answer,
                        actual=teacher_answer,
                    )
                    results.append(result)

                    by_difficulty[difficulty].append(matched)
                    by_template[template_label].append(matched)

            finally:
                await session.cleanup()

        if not results:
            return SyntheticEvalMetrics(
                accuracy=0.0,
                total=0,
                correct=0,
                avg_hook_recall=0.0,
                avg_hook_precision=0.0,
                mismatches=[],
            )

        total = len(results)
        correct = sum(1 for r in results if r.matched)
        mismatches = [r for r in results if not r.matched]

        avg_hook_recall = sum(hook_recalls) / len(hook_recalls) if hook_recalls else 0.0
        avg_hook_precision = (
            sum(hook_precisions) / len(hook_precisions) if hook_precisions else 0.0
        )

        return SyntheticEvalMetrics(
            accuracy=correct / total,
            total=total,
            correct=correct,
            avg_hook_recall=avg_hook_recall,
            avg_hook_precision=avg_hook_precision,
            accuracy_by_difficulty={
                k: sum(v) / len(v) for k, v in by_difficulty.items()
            },
            total_by_difficulty={k: len(v) for k, v in by_difficulty.items()},
            correct_by_difficulty={k: sum(v) for k, v in by_difficulty.items()},
            accuracy_by_template={k: sum(v) / len(v) for k, v in by_template.items()},
            mismatches=mismatches,
        )

    def evaluate(self) -> SyntheticEvalMetrics:
        return asyncio.run(self.evaluate_async())

    def print_report(self, metrics: SyntheticEvalMetrics) -> None:
        """Print formatted evaluation report."""
        print("=" * 60)
        print("SYNTHETIC EVALUATION REPORT")
        print("=" * 60)
        print()
        print(
            f"Overall Accuracy: {metrics.accuracy:.1%} ({metrics.correct}/{metrics.total})"
        )
        print(f"Avg Hook Recall:  {metrics.avg_hook_recall:.1%}")
        print(f"Avg Hook Precision: {metrics.avg_hook_precision:.1%}")
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
            print(f"  {template:55} {acc:.1%}")
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
        "--float-tol",
        type=float,
        default=0.1,
        help="Absolute tolerance for float comparison (default 0.1)",
    )
    parser.add_argument(
        "--p-value-tol",
        type=float,
        default=0.002,
        help="Absolute tolerance for p-value comparison (default 0.002)",
    )
    args = parser.parse_args()

    evaluator = SyntheticEvaluator(
        args.episodes,
        float_tol=args.float_tol,
        p_value_tol=args.p_value_tol,
    )
    metrics = evaluator.evaluate()
    evaluator.print_report(metrics)


if __name__ == "__main__":
    main()
