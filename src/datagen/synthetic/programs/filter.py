"""Deterministic filters for compositional programs (post-execution).

Filters are property-based and non-arbitrary.
No ambiguity or repair in Phase-1.
"""

from typing import List, Any
import json
import numpy as np


def filter_programs(
    programs: List[dict],
    min_group_size: int = 20,
    min_rows: int = 30,
    p_threshold: float = 0.05,
    target_count: int = 50,
) -> List[dict]:
    """
    Filter programs by deterministic property gates.

    Args:
        programs: List of program execution results (with answer, hooks, etc.)
        min_group_size: Minimum size for grouped tests
        min_rows: Minimum rows for scalar stats
        p_threshold: p-value threshold for significance
        target_count: Maximum programs to keep (keep top-K if > count)

    Returns:
        Filtered list of programs
    """
    filtered = []

    for prog in programs:
        # Extract answer and hooks
        answer = prog.get("answer", {})
        hooks = prog.get("hooks", [])
        stdout = prog.get("stdout", "")

        # Validity check: must have submitted answer
        if "Submitted:" not in stdout:
            continue

        # Validity gate: no NaN answers
        if _has_nan(answer):
            continue

        # Validity gate: min rows for scalar stats
        if _is_scalar_program(prog):
            if prog.get("row_count", 0) < min_rows:
                continue

        # Validity gate: no empty results
        if _has_empty_result(answer):
            continue

        # Signal gate: p-value threshold
        if _is_group_diff_program(prog):
            p_value = _extract_p_value(answer)
            if p_value is None or p_value > p_threshold:
                continue

        # Signal gate: unique winners for ranking
        if _is_ranking_program(prog):
            if not _has_unique_winner(stdout, hooks):
                continue

        filtered.append(prog)

    # Program count policy
    if len(filtered) > target_count:
        # Keep top-K by interestingness (property-based)
        filtered = sorted(
            filtered, key=lambda p: _interestingness_score(p), reverse=True
        )[:target_count]

    # Log if insufficient
    if len(filtered) < 20:
        print(f"[filter] Warning: Only {len(filtered)} programs generated (<20)")

    return filtered


def _has_nan(answer: Any) -> bool:
    """Check if answer contains NaN values."""
    if isinstance(answer, (float, int)):
        return np.isnan(answer)
    if isinstance(answer, dict):
        return any(_has_nan(v) for v in answer.values())
    if isinstance(answer, list):
        return any(_has_nan(v) for v in answer)
    return False


def _has_empty_result(answer: Any) -> bool:
    """Check if answer is empty."""
    if not answer:
        return True
    if isinstance(answer, dict):
        return len(answer) == 0 or all(v is None for v in answer.values())
    if isinstance(answer, (list, tuple)):
        return len(answer) == 0
    return False


def _is_scalar_program(prog: dict) -> bool:
    """Check if program is scalar-output type (mean, median, etc.)."""
    op_names = _op_names(prog)
    scalar_ops = {
        "mean",
        "median",
        "std",
        "variance",
        "mean_series",
        "median_series",
        "std_series",
        "max_series",
        "min_series",
    }
    return any(op in scalar_ops for op in op_names)


def _is_group_diff_program(prog: dict) -> bool:
    """Check if program is group-diff type (t-test, mwu)."""
    op_names = _op_names(prog)
    if "ttest_ind" in op_names:
        return True
    answer = prog.get("answer", {})
    return isinstance(answer, dict) and "p_value" in answer


def _is_ranking_program(prog: dict) -> bool:
    """Check if program is ranking type (argmax, argmin)."""
    op_names = _op_names(prog)
    ranking_ops = {
        "argmax_group",
        "argmin_group",
        "argmax_group_median",
        "argmin_group_median",
        "argmax_group_std",
        "argmin_group_std",
        "argmax_group_var",
        "argmin_group_var",
        "argmax_group_count",
        "argmin_group_count",
    }
    return any(op in ranking_ops for op in op_names)


def _extract_p_value(answer: dict) -> float:
    """Extract p-value from answer."""
    return float(answer.get("p_value", 1.0))


def _has_unique_winner(stdout: str, hooks: list) -> bool:
    """Check if ranking has a unique winner."""
    return "Unsorted" not in stdout and "Tie" not in stdout


def _interestingness_score(prog: dict) -> float:
    """Compute deterministic interestingness score."""
    answer = prog.get("answer", {})

    if "p_value" in answer:
        # Group diff: score by effect size (1 - p_value)
        return 1.0 - answer["p_value"]

    if isinstance(answer, dict):
        for key in ("variance", "std", "mean", "median"):
            if key in answer and isinstance(answer[key], (int, float)):
                return float(abs(answer[key]))

    if "group" in answer:
        # Ranking: score by gap
        return answer.get("gap", 0.0)

    return 0.0


def _op_names(prog: dict) -> list[str]:
    ops = prog.get("ops", [])
    names = []
    for op in ops:
        if isinstance(op, str):
            names.append(op)
        elif isinstance(op, dict) and "op_name" in op:
            names.append(str(op["op_name"]))
        else:
            op_name = getattr(op, "op_name", None)
            if op_name:
                names.append(str(op_name))
    return names
