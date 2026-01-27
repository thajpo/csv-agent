"""Filter utilities for synthetic questions.

Centralizes dataset viability, column eligibility, and program output filters.
"""

from dataclasses import dataclass
from typing import Any


# Method terms to avoid in verbalized questions
FORBIDDEN_METHOD_TERMS = {
    "calculate",
    "group by",
    "groupby",
    "use pandas",
    "pandas",
    ".sort",
    ".value_counts",
    ".agg",
    "apply",
    "lambda",
}


@dataclass
class FilterResult:
    passed: bool
    reason: str | None  # None if passed, explanation if dropped


def apply_filters(
    question: dict,
    profile: dict,
) -> FilterResult:
    """Apply minimal 'obvious junk' filters for synthetic questions.

    Scopes:
    - Dataset viability (min rows/cols, not all missing, has eligible columns)
    - Column eligibility (ID-like/near-unique, missing_pct >= 95, unique_count <= 1)
    - Verbalized question viability (no method terms, no column names, <= 3 sentences)
    - Program output filters (no NaN/empty, min rows for scalar stats, p-value threshold, unique winners, top-K by interestingness)

    Program output filters live in `synthetic/programs/filter.py` and are
    documented separately; shared filters should surface common drop reasons.

    Returns FilterResult with a drop reason if applicable.
    """
    # TODO: Implement filter logic based on profiler signals and question content
    # This is a placeholder - actual implementation will use profiler output

    return FilterResult(passed=True, reason=None)


def log_drop(question_id: str, reason: str) -> None:
    """Log dropped question for later analysis.

    Used to audit filter impact and tune thresholds without losing data on drop reasons.
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"[Filter drop] {question_id}: {reason}")
