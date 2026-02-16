"""Filter utilities for synthetic questions.

Centralizes dataset viability, column eligibility, and program output filters.
"""

import re
from dataclasses import dataclass

from src.core.config import config


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


def check_dataset_viability(profile: dict) -> FilterResult:
    """Check if dataset is viable for synthetic generation.

    Scope 1: Dataset viability (profiler-based)
    - Minimum rows/columns
    - Not all columns heavily missing
    - At least one eligible column
    """
    cfg = config.synthetic_config
    shape = profile.get("shape", {})
    rows = shape.get("rows", 0) or 0
    cols = shape.get("columns", 0) or 0

    if rows < cfg.min_rows:
        return FilterResult(passed=False, reason=f"too_few_rows:{rows}")

    if cols < cfg.min_columns:
        return FilterResult(passed=False, reason=f"too_few_cols:{cols}")

    # Check not all columns are heavily missing
    columns = profile.get("columns", {})
    if columns:
        high_missing = [
            col
            for col, info in columns.items()
            if info.get("missing_pct", 0) >= cfg.max_missing_pct
        ]
        if len(high_missing) == cols:
            return FilterResult(passed=False, reason="all_heavy_missing")

    # Check at least one eligible column exists
    eligible = _get_eligible_columns(profile)
    if not eligible:
        return FilterResult(passed=False, reason="no_eligible_columns")

    return FilterResult(passed=True, reason=None)


def check_column_eligibility(column_info: dict, row_count: int) -> FilterResult:
    """Check if a single column is eligible for use in questions.

    Scope 2: Column eligibility (profiler-based)
    - Not ID-like
    - Not heavily missing
    - Has variance (unique_count > 1)
    """
    cfg = config.synthetic_config

    # Check missingness
    if column_info.get("missing_pct", 0) >= cfg.heavy_missing_threshold:
        return FilterResult(passed=False, reason="heavy_missingness")

    # Check variance
    if column_info.get("unique_count", 0) <= 1:
        return FilterResult(passed=False, reason="zero_variance")

    # Check ID-like
    if _is_id_like_column_info(column_info, row_count):
        return FilterResult(passed=False, reason="id_like_column")

    return FilterResult(passed=True, reason=None)


def check_question_viability(question_text: str, profile: dict) -> FilterResult:
    """Check if verbalized question meets quality standards.

    Scope 3: Verbalized question viability (synthetic only)
    - No method terms
    - No explicit column names
    - Concise (<= max_sentences)
    """
    cfg = config.synthetic_config
    lowered = question_text.lower()

    # Check for forbidden method terms
    if any(term in lowered for term in FORBIDDEN_METHOD_TERMS):
        return FilterResult(passed=False, reason="forbidden_method_terms")

    # Check for explicit column names
    for col in profile.get("columns", {}):
        col_name = str(col).strip().lower()
        if len(col_name) >= 4 and col_name in lowered:
            return FilterResult(passed=False, reason="mentions_column_names")

    # Check sentence count
    question_for_counting = re.split(
        r"Return as JSON|e\.g\.:", question_text, maxsplit=1
    )[0]
    sentence_count = len(re.findall(r"[.!?]", question_for_counting))
    if sentence_count > cfg.max_sentences:
        return FilterResult(passed=False, reason="too_many_sentences")

    return FilterResult(passed=True, reason=None)


def check_program_output(program_result: dict) -> FilterResult:
    """Check if program execution produced valid output.

    Scope 4: Program output filters (program-specific)
    - Must have submitted answer
    - No NaN or empty answers
    - Minimum row count for scalar stats
    """
    cfg = config.synthetic_config

    answer = program_result.get("answer")

    # Must have answer
    if answer is None:
        return FilterResult(passed=False, reason="no_answer_submitted")

    # Check for NaN/empty
    if isinstance(answer, float) and (answer != answer):  # NaN check
        return FilterResult(passed=False, reason="nan_answer")

    if answer == "" or answer == [] or answer == {}:
        return FilterResult(passed=False, reason="empty_answer")

    # Check row count for scalar stats
    row_count = program_result.get("row_count", 0)
    if row_count < cfg.min_rows_for_stats:
        return FilterResult(passed=False, reason=f"too_few_rows:{row_count}")

    return FilterResult(passed=True, reason=None)


def _get_eligible_columns(profile: dict) -> list[str]:
    """Get all eligible columns (numeric or categorical)."""
    row_count = profile.get("shape", {}).get("rows", 0) or 0
    eligible = []

    for col, info in profile.get("columns", {}).items():
        col_type = info.get("type")
        if col_type not in ("numeric", "categorical"):
            continue

        result = check_column_eligibility(info, row_count)
        if result.passed:
            eligible.append(col)

    return eligible


def _is_id_like_column_info(info: dict, row_count: int) -> bool:
    """Check if column info indicates ID-like column."""
    cfg = config.synthetic_config

    # Check profiler's flag
    if info.get("is_index_like", False):
        return True

    # Check unique ratio
    unique_count = info.get("unique_count")
    if row_count and isinstance(unique_count, int):
        threshold = max(2, int(cfg.id_like_unique_threshold * row_count))
        if unique_count >= threshold:
            return True

    return False


def log_drop(question_id: str, reason: str) -> None:
    """Log dropped question for later analysis."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"[Filter drop] {question_id}: {reason}")


def apply_filters(question: dict, profile: dict) -> FilterResult:
    """Apply all relevant filters based on question source.

    This is a convenience wrapper that applies the appropriate filters
    based on question type. For more control, use individual check functions.
    """
    source = question.get("source")

    # All sources need dataset viability
    result = check_dataset_viability(profile)
    if not result.passed:
        return result

    # Template/procedural questions need question viability check
    if source in ("template", "procedural"):
        question_text = question.get("question_text") or question.get(
            "question_mechanical", ""
        )
        if question_text:
            result = check_question_viability(question_text, profile)
            if not result.passed:
                return result

    return FilterResult(passed=True, reason=None)
