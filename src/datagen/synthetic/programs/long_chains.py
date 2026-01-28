"""Long-chain templates for 10-15 step complexity.

Instead of relying on grammar search to accidentally find long chains,
we define explicit compositional patterns that are guaranteed to be complex.
"""

from typing import List
from src.datagen.synthetic.programs.spec import OpInstance, ProgramSpec


def make_long_chain_template_1(column: str, threshold: float = 0) -> List[OpInstance]:
    """Template: Filter → Sort → Top N → Transform → Aggregate

    Chain (10 steps):
    1. select_numeric_cols
    2. bind_numeric_col (target column)
    3. filter_by_threshold (keep high values)
    4. sort_by_column (descending)
    5. top_n (take top 20)
    6. zscore (normalize)
    7. mean_series (compute mean of normalized values)
    8. [Additional transforms can be added]

    Actually, this needs to be constructed carefully with the type system.
    Let me create explicit 10-step chains.
    """
    return [
        OpInstance("select_numeric_cols", {}),
        OpInstance("bind_numeric_col", {"selected_col": column}),
        OpInstance(
            "filter_by_threshold", {"selected_col": column, "threshold": threshold}
        ),
        OpInstance("sort_by_column", {"selected_col": column, "ascending": False}),
        OpInstance("top_n", {"n": 20}),
        OpInstance("zscore", {}),
        OpInstance("mean_series", {}),
    ]


def create_explicit_10_step_chain(
    col1: str, col2: str, threshold: float = 0
) -> ProgramSpec:
    """Create a guaranteed 10-step chain.

    Pattern:
    1-2. Setup: select + bind col1
    3-4. First transform: filter + sort col1
    5-6. Second transform: top_n + bind col2
    7-8. Third transform: filter + sort col2
    9-10. Analysis: zscore + mean
    """
    ops = [
        OpInstance("select_numeric_cols", {}),
        OpInstance("bind_numeric_col", {"selected_col": col1}),
        OpInstance(
            "filter_by_threshold", {"selected_col": col1, "threshold": threshold}
        ),
        OpInstance("sort_by_column", {"selected_col": col1, "ascending": False}),
        OpInstance("top_n", {"n": 50}),
        OpInstance("bind_num_col_1", {"num_col_1": col2}),
        OpInstance(
            "filter_by_threshold", {"selected_col": col2, "threshold": threshold}
        ),
        OpInstance("sort_by_column", {"selected_col": col2, "ascending": True}),
        OpInstance("top_n", {"n": 10}),
        OpInstance("mean", {}),
    ]

    return ProgramSpec(
        name=f"long_chain_{col1}_{col2}",
        ops=ops,
        output_type="dict",
        output_schema="Long chain with multiple filters and sorts",
        difficulty="HARD",
        tags=["long_chain", "explicit_template"],
    )


def create_explicit_12_step_chain(col1: str, col2: str) -> ProgramSpec:
    """Create a guaranteed 12-step chain.

    Pattern with cumulative and windowed operations:
    1-2. Setup: select + bind
    3. cumulative_sum (track running total)
    4. rolling_mean (smooth with window)
    5. filter_by_threshold (filter on smoothed values)
    6. sort_by_column
    7. top_n
    8. bind second column
    9. diff (compute changes)
    10. percentile_rank
    11. bind_numeric_col (rebind for final analysis)
    12. mean (final aggregate)
    """
    ops = [
        OpInstance("select_numeric_cols", {}),
        OpInstance("bind_numeric_col", {"selected_col": col1}),
        OpInstance("cumulative_sum", {"selected_col": col1}),
        OpInstance("rolling_mean", {"selected_col": col1, "window": 3}),
        OpInstance("filter_by_threshold", {"selected_col": col1, "threshold": 0}),
        OpInstance("sort_by_column", {"selected_col": col1, "ascending": False}),
        OpInstance("top_n", {"n": 30}),
        OpInstance("bind_num_col_1", {"num_col_1": col2}),
        OpInstance("diff", {"selected_col": col2}),
        OpInstance("percentile_rank", {"selected_col": col2}),
        OpInstance("bind_numeric_col", {"selected_col": col2}),
        OpInstance("mean", {}),
    ]

    return ProgramSpec(
        name=f"very_long_chain_{col1}_{col2}",
        ops=ops,
        output_type="dict",
        output_schema="Very long chain with cumulative, rolling, and ranking operations",
        difficulty="VERY_HARD",
        tags=["very_long_chain", "explicit_template"],
    )


def generate_long_chain_programs(profile: dict) -> List[ProgramSpec]:
    """Generate explicit long-chain programs.

    These bypass the grammar search and use predefined templates
    guaranteed to produce 10-15 step chains.
    """
    from src.datagen.synthetic.programs.operators import get_eligible_numeric_cols

    numeric_cols = get_eligible_numeric_cols(profile)
    programs = []

    # Generate 10-step chains for each column pair
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1 :]:
            programs.append(create_explicit_10_step_chain(col1, col2))
            programs.append(create_explicit_12_step_chain(col1, col2))

    return programs
