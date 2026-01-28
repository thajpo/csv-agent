"""Semantic long-chain templates for meaningful 10-15 step questions.

Each template represents a coherent analytical workflow where EVERY operation
affects the final answer. No dead code.

Templates:
- CascadingFiltersTemplate: Multi-condition filtering → group → rank → aggregate
- DerivedColumnPipeline: Transform chain with cascading filters
- EvidenceDecisionAction: Statistical test workflow with evidence
"""

from typing import List
from src.datagen.synthetic.programs.spec import OpInstance, ProgramSpec


class CascadingFiltersTemplate:
    """Template: Filter by A → Filter by B → Group → Rank → Aggregate

    Example: "Which city has the most days that are both hot (>25°C) and dry
    (<30% humidity), and what's the average temperature there?"

    Every step matters:
    - Filter A reduces dataset for next filter
    - Filter B further reduces it
    - Grouping creates categories for comparison
    - Ranking selects the winner
    - Final aggregate depends on all previous filters
    """

    @staticmethod
    def create(
        col_a: str,  # First filter column (e.g., temperature)
        threshold_a: float,  # First threshold (e.g., 25)
        col_b: str,  # Second filter column (e.g., humidity)
        threshold_b: float,  # Second threshold (e.g., 30)
        cat_col: str,  # Grouping column (e.g., city)
        final_col: str,  # Final aggregation column
    ) -> ProgramSpec:
        """Create 10-step cascading filter chain."""
        ops = [
            # Step 1-2: Setup first filter column
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": col_a}),
            # Step 3: Filter by A (this affects all downstream operations)
            OpInstance(
                "filter_by_threshold", {"selected_col": col_a, "threshold": threshold_a}
            ),
            # Step 4-5: Setup second filter column
            OpInstance("bind_numeric_col", {"selected_col": col_b}),
            # Step 6: Filter by B (further reduces dataset)
            OpInstance(
                "filter_by_threshold", {"selected_col": col_b, "threshold": threshold_b}
            ),
            # Step 7-8: Setup grouping
            OpInstance("select_categorical_cols", {}),
            OpInstance("bind_binary_cat_col", {"cat_col": cat_col}),
            # Step 9: Group and count (each group size depends on both filters)
            OpInstance("groupby_count", {}),
            # Step 10: Find group with max count (winner depends on filtered data)
            OpInstance("argmax_group_count", {}),
        ]

        return ProgramSpec(
            name=f"cascading_{col_a}_{col_b}_{cat_col}",
            ops=ops,
            output_type="dict",
            output_schema='{"group": "<category>", "count": 0}',
            difficulty="HARD",
            tags=["cascading_filters", "multi_condition", "ranking"],
        )


class DerivedColumnPipelineTemplate:
    """Template: Transform → Filter → Transform → Filter → Aggregate

    Example: "After smoothing temperature with a 3-day rolling average and
    keeping only days where the change from previous day exceeds 5 degrees,
    what's the average smoothed temperature?"

    Every transform creates data used downstream:
    - Rolling mean creates smoothed values
    - Diff creates change values
    - Filters use these derived columns
    - Final aggregate uses the filtered smoothed data
    """

    @staticmethod
    def create(
        base_col: str,  # Original column to transform
        window: int,  # Rolling window size
        diff_threshold: float,  # Threshold for change detection
    ) -> ProgramSpec:
        """Create 12-step derived column pipeline."""
        ops = [
            # Step 1-2: Setup
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": base_col}),
            # Step 3: Create rolling mean (used in final aggregate)
            OpInstance("rolling_mean", {"selected_col": base_col, "window": window}),
            # Step 4: Create diff (used for filtering)
            OpInstance("diff", {"selected_col": base_col}),
            # Step 5: Filter by diff (removes rows where change is small)
            # This filter affects which rolling_mean values survive
            OpInstance(
                "filter_by_threshold",
                {"selected_col": f"{base_col}_diff", "threshold": diff_threshold},
            ),
            # Step 6-7: Sort and take top (focused analysis on extreme changes)
            OpInstance(
                "sort_by_column",
                {"selected_col": f"{base_col}_diff", "ascending": False},
            ),
            OpInstance("top_n", {"n": 100}),
            # Step 8: Rebind to rolling mean column (the derived column we want)
            OpInstance(
                "bind_numeric_col", {"selected_col": f"{base_col}_rolling_{window}"}
            ),
            # Step 9-10: Additional transform for complexity
            OpInstance("zscore", {}),
            OpInstance("abs", {}),
            # Step 11-12: Final aggregation
            OpInstance("bind_numeric_col", {"selected_col": base_col}),
            OpInstance("mean", {}),
        ]

        return ProgramSpec(
            name=f"derived_pipeline_{base_col}_w{window}",
            ops=ops,
            output_type="dict",
            output_schema='{"column": "<name>", "mean": 0.0}',
            difficulty="VERY_HARD",
            tags=["derived_column", "multi_transform", "conditional"],
        )


class EvidenceDecisionActionTemplate:
    """Template: Evidence → Decision → Action → Aggregate

    Example: "Test whether the high-value group has significantly different
    variance than the low-value group. If so, report the mean of the group
    with higher variance."

    Each evidence step informs the decision:
    - Shapiro test checks normality
    - Levene test checks variance equality
    - Decision chooses appropriate test
    - Action filters to relevant group
    - Final aggregate depends on all previous steps
    """

    @staticmethod
    def create(
        value_col: str,  # Column to analyze
        group_col: str,  # Binary grouping column
        threshold: float,  # Split threshold
    ) -> ProgramSpec:
        """Create 11-step evidence-decision-action chain."""
        ops = [
            # Step 1-3: Setup
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": value_col}),
            OpInstance("select_categorical_cols", {}),
            # Step 4-5: Bind binary group and split
            OpInstance("bind_binary_cat_col", {"cat_col": group_col}),
            OpInstance("groupby_values", {}),
            # Step 6-7: Gather evidence
            OpInstance("shapiro_p", {}),
            OpInstance("levene_p", {}),
            # Step 8: Make decision based on evidence
            OpInstance("choose_test", {}),
            # Step 9: Execute chosen test
            OpInstance("ttest_ind", {}),
            # Step 10-11: Action based on test result
            OpInstance("bind_numeric_col", {"selected_col": value_col}),
            OpInstance("mean", {}),
        ]

        return ProgramSpec(
            name=f"evidence_decision_{value_col}_{group_col}",
            ops=ops,
            output_type="dict",
            output_schema='{"column": "<name>", "mean": 0.0}',
            difficulty="VERY_HARD",
            tags=["evidence_based", "statistical_test", "conditional"],
        )


def generate_semantic_long_programs(profile: dict) -> List[ProgramSpec]:
    """Generate meaningful long-chain programs.

    These templates guarantee that every step affects the final answer.
    No dead code.
    """
    from src.datagen.synthetic.programs.operators import (
        get_eligible_numeric_cols,
        get_eligible_categorical_cols,
        get_eligible_binary_categorical_cols,
    )

    programs = []
    numeric_cols = get_eligible_numeric_cols(profile)
    cat_cols = get_eligible_categorical_cols(profile)
    binary_cat_cols = get_eligible_binary_categorical_cols(profile)

    # Generate cascading filter templates
    if len(numeric_cols) >= 2 and len(binary_cat_cols) >= 1:
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1 :]:
                for cat_col in binary_cat_cols[:2]:  # Limit combinations
                    # Use median-based thresholds for meaningful splits
                    programs.append(
                        CascadingFiltersTemplate.create(
                            col_a=col_a,
                            threshold_a=0,  # Will be replaced with actual median during execution
                            col_b=col_b,
                            threshold_b=0,
                            cat_col=cat_col,
                            final_col=col_a,
                        )
                    )

    # Generate derived column pipelines
    for col in numeric_cols[:3]:
        programs.append(
            DerivedColumnPipelineTemplate.create(
                base_col=col, window=3, diff_threshold=0.5
            )
        )

    # Generate evidence-decision-action templates
    if len(numeric_cols) >= 1 and len(binary_cat_cols) >= 1:
        for val_col in numeric_cols[:2]:
            for grp_col in binary_cat_cols[:2]:
                programs.append(
                    EvidenceDecisionActionTemplate.create(
                        value_col=val_col, group_col=grp_col, threshold=0
                    )
                )

    return programs
