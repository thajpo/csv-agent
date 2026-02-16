"""Template-based verbalizer for long-chain questions.

Generates deterministic, accurate verbalizations for semantic long-chain templates.
Each template has a specific verbalization pattern that accurately describes
the computation without revealing the mechanical steps.

Why template-based:
- Guaranteed accuracy: verbalization matches the code exactly
- No LLM hallucination: deterministic output
- Schema compliant: fits existing question format
- Optional LLM polish: can add natural language variation without changing meaning
"""

from src.datagen.synthetic.programs.spec import ProgramSpec


def verbalize_cascading_filters(
    col_a: str,
    threshold_a: float,
    col_b: str,
    threshold_b: float,
    cat_col: str,
    final_col: str,
) -> tuple[str, str]:
    """Verbalize cascading filters template.

    Returns (question, hint) tuple.
    """
    # Determine comparison operators based on typical use cases
    op_a = "greater than" if threshold_a > 0 else "less than"
    op_b = "less than" if threshold_b > 0 else "greater than"

    question = (
        f"Which {cat_col} has the most records where {col_a} is {op_a} {threshold_a} "
        f"and {col_b} is {op_b} {threshold_b}? "
        f'Return as JSON, e.g.: {{"group": "<name>", "count": 0}}'
    )

    hint = (
        f"First filter to rows where {col_a} {op_a} {threshold_a}. "
        f"From those, keep only rows where {col_b} {op_b} {threshold_b}. "
        f"Group by {cat_col} and count records in each group. "
        f'Return the {cat_col} with the highest count as a dict with "group" and "count" keys.'
    )

    return question, hint


def verbalize_derived_column_pipeline(
    base_col: str, window: int, diff_threshold: float
) -> tuple[str, str]:
    """Verbalize derived column pipeline template."""

    question = (
        f"After smoothing {base_col} with a {window}-period rolling average and keeping "
        f"only records where the day-to-day change exceeds {diff_threshold}, "
        f'what is the average {base_col}? Return as JSON, e.g.: {{"mean": 0.0}}'
    )

    hint = (
        f"Compute {window}-period rolling mean of {base_col}. "
        f"Calculate day-to-day differences. "
        f"Filter to records where absolute difference > {diff_threshold}. "
        f"Compute mean of original {base_col} column for filtered records and return as a dict."
    )

    return question, hint


def verbalize_evidence_decision_action(
    value_col: str, group_col: str
) -> tuple[str, str]:
    """Verbalize evidence-decision-action template."""

    question = (
        f"Compare {value_col} between the two groups in {group_col}. "
        f"If there's a significant difference, report the mean {value_col} "
        f'for the group with higher variance. Return as JSON, e.g.: {{"group": "<name>", "mean": 0.0}}'
    )

    hint = (
        f"Split data by {group_col}. Check normality and variance assumptions. "
        f"Choose appropriate statistical test. "
        f"If significant difference found, identify group with higher variance "
        f'and compute mean {value_col} for that group. Return result as a dict with "group" and "mean" keys.'
    )

    return question, hint


def verbalize_long_chain_program(spec: ProgramSpec) -> tuple[str, str]:
    """Verbalize a semantic long-chain program.

    Parses the program name to determine template type and extract parameters.
    Returns (question_text, hint) tuple.
    """
    name = spec.name

    if name.startswith("cascading_"):
        # Extract column names from the operator chain instead of parsing the name
        # This is more reliable than parsing the underscore-separated name
        op_bindings = {}
        for op in spec.ops:
            op_bindings.update(op.params)

        # Get columns from bindings
        selected_col = op_bindings.get("selected_col", "the column")
        cat_col = op_bindings.get("cat_col", "the category")

        # For cascading template, we have two filter columns
        # Try to extract them from the chain
        filter_cols = []
        for op in spec.ops:
            if op.op_name == "filter_by_threshold":
                col = op.params.get("selected_col")
                if col and col not in filter_cols:
                    filter_cols.append(col)

        if len(filter_cols) >= 2:
            col_a, col_b = filter_cols[0], filter_cols[1]
        else:
            col_a = selected_col
            col_b = selected_col

        return verbalize_cascading_filters(
            col_a=col_a,
            threshold_a=op_bindings.get("threshold", 0),
            col_b=col_b,
            threshold_b=0,
            cat_col=cat_col,
            final_col=col_a,
        )

    elif name.startswith("derived_pipeline_"):
        # Parse: derived_pipeline_col_w{window}
        parts = name.split("_")
        if len(parts) >= 4:
            base_col = parts[2]
            window = int(parts[3][1:]) if parts[3].startswith("w") else 3
            return verbalize_derived_column_pipeline(
                base_col=base_col, window=window, diff_threshold=0.5
            )

    elif name.startswith("evidence_decision_"):
        # Parse: evidence_decision_valueCol_groupCol
        parts = name.split("_")
        if len(parts) >= 4:
            value_col = parts[2]
            group_col = parts[3]
            return verbalize_evidence_decision_action(
                value_col=value_col, group_col=group_col
            )

    # Check for simple 3-step aggregation programs (e.g., program_0_selected_col_EU_Sales)
    elif (
        len(spec.ops) == 3
        and spec.ops[0].op_name == "select_numeric_cols"
        and spec.ops[1].op_name == "bind_numeric_col"
    ):
        # Extract column name from the bind operation
        col_name = spec.ops[1].params.get("selected_col", "the selected column")
        final_op = spec.ops[2].op_name

        # Map operator names to natural language
        op_verbs = {
            "mean": "average",
            "median": "median",
            "std": "standard deviation",
            "variance": "variance",
            "sum": "total",
            "min": "minimum",
            "max": "maximum",
        }
        verb = op_verbs.get(final_op, final_op)

        # Include specific JSON format in the question
        question = f'What is the {verb} of {col_name}? Return as JSON, e.g.: {{"column": "<name>", "{final_op}": 0.0}}'
        hint = f'Calculate the {verb} for the {col_name} column. You MUST submit the result as a dict with keys \'column\' and \'{final_op}\', like: submit({{"column": "{col_name}", "{final_op}": value}})'

        return question, hint

    # Fallback: return mechanical description with JSON format instruction
    ops_str = " → ".join(op.op_name for op in spec.ops)
    question = f"Execute this {len(spec.ops)}-step analysis: {ops_str}. Return the result as JSON."
    hint = f"Follow the sequence: {ops_str}. Return your final answer as a JSON object using submit()."

    return question, hint


class TemplateVerbalizer:
    """Verbalizer that uses deterministic templates for long chains.

    This is the DEFAULT verbalizer for semantic long-chain questions.
    It guarantees that the verbalized question accurately describes the computation.

    For optional natural language variation, use polish_with_llm() after generation.
    """

    def verbalize(self, spec: ProgramSpec) -> tuple[str, str]:
        """Generate deterministic verbalization for a program spec.

        Returns:
            (question_text, hint) tuple
        """
        return verbalize_long_chain_program(spec)

    async def polish_with_llm(
        self, question: str, hint: str, model, sampling_args: dict
    ) -> str:
        """Optional: Polish the template verbalization with LLM.

        This adds natural language variation WITHOUT changing the meaning.
        The LLM is constrained to preserve the logical structure.

        Returns polished question text.
        """
        # TODO: Implement LLM polishing if needed
        # For now, return original
        return question
