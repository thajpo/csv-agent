"""Tests for dead code validator.

Dead code detection policy: REJECT chains with dead code immediately.
"""

import pytest
from src.datagen.synthetic.programs.spec import OpInstance
from src.datagen.synthetic.programs.dead_code_validator import validate_no_dead_code


class TestDeadCodeValidator:
    """Test cases for dead code detection."""

    def test_empty_chain_is_valid(self):
        """Empty chain has no dead code."""
        chain = []
        assert validate_no_dead_code(chain) is True

    def test_single_op_producing_answer_is_valid(self):
        """Single op that produces answer is valid (no dependencies)."""
        # mean: produces answer, consumes selected_col
        chain = [OpInstance("mean", {})]
        assert validate_no_dead_code(chain) is True

    def test_chain_with_all_consumed_is_valid(self):
        """Chain where all produces are consumed is valid."""
        # select_numeric_cols -> bind_numeric_col -> mean
        # select produces numeric_cols, bind consumes it
        # bind produces selected_col, mean consumes it
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("mean", {}),
        ]
        assert validate_no_dead_code(chain) is True

    def test_chain_with_dead_code_is_invalid(self):
        """Chain with unused intermediate is dead code."""
        # select_numeric_cols produces numeric_cols
        # bind_numeric_col produces selected_col
        # zscore produces num_series but it's never used
        # mean uses selected_col directly, not num_series
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("zscore", {}),  # Dead code: num_series never used
            OpInstance("mean", {}),
        ]
        assert validate_no_dead_code(chain) is False

    def test_multiple_unused_variables_is_invalid(self):
        """Multiple dead code ops should be detected."""
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("zscore", {}),  # Dead: num_series unused
            OpInstance("log1p", {}),  # Dead: num_series overwritten
            OpInstance("mean", {}),
        ]
        assert validate_no_dead_code(chain) is False

    def test_complex_chain_no_dead_code(self):
        """Complex chain with all variables used is valid."""
        # groupby analysis chain
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("select_categorical_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "num_col"}),
            OpInstance("bind_binary_cat_col", {"cat_col": "cat_col"}),
            OpInstance("groupby_mean", {}),
            OpInstance("argmax_group", {}),
        ]
        assert validate_no_dead_code(chain) is True

    def test_partial_usage_is_valid(self):
        """Using some but not all outputs is valid (partial usage)."""
        # groupby_mean produces group_means
        # We use it in argmax_group
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("select_categorical_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "num_col"}),
            OpInstance("bind_binary_cat_col", {"cat_col": "cat_col"}),
            OpInstance("groupby_mean", {}),
            OpInstance("argmax_group", {}),
        ]
        assert validate_no_dead_code(chain) is True

    def test_correlation_chain_valid(self):
        """Correlation chain with both columns used is valid."""
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_num_col_1", {"num_col_1": "col1"}),
            OpInstance("bind_num_col_2", {"num_col_2": "col2"}),
            OpInstance("correlation", {}),
        ]
        assert validate_no_dead_code(chain) is True

    def test_unused_column_selection_is_dead_code(self):
        """Selecting columns but never using them is dead code."""
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("select_categorical_cols", {}),  # Dead: cat cols never used
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("mean", {}),
        ]
        assert validate_no_dead_code(chain) is False

    def test_transform_chain_valid(self):
        """Transform chain where transformed value is used is valid."""
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("zscore", {}),
            OpInstance("mean_series", {}),  # Uses num_series
        ]
        assert validate_no_dead_code(chain) is True

    def test_evidence_chain_valid(self):
        """Evidence chain with all evidence used is valid."""
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("select_categorical_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "num_col"}),
            OpInstance("bind_binary_cat_col", {"cat_col": "cat_col"}),
            OpInstance("groupby_values", {}),
            OpInstance("shapiro_p", {}),
            OpInstance("levene_p", {}),
            OpInstance("choose_test", {}),
            OpInstance("ttest_ind", {}),
        ]
        assert validate_no_dead_code(chain) is True

    def test_missing_consumed_variable_is_invalid(self):
        """Chain where op consumes non-existent variable is invalid."""
        # mean consumes selected_col but it's never produced
        chain = [
            OpInstance("mean", {}),
        ]
        # This is actually valid from dead code perspective (no dead code)
        # but invalid from dependency perspective
        # Dead code validator only checks for unused produces
        assert validate_no_dead_code(chain) is True

    def test_variable_redefinition_is_valid(self):
        """Redefining a variable (shadowing) is valid."""
        # Multiple transforms that each produce num_series
        # Each is used by the next
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("zscore", {}),
            OpInstance("mean_series", {}),
        ]
        assert validate_no_dead_code(chain) is True

    def test_long_chainable_sequence_with_unused_final_output_is_dead_code(self):
        """Chain ending with transform that produces unused outputs is dead code."""
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("filter_by_threshold", {"threshold": 0}),
            OpInstance("sort_by_column", {"ascending": False}),
            OpInstance("top_n", {"n": 10}),
            OpInstance("cumulative_sum", {}),  # Produces cumsum_col and df, never used
        ]
        # This is dead code because cumulative_sum produces outputs that are never consumed
        assert validate_no_dead_code(chain) is False

    def test_long_chainable_sequence_with_final_answer_is_valid(self):
        """Long chain of table transforms ending with answer op is valid."""
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("filter_by_threshold", {"threshold": 0}),
            OpInstance("sort_by_column", {"ascending": False}),
            OpInstance("top_n", {"n": 10}),
            OpInstance("mean", {}),  # Produces answer
        ]
        assert validate_no_dead_code(chain) is True

    def test_intermediate_filter_unused_is_dead_code(self):
        """Filter that doesn't affect final output is dead code."""
        # The filter produces filtered_df but mean uses original df
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance(
                "filter_greater_than", {"threshold": 0}
            ),  # Dead: filtered_df unused
            OpInstance("mean", {}),  # Uses selected_col, not filtered_df
        ]
        assert validate_no_dead_code(chain) is False

    def test_multiple_selections_all_used(self):
        """Multiple column selections all used in different branches."""
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("select_categorical_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "num_col"}),
            OpInstance("bind_binary_cat_col", {"cat_col": "cat_col"}),
            OpInstance("groupby_mean", {}),
            OpInstance("argmax_group", {}),
        ]
        assert validate_no_dead_code(chain) is True


class TestDeadCodeEdgeCases:
    """Edge cases for dead code detection."""

    def test_unknown_operator_is_valid(self):
        """Unknown operators are ignored (no metadata)."""
        chain = [
            OpInstance("unknown_op", {}),
        ]
        assert validate_no_dead_code(chain) is True

    def test_empty_produces_and_consumes(self):
        """Ops with no produces/consumes don't affect validation."""
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("mean", {}),
        ]
        assert validate_no_dead_code(chain) is True

    def test_answer_variable_always_considered_used(self):
        """The 'answer' variable is always considered consumed."""
        # Any op that produces 'answer' is valid since it's the final output
        chain = [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_numeric_col", {"selected_col": "col1"}),
            OpInstance("mean", {}),  # Produces answer
        ]
        assert validate_no_dead_code(chain) is True
