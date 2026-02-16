"""Tests for shared filters module."""

from src.datagen.shared.filters import (
    check_dataset_viability,
    check_column_eligibility,
    check_question_viability,
    check_program_output,
)


class TestDatasetViability:
    def test_passes_with_good_dataset(self):
        profile = {
            "shape": {"rows": 100, "columns": 5},
            "columns": {
                "col1": {"type": "numeric", "missing_pct": 0, "unique_count": 50},
                "col2": {"type": "categorical", "missing_pct": 0, "unique_count": 10},
            },
        }
        result = check_dataset_viability(profile)
        assert result.passed is True
        assert result.reason is None

    def test_fails_too_few_rows(self):
        profile = {
            "shape": {"rows": 10, "columns": 5},
            "columns": {
                "col1": {"type": "numeric", "missing_pct": 0, "unique_count": 10}
            },
        }
        result = check_dataset_viability(profile)
        assert result.passed is False
        assert "too_few_rows" in result.reason

    def test_fails_too_few_columns(self):
        profile = {
            "shape": {"rows": 100, "columns": 1},
            "columns": {
                "col1": {"type": "numeric", "missing_pct": 0, "unique_count": 50}
            },
        }
        result = check_dataset_viability(profile)
        assert result.passed is False
        assert "too_few_cols" in result.reason

    def test_fails_all_heavy_missing(self):
        profile = {
            "shape": {"rows": 100, "columns": 2},
            "columns": {
                "col1": {"type": "numeric", "missing_pct": 96, "unique_count": 50},
                "col2": {"type": "numeric", "missing_pct": 97, "unique_count": 50},
            },
        }
        result = check_dataset_viability(profile)
        assert result.passed is False
        assert result.reason == "all_heavy_missing"

    def test_fails_no_eligible_columns(self):
        # Columns that are NOT heavily missing but have other issues (zero variance)
        profile = {
            "shape": {"rows": 100, "columns": 2},
            "columns": {
                "col1": {
                    "type": "numeric",
                    "missing_pct": 5,  # Not heavily missing
                    "unique_count": 1,  # But zero variance
                },
                "col2": {
                    "type": "numeric",
                    "missing_pct": 5,  # Not heavily missing
                    "unique_count": 1,  # But zero variance
                },
            },
        }
        result = check_dataset_viability(profile)
        assert result.passed is False
        assert result.reason == "no_eligible_columns"


class TestColumnEligibility:
    def test_passes_good_numeric_column(self):
        info = {
            "type": "numeric",
            "missing_pct": 5,
            "unique_count": 50,
            "is_index_like": False,
        }
        result = check_column_eligibility(info, row_count=100)
        assert result.passed is True

    def test_fails_heavy_missing(self):
        info = {"type": "numeric", "missing_pct": 96, "unique_count": 50}
        result = check_column_eligibility(info, row_count=100)
        assert result.passed is False
        assert result.reason == "heavy_missingness"

    def test_fails_zero_variance(self):
        info = {"type": "numeric", "missing_pct": 0, "unique_count": 1}
        result = check_column_eligibility(info, row_count=100)
        assert result.passed is False
        assert result.reason == "zero_variance"

    def test_fails_id_like_by_flag(self):
        info = {
            "type": "numeric",
            "missing_pct": 0,
            "unique_count": 50,
            "is_index_like": True,
        }
        result = check_column_eligibility(info, row_count=100)
        assert result.passed is False
        assert result.reason == "id_like_column"

    def test_fails_id_like_by_uniqueness(self):
        # 99% unique (above 98% threshold)
        info = {
            "type": "numeric",
            "missing_pct": 0,
            "unique_count": 99,
            "is_index_like": False,
        }
        result = check_column_eligibility(info, row_count=100)
        assert result.passed is False
        assert result.reason == "id_like_column"


class TestQuestionViability:
    def test_passes_good_question(self):
        question = "What is the average age of passengers?"
        profile = {"columns": {"age": {"type": "numeric"}}}
        result = check_question_viability(question, profile)
        assert result.passed is True

    def test_fails_forbidden_method_terms(self):
        for term in ["calculate", "group by", "pandas"]:
            question = f"{term} the average age"
            profile = {"columns": {}}
            result = check_question_viability(question, profile)
            assert result.passed is False, f"Should fail for term: {term}"
            assert result.reason == "forbidden_method_terms"

    def test_fails_mentions_column_name(self):
        question = "What is the average passenger_age?"  # Long column name (>=4 chars)
        profile = {"columns": {"passenger_age": {"type": "numeric"}}}
        result = check_question_viability(question, profile)
        assert result.passed is False
        assert result.reason == "mentions_column_names"

    def test_fails_too_many_sentences(self):
        question = "First sentence. Second sentence. Third sentence. Fourth sentence."
        profile = {"columns": {}}
        result = check_question_viability(question, profile)
        assert result.passed is False
        assert result.reason == "too_many_sentences"


class TestProgramOutput:
    def test_passes_good_output(self):
        result = check_program_output({"answer": 42, "row_count": 50})
        assert result.passed is True

    def test_fails_no_answer(self):
        result = check_program_output({"answer": None, "row_count": 50})
        assert result.passed is False
        assert result.reason == "no_answer_submitted"

    def test_fails_nan_answer(self):

        result = check_program_output({"answer": float("nan"), "row_count": 50})
        assert result.passed is False
        assert result.reason == "nan_answer"

    def test_fails_empty_answer(self):
        result = check_program_output({"answer": "", "row_count": 50})
        assert result.passed is False
        assert result.reason == "empty_answer"

    def test_fails_too_few_rows(self):
        result = check_program_output({"answer": 42, "row_count": 10})
        assert result.passed is False
        assert "too_few_rows" in result.reason
