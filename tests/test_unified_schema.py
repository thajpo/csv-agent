"""Smoke tests for unified schema compliance.

These tests verify that all question sources output valid unified schema.
"""

import pytest
import pytest_asyncio
import json
import tempfile
from pathlib import Path

from src.datagen.shared.questions_io import load_questions, validate_question
from src.datagen.shared.verification import verify_synthetic
from src.datagen.synthetic.programs.program_generator import (
    run_pipeline as run_program_pipeline,
    _generate_mechanical_description,
)


class TestTemplateUnifiedSchema:
    """Test 1: Generate template question, load via unified schema."""

    @pytest.mark.asyncio
    async def test_template_question_schema(self):
        """Generate a template question and verify unified schema compliance."""
        # Use synthetic generator to create template questions
        from src.datagen.synthetic.generator import CompositionalQuestionGenerator
        from src.core.config import config

        csv_path = "data/csv/data.csv"
        if not Path(csv_path).exists():
            pytest.skip(f"Test CSV not found: {csv_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            generator = CompositionalQuestionGenerator(
                csv_path=csv_path,
                model=config.question_gen_model,
            )
            await generator.setup()

            result = await generator.generate(
                n_questions=1,
                output_path=output_path,
            )

            await generator.cleanup()

            # Check questions were generated
            assert len(result["questions"]) > 0, "Should generate at least one question"

            # Save and reload to test load_questions
            questions_file = output_path / "questions.json"
            with open(questions_file, "w") as f:
                json.dump(result["questions"], f)

            loaded = load_questions(str(questions_file))
            assert len(loaded) > 0

            # Validate each question
            for q in loaded:
                errors = validate_question(q)
                assert len(errors) == 0, f"Validation errors: {errors}"

                # Check required fields
                assert q["source"] == "template"
                assert "question_mechanical" in q
                assert "code" in q
                assert "ground_truth" in q


class TestProgramUnifiedSchema:
    """Test 2: Generate program question, load via unified schema."""

    @pytest.mark.asyncio
    async def test_program_question_schema(self):
        """Generate a program question and verify unified schema compliance."""
        csv_path = "data/csv/data.csv"
        if not Path(csv_path).exists():
            pytest.skip(f"Test CSV not found: {csv_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            questions = await run_program_pipeline(
                csv_path=csv_path,
                max_programs=1,
                max_verbalize=1,
                skip_verbalization=True,  # Skip LLM for speed
                output_dir=str(output_path),
            )

            assert len(questions) > 0, "Should generate at least one program question"

            # Verify unified schema
            for q in questions:
                errors = validate_question(q)
                assert len(errors) == 0, f"Validation errors: {errors}"

                assert q["source"] == "procedural"
                assert "program_name" in q
                assert "program_ops" in q


class TestSyntheticVerification:
    """Test 3: Run synthetic verification trace."""

    @pytest.mark.asyncio
    async def test_verify_synthetic_returns_result(self):
        """Run verify_synthetic and check result structure."""
        csv_path = "data/csv/data.csv"
        if not Path(csv_path).exists():
            pytest.skip(f"Test CSV not found: {csv_path}")

        # Create a simple synthetic question
        question = {
            "id": "test_question",
            "source": "template",
            "dataset": "test",
            "question_mechanical": "What is the mean of column A?",
            "question_text": "What is the average value?",
            "code": "df['A'].mean()",
            "code_hash": "abc123",
            "ground_truth": 42.0,
            "ground_truth_hash": "hash123",
            "output_schema": "scalar:float",
            "n_steps": 1,
        }

        # Run verification
        result = await verify_synthetic(
            question=question,
            csv_path=csv_path,
        )

        # Check result structure
        assert hasattr(result, "success")
        assert hasattr(result, "match")
        assert hasattr(result, "trace")
        assert hasattr(result, "error")

        # Type checks
        assert isinstance(result.success, bool)


class TestLLMQuestionLoad:
    """Test 4: LLM question load still works with unified schema."""

    def test_llm_question_fixture(self):
        """Load LLM questions from fixture and verify fields."""
        # Create a mock LLM question
        llm_question = {
            "id": "llm_test_001",
            "source": "llm",
            "dataset": "test_dataset",
            "question_text": "What is the correlation between age and income?",
            "hint": "Consider using correlation analysis",
            "difficulty": "medium",
            "n_steps": 2,
        }

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([llm_question], f)
            temp_path = f.name

        try:
            # Load with shared loader
            loaded = load_questions(temp_path)
            assert len(loaded) == 1

            q = loaded[0]
            assert q["source"] == "llm"
            
            assert "question_text" in q
            assert (
                q["question_text"] == "What is the correlation between age and income?"
            )

            # Validate
            errors = validate_question(q)
            assert len(errors) == 0, f"Validation errors: {errors}"
        finally:
            Path(temp_path).unlink()


class TestMechanicalDescription:
    """Test mechanical description generation."""

    def test_basic_ops(self):
        """Test mechanical description for common ops."""
        ops = ["select_numeric_cols", "bind_numeric_col", "mean"]
        desc = _generate_mechanical_description(ops)
        assert "select" in desc.lower() or "program" in desc.lower()

    def test_empty_ops(self):
        """Test mechanical description for empty ops."""
        desc = _generate_mechanical_description([])
        assert "analyze" in desc.lower()

    def test_unknown_ops(self):
        """Test mechanical description for unknown ops."""
        ops = ["unknown_op1", "unknown_op2"]
        desc = _generate_mechanical_description(ops)
        assert "execute program" in desc.lower() or "program" in desc.lower()
