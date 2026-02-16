"""Tests for program generator unified schema output."""

import pytest
import pytest_asyncio
import json
from pathlib import Path
import tempfile

from src.datagen.synthetic.programs.program_generator import (
    run_pipeline,
    _generate_mechanical_description,
)
from src.datagen.shared.questions_io import load_questions, validate_question


class TestMechanicalDescription:
    def test_basic_ops(self):
        ops = ["select_numeric_cols", "bind_numeric_col", "mean"]
        desc = _generate_mechanical_description(ops)
        assert "select" in desc.lower()
        assert "mean" in desc.lower()

    def test_empty_ops(self):
        desc = _generate_mechanical_description([])
        assert "analyze" in desc.lower()

    def test_unknown_ops(self):
        ops = ["unknown_op1", "unknown_op2"]
        desc = _generate_mechanical_description(ops)
        assert "execute program" in desc.lower()


@pytest_asyncio.fixture
async def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_program_generator_outputs_unified_schema(temp_output_dir):
    """Test that program generator outputs unified schema questions."""
    # Use a test CSV
    csv_path = "data/csv/data.csv"

    # Skip if test data doesn't exist
    if not Path(csv_path).exists():
        pytest.skip(f"Test CSV not found: {csv_path}")

    questions = await run_pipeline(
        csv_path=csv_path,
        max_programs=2,  # Small number for test
        max_verbalize=2,
        skip_verbalization=True,  # Skip LLM for test speed
        output_dir=str(temp_output_dir),
    )

    # Verify we got questions
    assert len(questions) > 0, "Should generate at least one question"

    # Verify unified schema fields
    for q in questions:
        assert q["id"].startswith("prog_"), f"ID should start with prog_: {q['id']}"
        assert q["source"] == "procedural"
        assert "dataset" in q
        assert "question_mechanical" in q
        assert "question_text" in q
        assert "hint" in q
        assert "code" in q
        assert "code_hash" in q
        assert "ground_truth" in q
        assert "ground_truth_hash" in q
        assert "output_schema" in q
        assert "n_steps" in q
        assert "difficulty" in q
        assert "dataset_description" in q
        assert "program_name" in q
        assert "program_ops" in q


@pytest.mark.asyncio
async def test_program_questions_pass_validation(temp_output_dir):
    """Test that generated questions pass schema validation."""
    csv_path = "data/csv/data.csv"

    if not Path(csv_path).exists():
        pytest.skip(f"Test CSV not found: {csv_path}")

    questions = await run_pipeline(
        csv_path=csv_path,
        max_programs=1,
        max_verbalize=1,
        skip_verbalization=True,
        output_dir=str(temp_output_dir),
    )

    # Save and reload to test load_questions
    questions_file = temp_output_dir / "test_questions.json"
    with open(questions_file, "w") as f:
        json.dump(questions, f)

    # Load with shared loader
    loaded = load_questions(str(questions_file))
    assert len(loaded) == len(questions)

    # Validate each question
    for q in loaded:
        errors = validate_question(q)
        assert len(errors) == 0, f"Validation errors: {errors}"


@pytest.mark.asyncio
async def test_program_questions_saved_to_correct_location(temp_output_dir):
    """Test that questions are saved to data/questions_synthetic/{dataset}/questions.json"""
    csv_path = "data/csv/data.csv"

    if not Path(csv_path).exists():
        pytest.skip(f"Test CSV not found: {csv_path}")

    await run_pipeline(
        csv_path=csv_path,
        max_programs=1,
        max_verbalize=1,
        skip_verbalization=True,
        output_dir=str(temp_output_dir),
    )

    # Check file was created
    expected_file = temp_output_dir / "questions.json"
    assert expected_file.exists(), f"Questions file not found: {expected_file}"

    # Verify it's valid JSON
    with open(expected_file) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) > 0
