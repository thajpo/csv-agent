"""Question I/O utilities for unified schema.

Handles loading and saving questions in the unified schema with fail-fast validation.
"""

import json
from pathlib import Path
from typing import TypedDict, Literal, Any


class QuestionRecord(TypedDict, total=False):
    id: str
    source: Literal["template", "procedural", "llm"]
    dataset: str
    question_text: str | None
    question_mechanical: str | None
    hint: str | None
    code: str | None
    code_hash: str | None
    ground_truth: Any
    ground_truth_hash: str | None
    output_schema: str | None
    n_steps: int | None
    difficulty: str | None
    dataset_description: str | None


def load_questions(path: str) -> list[QuestionRecord]:
    """Load questions from JSON file in the unified schema.

    Fail fast if the file does not match the schema (no legacy normalization).

    Works for both LLM and synthetic question files (unified schema).

    Args:
        path: Path to JSON file.

    Returns:
        List of QuestionRecord objects.

    Raises:
        ValueError: if file is not in unified schema.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    questions = []
    for q in data:
        errors = validate_question(q)
        if errors:
            raise ValueError(
                f"Invalid question {q.get('id', '<no id>')}: {'; '.join(errors)}"
            )
        questions.append(q)

    return questions


def save_questions(questions: list[QuestionRecord], path: str) -> None:
    """Save questions to JSON file in unified schema.

    Args:
        questions: List of QuestionRecord objects.
        path: Path to output JSON file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(questions, f, indent=2, default=str)


def validate_question(q: dict) -> list[str]:
    """Return list of validation errors (empty if valid).

    Validation = schema/contract checks (required fields + source-specific fields).

    Checks:
    - Required fields present
    - Source-specific fields present (mechanical/code/ground_truth for synthetic, question_text for LLM)

    Args:
        q: Question dict to validate.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Required for all
    for field in ("id", "source", "dataset"):
        if field not in q or q[field] is None:
            errors.append(f"Missing required field: {field}")

    source = q.get("source")
    if source in ("template", "procedural"):
        # Required for deterministic (template/procedural)
        for field in (
            "question_mechanical",
            "code",
            "code_hash",
            "ground_truth",
            "ground_truth_hash",
            "output_schema",
            "n_steps",
        ):
            if field not in q or q[field] is None:
                errors.append(f"Missing required field for {source}: {field}")
    elif source == "llm":
        # Required for LLM
        if "question_text" not in q:
            errors.append("Missing required field for LLM: question_text")
    else:
        errors.append(f"Invalid source: {source}")

    return errors
