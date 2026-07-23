import json
import tempfile

import pytest

from csv_spec import hash_artifact
from src.datagen.shared.questions_io import load_questions, validate_question
from src.datagen.shared.verification import verify_synthetic
from src.datagen.shared.submission import validate_submission_position


def _synthetic_question(**overrides):
    question = {
        "id": "q1",
        "source": "template",
        "dataset": "test",
        "question_mechanical": "Return 1",
        "code": "submit(1)",
        "code_hash": "abc",
        "ground_truth": 1,
        "ground_truth_hash": "expected-hash",
        "output_schema": "scalar:int",
        "n_steps": 1,
    }
    question.update(overrides)
    return question


def test_submit_must_be_the_final_top_level_operation() -> None:
    validate_submission_position("result = 1\nsubmit(result)")

    with pytest.raises(ValueError, match="final top-level"):
        validate_submission_position("submit(1)\nprint('after')")

    with pytest.raises(ValueError, match="at most one"):
        validate_submission_position("submit(1)\nsubmit(2)")


def test_submit_cannot_be_called_through_an_alias() -> None:
    with pytest.raises(ValueError, match="direct call"):
        validate_submission_position(
            "finish = submit\nfinish(1)\nprint('after')",
            require_submission=True,
        )


@pytest.mark.parametrize("require_submission", [False, True])
def test_submission_validation_rejects_invalid_python(require_submission) -> None:
    with pytest.raises(ValueError, match="valid Python"):
        validate_submission_position("submit(1", require_submission=require_submission)


@pytest.mark.parametrize("legacy_key", ["_ground_truth", "_ground_truths"])
def test_validate_question_rejects_legacy_answer_keys(legacy_key):
    question = _synthetic_question(**{legacy_key: 1})
    errors = validate_question(question)
    assert any(legacy_key in err for err in errors)


def test_load_questions_rejects_legacy_answer_keys():
    question = _synthetic_question(_ground_truth=1)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([question], f)
        path = f.name

    with pytest.raises(ValueError, match="_ground_truth"):
        load_questions(path)


@pytest.mark.asyncio
async def test_verify_synthetic_does_not_fallback_to_legacy_answer_key(monkeypatch):
    import src.datagen.teacher as teacher

    async def fake_execute_teacher_trace(**kwargs):
        return (
            {
                "success": True,
                "final_answer_hash": hash_artifact(1),
                "final_answer": 1,
            },
            [],
            "",
            0.01,
        )

    monkeypatch.setattr(teacher, "execute_teacher_trace", fake_execute_teacher_trace)
    monkeypatch.setattr(teacher, "answers_match", lambda *args, **kwargs: True)

    question = _synthetic_question(
        ground_truth=None,
        _ground_truth=1,
        ground_truth_hash="expected-hash",
    )

    result = await verify_synthetic(question=question, csv_path="fake.csv")
    assert result.success is False
    assert result.match is False


@pytest.mark.asyncio
async def test_verify_synthetic_uses_unified_ground_truth_key(monkeypatch):
    import src.datagen.teacher as teacher

    async def fake_execute_teacher_trace(**kwargs):
        return (
            {
                "success": True,
                "final_answer_hash": hash_artifact(1),
                "final_answer": 1,
            },
            [],
            "",
            0.01,
        )

    monkeypatch.setattr(teacher, "execute_teacher_trace", fake_execute_teacher_trace)
    monkeypatch.setattr(teacher, "answers_match", lambda *args, **kwargs: True)

    question = _synthetic_question(ground_truth=1, ground_truth_hash="expected-hash")
    result = await verify_synthetic(question=question, csv_path="fake.csv")

    assert result.success is True
    assert result.match is True
