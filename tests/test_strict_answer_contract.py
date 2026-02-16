import json
import tempfile

import pytest

from src.datagen.shared.questions_io import load_questions, validate_question
from src.datagen.shared.verification import verify_synthetic


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
                "final_answer_hash": "actual-hash",
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
                "final_answer_hash": "actual-hash",
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
