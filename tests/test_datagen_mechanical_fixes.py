from types import SimpleNamespace

import pytest

from csv_spec import TriangulationResult, hash_artifact
from src.datagen.shared.questions_io import question_prompt_text
from src.datagen.teacher import batch_triangulate, build_trace_dict


class _BaseUI:
    def print_status(self, *_args, **_kwargs):
        pass


class _EpisodeUI:
    base = _BaseUI()

    def acquire_focus(self, _focus_id):
        return True

    def release_focus(self):
        pass

    def print_question_header(self, *_args, **_kwargs):
        pass

    def print_progress_summary(self, *_args, **_kwargs):
        pass


def test_question_prompt_text_uses_unified_question_text():
    question = {
        "id": "q1",
        "question_text": "Natural question",
        "question_mechanical": "Mechanical question",
    }

    assert question_prompt_text(question) == "Natural question"
    assert question_prompt_text(question, prefer_mechanical=True) == "Mechanical question"


def test_build_trace_dict_hashes_falsy_final_answer():
    final_state = SimpleNamespace(
        submitted_answer=0,
        execution_results_per_turn=[],
    )

    trace = build_trace_dict(final_state, conversation_messages=[])

    assert trace["success"] is True
    assert trace["final_answer"] == 0
    assert trace["final_answer_hash"] == hash_artifact(0)


@pytest.mark.asyncio
async def test_batch_triangulate_accepts_question_text_only(monkeypatch):
    seen = {}

    async def fake_triangulate_teacher(**kwargs):
        seen.update(kwargs)
        return TriangulationResult(
            gold_trace={"turns": [], "final_answer": 1, "final_answer_hash": "h", "success": True},
            gold_conversation=[],
            system_prompt="",
            consistency_results=[],
            verified=True,
            timing_metadata={},
            majority_answer_hash="h",
            majority_count=1,
        )

    monkeypatch.setattr(
        "src.datagen.teacher.triangulate_teacher", fake_triangulate_teacher
    )

    results = await batch_triangulate(
        csv_path="data.csv",
        questions=[
            {
                "id": "q1",
                "question_text": "Use the unified field",
                "hint": "",
                "n_steps": 1,
                "difficulty": "EASY",
            }
        ],
        model="fake-model",
        n_consistency=2,
        n_question_slots=1,
        use_container_pool=False,
        ui=_EpisodeUI(),
    )

    assert seen["question"] == "Use the unified field"
    assert results[0].question["id"] == "q1"
