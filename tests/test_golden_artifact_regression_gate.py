import copy
import json
import uuid
from pathlib import Path
from typing import Literal, cast

import pytest
from csv_spec import EpisodeJSONL

from src.datagen.shared.episode_factory import create_episode
from src.datagen.shared.questions_io import validate_question
from src.datagen.shared.verification import VerificationResult


SOURCES = ("template", "procedural", "llm_gen")
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden_artifacts"
NORMALIZED_ID = "<normalized-id>"
NORMALIZED_EPISODE_ID = "<normalized-episode-id>"
NORMALIZED_TIMESTAMP = "<normalized-timestamp>"


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _golden_question(source: str) -> dict:
    return _read_json(FIXTURE_DIR / "questions" / f"{source}.json")


def _golden_episode(source: str) -> dict:
    return _read_json(FIXTURE_DIR / "episodes" / f"{source}.json")


def _sample_question_artifact(source: str) -> dict:
    question = {
        "id": f"{source}-{uuid.uuid4().hex[:8]}",
        "source": source,
        "dataset": "fixture_dataset",
        "question_text": f"[{source}] What is the average value of column A?",
        "question_mechanical": (
            None
            if source == "llm_gen"
            else f"[{source}] Compute mean(A) and return scalar:float"
        ),
        "hint": "Use df['A'].mean()",
        "code": None
        if source == "llm_gen"
        else "result = df['A'].mean()\nsubmit(result)",
        "code_hash": None if source == "llm_gen" else "codehash-001",
        "ground_truth": None if source == "llm_gen" else 42.5,
        "ground_truth_hash": None if source == "llm_gen" else "answerhash-001",
        "ground_truth_hashes": None if source == "llm_gen" else ["answerhash-001"],
        "output_schema": None if source == "llm_gen" else "scalar:float",
        "n_steps": 1,
        "difficulty": "EASY",
        "dataset_description": "fixture dataset",
    }

    if source == "template":
        question["template_name"] = "mean_template"
        question["template_params"] = {"column": "A"}
        question["output_type"] = "float"
    elif source == "procedural":
        question["program_name"] = "mean_program"
        question["program_ops"] = ["select_numeric_cols", "mean"]

    return question


def _sample_trace() -> dict:
    return {
        "turns": [
            {
                "turn_index": 0,
                "reasoning": "Compute mean",
                "code": "result = df['A'].mean()\nsubmit(result)",
                "execution": {
                    "success": True,
                    "stdout": '\u2713 Submitted: {"__csv_agent_answer__": 42.5, "hooks": []}',
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 42.5,
                },
                "correction": None,
            }
        ],
        "final_answer": 42.5,
        "final_answer_hash": "answerhash-001",
        "success": True,
    }


async def _sample_episode_artifact(source: str) -> dict:
    question = _sample_question_artifact(source)
    consistency_traces = (
        [] if source != "llm_gen" else [_sample_trace(), _sample_trace()]
    )

    verification_result = VerificationResult(
        success=True,
        match=True,
        trace=_sample_trace(),
        traces=consistency_traces,
        majority_answer_hash="answerhash-001",
        error=None,
    )

    episode = await create_episode(
        question=question,
        verification_result=verification_result,
        source=cast(Literal["template", "procedural", "llm_gen"], source),
        csv_path="data/csv/fixture.csv",
    )
    return episode.model_dump(mode="json")


def _normalize_question(question: dict) -> dict:
    normalized = copy.deepcopy(question)
    normalized["id"] = NORMALIZED_ID
    return normalized


def _normalize_episode(episode: dict) -> dict:
    normalized = copy.deepcopy(episode)
    normalized["episode_id"] = NORMALIZED_EPISODE_ID
    normalized["timestamp"] = NORMALIZED_TIMESTAMP
    normalized["question"]["id"] = NORMALIZED_ID
    return normalized


def _episode_for_schema_validation(fixture: dict) -> dict:
    candidate = copy.deepcopy(fixture)
    candidate["episode_id"] = "episode-fixed-id"
    candidate["timestamp"] = "2026-01-01T00:00:00"
    candidate["question"]["id"] = "question-fixed-id"
    return candidate


def _assert_matches_golden(actual: dict, expected: dict, normalizer) -> None:
    normalized_actual = normalizer(actual)
    if normalized_actual != expected:
        raise AssertionError(
            "Golden artifact drift detected.\n"
            f"Expected:\n{json.dumps(expected, indent=2, sort_keys=True)}\n"
            f"Actual:\n{json.dumps(normalized_actual, indent=2, sort_keys=True)}"
        )


@pytest.mark.parametrize("source", SOURCES)
def test_question_fixture_is_valid_contract(source: str):
    fixture = _golden_question(source)
    errors = validate_question(fixture)
    assert errors == []


@pytest.mark.parametrize("source", SOURCES)
def test_question_contract_matches_golden(source: str):
    actual = _sample_question_artifact(source)
    expected = _golden_question(source)
    _assert_matches_golden(actual, expected, _normalize_question)


@pytest.mark.parametrize("source", SOURCES)
def test_episode_fixture_is_valid_contract(source: str):
    fixture = _episode_for_schema_validation(_golden_episode(source))
    EpisodeJSONL(**fixture)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.asyncio
async def test_episode_contract_matches_golden(source: str):
    actual = await _sample_episode_artifact(source)
    expected = _golden_episode(source)
    _assert_matches_golden(actual, expected, _normalize_episode)


def test_question_drift_failure_check():
    actual = _sample_question_artifact("template")
    actual["source"] = "llm_gen"
    expected = _golden_question("template")

    with pytest.raises(AssertionError, match="Golden artifact drift detected"):
        _assert_matches_golden(actual, expected, _normalize_question)


@pytest.mark.asyncio
async def test_episode_drift_failure_check():
    actual = await _sample_episode_artifact("template")
    actual["triangulation"]["gold_matches_majority"] = False
    expected = _golden_episode("template")

    with pytest.raises(AssertionError, match="Golden artifact drift detected"):
        _assert_matches_golden(actual, expected, _normalize_episode)
