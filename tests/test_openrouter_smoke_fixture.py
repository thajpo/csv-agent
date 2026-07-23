from pathlib import Path

import pandas as pd
import pytest
from csv_spec import EpisodeJSONL, hash_artifact

from src.datagen.process_report import validate_trace_submissions
from src.eval.evaluator import Evaluator


SMOKE_EPISODES = "data/fixtures/openrouter-smoke.jsonl"


def test_openrouter_smoke_fixture_matches_checked_in_csv() -> None:
    episodes = Evaluator(model="unused").load_episodes(SMOKE_EPISODES)

    assert len(episodes) == 1
    episode: EpisodeJSONL = episodes[0]
    csv_path = Path(episode.csv_source)
    expected_answer = episode.question["ground_truth"]

    assert csv_path.is_file()
    assert len(pd.read_csv(csv_path)) == expected_answer
    assert hash_artifact(expected_answer) in episode.question["ground_truth_hashes"]
    validate_trace_submissions(episode.gold_trace)


@pytest.mark.asyncio
async def test_evaluator_uses_current_episode_ground_truth_contract(monkeypatch) -> None:
    class FakeState:
        submitted_answer = 2796
        current_turn = 1

    class FakeEnvironment:
        async def rollout(self):
            return FakeState()

    async def fake_from_params(**_kwargs):
        return FakeEnvironment()

    monkeypatch.setattr(
        "src.eval.evaluator.Environment.from_params", fake_from_params
    )
    episode = Evaluator(model="unused").load_episodes(SMOKE_EPISODES)[0]

    result = await Evaluator(model="unused").evaluate_episode(episode)

    assert result.execution_success is True
    assert result.final_answer_correct is True
    assert result.expected_answer == 2796


@pytest.mark.asyncio
async def test_evaluator_rejects_missing_hash_provenance_before_rollout(
    monkeypatch,
) -> None:
    async def unexpected_from_params(**_kwargs):
        raise AssertionError("rollout must not start without verifier hashes")

    monkeypatch.setattr(
        "src.eval.evaluator.Environment.from_params", unexpected_from_params
    )
    episode = Evaluator(model="unused").load_episodes(SMOKE_EPISODES)[0]
    episode.question["ground_truth_hash"] = None
    episode.question["ground_truth_hashes"] = None

    with pytest.raises(ValueError, match="hash provenance"):
        await Evaluator(model="unused").evaluate_episode(episode)
