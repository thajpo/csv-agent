from pathlib import Path

import pandas as pd
import pytest
from csv_spec import EpisodeJSONL, hash_artifact

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
