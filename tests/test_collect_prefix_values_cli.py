"""Tests for the bounded prefix-value collection command."""

from argparse import Namespace
import json
from pathlib import Path

import pytest

from scripts.experiments.collect_prefix_values import (
    MAX_CONTINUATIONS,
    MAX_PROVIDER_REQUESTS,
    load_episodes,
    maximum_provider_requests,
    validate_args,
)


@pytest.fixture
def episodes_jsonl(tmp_path: Path) -> Path:
    episode = json.loads(Path("tests/fixtures/expected_episode.json").read_text())
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(episode) + "\n")
    return path


def _args(episodes: Path, **overrides) -> Namespace:
    values = {
        "episodes": episodes,
        "csv": None,
        "max_episodes": 1,
        "turn_count": 1,
        "max_turns": 10,
        "continuations": 4,
    }
    values.update(overrides)
    return Namespace(**values)


def test_defaults_are_bounded_and_fixture_loads(episodes_jsonl: Path) -> None:
    args = _args(episodes_jsonl)

    validate_args(args)
    episodes = load_episodes(args.episodes, args.max_episodes)

    assert len(episodes) == 1
    assert maximum_provider_requests(args) <= MAX_PROVIDER_REQUESTS


def test_continuation_cap_rejects_accidental_large_run(
    episodes_jsonl: Path,
) -> None:
    with pytest.raises(ValueError, match="continuations"):
        validate_args(_args(episodes_jsonl, continuations=MAX_CONTINUATIONS + 1))


def test_combined_request_budget_rejects_large_run(episodes_jsonl: Path) -> None:
    with pytest.raises(ValueError, match="provider requests"):
        validate_args(
            _args(
                episodes_jsonl,
                max_episodes=2,
                continuations=MAX_CONTINUATIONS,
                max_turns=10,
            )
        )
