"""Tests for the bounded prefix-value collection command."""

from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments.collect_prefix_values import (
    MAX_CONTINUATIONS,
    MAX_PROVIDER_REQUESTS,
    current_commit,
    load_episodes,
    maximum_provider_requests,
    select_boundary,
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


def test_boundary_selection_does_not_require_a_later_execution() -> None:
    messages = [
        {"role": "assistant", "content": "Executed one turn."},
        {"role": "user", "content": "Execution completed."},
    ]
    source = SimpleNamespace(
        trace={"turns": [{"turn_index": 0}]},
        turn_completed=[False],
        boundary_messages=[messages],
        boundary_consumed_turns=[1],
    )

    selected_messages, consumed_turns = select_boundary(source, 1, 3)

    assert selected_messages == messages
    assert consumed_turns == 1


@pytest.mark.parametrize(
    ("completed", "consumed_turns", "error"),
    [(True, 1, "terminal"), (False, 3, "remaining turn budget")],
)
def test_boundary_selection_requires_nonterminal_budget(
    completed: bool, consumed_turns: int, error: str
) -> None:
    source = SimpleNamespace(
        trace={"turns": [{"turn_index": 0}]},
        turn_completed=[completed],
        boundary_messages=[[{"role": "assistant", "content": "response"}]],
        boundary_consumed_turns=[consumed_turns],
    )

    with pytest.raises(ValueError, match=error):
        select_boundary(source, 1, 3)


def test_current_commit_rejects_dirty_worktree(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout=" M src/value/collection.py\n")

    monkeypatch.setattr(
        "scripts.experiments.collect_prefix_values.subprocess.run", fake_run
    )

    with pytest.raises(RuntimeError, match="uncommitted code changes"):
        current_commit()
