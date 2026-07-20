"""Tests for the bounded prefix-value collection command."""

from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.experiments.collect_prefix_values as collect_module

from scripts.experiments.collect_prefix_values import (
    MAX_CANDIDATES_PER_EPISODE,
    MAX_CONTINUATIONS,
    MAX_PROVIDER_REQUESTS,
    candidate_request,
    collect,
    current_commit,
    first_action_identity,
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
        "model": "test-model",
        "output": Path("unused.jsonl"),
        "dataset_revision": "dataset-revision",
        "csv": None,
        "max_episodes": 1,
        "turn_count": 1,
        "max_turns": 10,
        "continuations": 4,
        "candidates_per_episode": 3,
        "concurrency": 8,
        "seed": 42,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 1200,
        "request_timeout_seconds": 30.0,
        "float_tolerance": 0.1,
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
                max_episodes=64,
                continuations=MAX_CONTINUATIONS,
                max_turns=10,
            )
        )


def test_candidate_cap_rejects_accidental_large_branching(
    episodes_jsonl: Path,
) -> None:
    with pytest.raises(ValueError, match="candidates-per-episode"):
        validate_args(
            _args(
                episodes_jsonl,
                candidates_per_episode=MAX_CANDIDATES_PER_EPISODE + 1,
            )
        )


def test_initial_state_rejects_multiple_identical_candidates(
    episodes_jsonl: Path,
) -> None:
    with pytest.raises(ValueError, match="initial-state collection"):
        validate_args(_args(episodes_jsonl, turn_count=0, candidates_per_episode=2))


def test_first_action_identity_uses_normalized_code_not_execution() -> None:
    first = SimpleNamespace(
        trace={
            "turns": [
                {
                    "code": "print( df.columns )",
                    "execution": {"stdout": "first result"},
                }
            ]
        }
    )
    repeated = SimpleNamespace(
        trace={
            "turns": [
                {
                    "code": "print(df.columns)",
                    "execution": {"stdout": "different result"},
                }
            ]
        }
    )
    distinct = SimpleNamespace(
        trace={"turns": [{"code": "print(df.shape)", "execution": {}}]}
    )

    assert first_action_identity(first) == first_action_identity(repeated)
    assert first_action_identity(first) != first_action_identity(distinct)


def test_candidate_request_excludes_previously_sampled_actions() -> None:
    first = candidate_request([])
    later = candidate_request(["print(df.head())", "print(df.describe())"])

    assert "column names" in first
    assert "1-3 sentences" in first
    assert "exactly one fenced ```python code block" in first
    assert "Do not call submit()" in first
    assert "print(df.head())" not in first
    assert "print(df.head())" in later
    assert "print(df.describe())" in later
    assert "substantively different operation" in later


@pytest.mark.asyncio
async def test_source_failure_consumes_retry_instead_of_aborting_collection(
    monkeypatch, episodes_jsonl
) -> None:
    attempts = 0

    async def flaky_source(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TypeError("provider returned null content")
        response = (
            f"Reasoning for candidate {attempts}.\n```python\nprint({attempts})\n```"
        )
        return SimpleNamespace(
            trace={
                "turns": [
                    {
                        "turn_index": 0,
                        "reasoning": f"Candidate {attempts}",
                        "code": f"print({attempts})",
                        "execution": {
                            "success": True,
                            "stdout": str(attempts),
                            "stderr": "",
                            "hooks": [],
                            "submitted_answer": None,
                        },
                    }
                ]
            },
            system_prompt="prompt",
            turn_responses=[response],
            turn_completed=[False],
            boundary_messages=[
                [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": "Execution completed."},
                ]
            ],
            boundary_consumed_turns=[1],
        )

    class FakeRecord:
        value = 1.0
        labeled_continuations = 1
        attempted_continuations = 1

        def model_dump(self, **_kwargs):
            return {"value": self.value}

    async def fake_collect_prefix_value(**_kwargs):
        return FakeRecord()

    monkeypatch.setattr(collect_module, "run_initial_model_trace", flaky_source)
    monkeypatch.setattr(
        collect_module, "collect_prefix_value", fake_collect_prefix_value
    )
    monkeypatch.setattr(collect_module, "current_commit", lambda: "abc123")

    args = _args(
        episodes_jsonl,
        candidates_per_episode=1,
        continuations=1,
    )
    records = await collect(args)

    assert attempts == 2
    assert len(records) == 1


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
    calls: list[list[str]] = []

    def fake_run(*_args, **_kwargs):
        calls.append(_args[0])
        return SimpleNamespace(stdout=" M src/value/collection.py\n")

    monkeypatch.setattr(
        "scripts.experiments.collect_prefix_values.subprocess.run", fake_run
    )

    with pytest.raises(RuntimeError, match="uncommitted code changes"):
        current_commit()

    assert "src" in calls[0]
    assert ".prime-rl" not in calls[0]
