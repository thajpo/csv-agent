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
    action_identity,
    build_collection_contract,
    candidate_request,
    collect,
    current_commit,
    first_action_identity,
    load_episodes,
    maximum_provider_requests,
    select_boundary,
    validate_resume_records,
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


def _contract(**overrides):
    policy = collect_module.ContinuationPolicy(
        model="test-model",
        sampling_args={"temperature": 0.9},
        request_timeout_seconds=30.0,
    )
    values = {
        "code_commit": "abc123",
        "dataset_revision": "revision",
        "policy": policy,
        "episode_inputs_hash": "inputs-hash",
        "turn_count": 1,
        "max_turns": 3,
        "continuations": 1,
        "candidates_per_episode": 3,
        "seed": 42,
        "float_tolerance": 0.1,
    }
    values.update(overrides)
    return collect_module.PrefixValueCollectionContract(**values)


def _resume_record(contract, *, labeled: bool) -> dict:
    response = "Inspect the shape.\n```python\nprint(df.shape)\n```"
    turns = [
        {
            "turn_index": 0,
            "reasoning": "Inspect the shape.",
            "code": "print(df.shape)",
            "execution": {
                "success": True,
                "stdout": "(3, 2)",
                "stderr": "",
                "hooks": [],
                "submitted_answer": None,
            },
        }
    ]
    turn_responses = [response]
    turn_completed = [False]
    conversation_messages = [
        {"role": "assistant", "content": response},
        {"role": "user", "content": "Execution completed."},
    ]
    consumed_turns = 1
    if contract.turn_count == 0:
        turns = []
        turn_responses = []
        turn_completed = []
        conversation_messages = []
        consumed_turns = 0
    continuation = {"rollout_index": 0, "seed": 43}
    if labeled:
        continuation.update(
            {
                "trace": {
                    "turns": [],
                    "final_answer": None,
                    "final_answer_hash": None,
                    "success": False,
                },
                "verifier_verdict": False,
            }
        )
    else:
        continuation["error"] = "TimeoutError: transient failure"
    return collect_module.PrefixValueRecord(
        prefix={
            "prefix_id": "episode-1:1:hash",
            "episode_id": "episode-1",
            "csv_source": "dataset/data.csv",
            "system_prompt": "prompt",
            "question_text": "question",
            "turns": turns,
            "turn_responses": turn_responses,
            "turn_completed": turn_completed,
            "conversation_messages": conversation_messages,
            "consumed_turns": consumed_turns,
            "max_turns": contract.max_turns,
        },
        policy=contract.policy,
        continuations=[continuation],
        attempted_continuations=1,
        labeled_continuations=int(labeled),
        successful_continuations=0,
        value=0.0 if labeled else None,
        code_commit=contract.code_commit,
        dataset_revision=contract.dataset_revision,
        collection_contract=contract,
    ).model_dump(mode="json")


def _initial_source(index: int) -> SimpleNamespace:
    response = f"Reasoning for candidate {index}.\n```python\nprint({index})\n```"
    return SimpleNamespace(
        trace={
            "turns": [
                {
                    "turn_index": 0,
                    "reasoning": f"Candidate {index}",
                    "code": f"print({index})",
                    "execution": {
                        "success": True,
                        "stdout": str(index),
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


def test_action_identity_normalizes_local_variable_names() -> None:
    first = "missing = df.isna().mean() * 100\nprint(missing[missing > 0])"
    renamed = "missing_pct = df.isna().mean() * 100\nprint(missing_pct[missing_pct > 0])"
    distinct = "missing_pct = df.isna().sum()\nprint(missing_pct[missing_pct > 0])"

    assert action_identity(first) == action_identity(renamed)
    assert action_identity(first) != action_identity(distinct)


def test_candidate_request_assigns_distinct_proposal_roles() -> None:
    first = candidate_request(0)
    second = candidate_request(1)
    third = candidate_request(2)

    assert "column names" in first
    assert "1-3 sentences" in first
    assert "exactly one fenced ```python code block" in first
    assert "Do not call submit()" in first
    assert "prerequisite selection" in first
    assert "necessary calculation" in second
    assert "direct partial result" in third
    assert "print(df.head())" not in third


def test_resume_accepts_an_incomplete_record_for_recollection() -> None:
    contract = _contract()

    records = validate_resume_records(
        [_resume_record(contract, labeled=False)],
        episode_ids={"episode-1"},
        contract=contract,
    )

    assert records[0].value is None


def test_resume_accepts_a_completed_initial_state_record() -> None:
    contract = _contract(turn_count=0, candidates_per_episode=1)

    records = validate_resume_records(
        [_resume_record(contract, labeled=True)],
        episode_ids={"episode-1"},
        contract=contract,
    )

    assert records[0].value == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("code_commit", "different-commit"),
        ("dataset_revision", "different-revision"),
        (
            "policy",
            collect_module.ContinuationPolicy(
                model="different-model",
                sampling_args={"temperature": 0.9},
                request_timeout_seconds=30.0,
            ),
        ),
        ("episode_inputs_hash", "different-inputs"),
        ("turn_count", 0),
        ("max_turns", 4),
        ("continuations", 2),
        ("candidates_per_episode", 4),
        ("seed", 99),
        ("float_tolerance", 0.2),
    ],
)
def test_resume_requires_an_exact_collection_contract(field, value) -> None:
    original = _contract()
    changed = original.model_copy(update={field: value})

    with pytest.raises(ValueError, match="collection contract does not match"):
        validate_resume_records(
            [_resume_record(original, labeled=True)],
            episode_ids={"episode-1"},
            contract=changed,
        )


def test_collection_contract_hashes_csv_contents(
    episodes_jsonl: Path, tmp_path: Path
) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("value\n1\n")
    args = _args(episodes_jsonl, csv=csv_path)
    episodes = load_episodes(episodes_jsonl, 1)
    policy = _contract().policy

    first = build_collection_contract(args, episodes, policy, "abc123")
    csv_path.write_text("value\n2\n")
    second = build_collection_contract(args, episodes, policy, "abc123")

    assert first.episode_inputs_hash != second.episode_inputs_hash


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
        return _initial_source(attempts)

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
        csv=episodes_jsonl,
    )
    records = await collect(args)

    assert attempts == 2
    assert len(records) == 1


@pytest.mark.asyncio
async def test_incomplete_candidate_is_persisted_and_retried(
    monkeypatch, episodes_jsonl
) -> None:
    source_attempts = 0
    collection_attempts = 0
    persisted: list[dict] = []

    async def fake_source(**_kwargs):
        nonlocal source_attempts
        source_attempts += 1
        return _initial_source(source_attempts)

    class FakeRecord:
        attempted_continuations = 1
        successful_continuations = 0

        def __init__(self, labeled: bool):
            self.labeled_continuations = int(labeled)
            self.value = 0.0 if labeled else None

        def model_dump(self, **_kwargs):
            return {
                "attempted_continuations": self.attempted_continuations,
                "labeled_continuations": self.labeled_continuations,
                "value": self.value,
            }

    async def fake_collect_prefix_value(**_kwargs):
        nonlocal collection_attempts
        collection_attempts += 1
        return FakeRecord(labeled=collection_attempts > 1)

    monkeypatch.setattr(collect_module, "run_initial_model_trace", fake_source)
    monkeypatch.setattr(
        collect_module, "collect_prefix_value", fake_collect_prefix_value
    )
    monkeypatch.setattr(collect_module, "current_commit", lambda: "abc123")

    records = await collect(
        _args(
            episodes_jsonl,
            candidates_per_episode=1,
            continuations=1,
            csv=episodes_jsonl,
        ),
        record_sink=persisted.append,
    )

    assert collection_attempts == 2
    assert [record["value"] for record in persisted] == [None, 0.0]
    assert [record["value"] for record in records] == [0.0]


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
