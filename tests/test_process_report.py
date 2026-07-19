"""Tests for diagnostic process reports."""

import contextlib
import io
import json
from types import SimpleNamespace

import pytest

from csv_spec import hash_artifact
from src.core.conversation import ConversationHistory
from src.core.environment import Environment, _parse_hook_records
from src.datagen.process_report import build_process_report
from src.datagen.teacher import build_trace_dict
from src.envs.csv_env import get_setup_code
from src.utils.parsing import extract_python_cells


def _submission_stdout(answer) -> str:
    return "✓ Submitted: " + json.dumps({"__csv_agent_answer__": answer})


def _trace(*, hooks: list[dict], answer=7, success=True) -> dict:
    final_answer = answer if success else None
    return {
        "turns": [
            {
                "turn_index": 0,
                "reasoning": "Solve",
                "code": (
                    "filtered = df[df['x'] > 1]\n"
                    "hook(filtered, \"filtered = df[df['x'] > 1]\", name='filtered')\n"
                    "total = filtered['y'].sum()\n"
                    "submit(total)"
                ),
                "execution": {
                    "success": success,
                    "stdout": _submission_stdout(final_answer) if success else "",
                    "stderr": "",
                    "hooks": hooks,
                    "submitted_answer": final_answer,
                },
            }
        ],
        "final_answer": final_answer,
        "final_answer_hash": hash_artifact(final_answer)
        if final_answer is not None
        else None,
        "success": success,
    }


def _hook(name: str, value_hash: str, **overrides) -> dict:
    hook = {
        "variable_name": name,
        "code_line": "filtered = df[df['x'] > 1]",
        "value": {"rows": 3},
        "value_hash": value_hash,
        "description": None,
        "depends_on": [],
        "event_line": 2,
        "event_provenance_reason": None,
    }
    hook.update(overrides)
    return hook


def test_template_hook_is_heuristic_and_submit_is_verified():
    report = build_process_report(
        source="template",
        gold_trace=_trace(hooks=[_hook("filtered", "hash-filtered")]),
        consistency_traces=[],
        verifier_verdict=True,
    )

    assert report["summary"] == {
        "total_steps": 2,
        "labeled_steps": 2,
        "verified_steps": 1,
        "heuristic_steps": 1,
        "unlabeled_steps": 0,
        "positive_steps": 2,
        "negative_steps": 0,
    }
    hook_step, submit_step = report["steps"]
    assert hook_step["label"] == 1.0
    assert hook_step["label_kind"] == "heuristic"
    assert submit_step["label"] == 1.0
    assert submit_step["label_kind"] == "verified"
    assert submit_step["label_source"] == "terminal_verifier"


def test_bad_hook_evidence_is_a_negative_heuristic_not_verified_truth():
    report = build_process_report(
        source="template",
        gold_trace=_trace(
            hooks=[
                _hook(
                    "filtered",
                    "hash-filtered",
                    code_line="not actually executed",
                    depends_on=["missing_step"],
                )
            ]
        ),
        consistency_traces=[],
        verifier_verdict=True,
    )

    hook_step = report["steps"][0]
    assert hook_step["label"] == 0.0
    assert hook_step["label_kind"] == "heuristic"
    assert hook_step["evidence"]["code_line_grounded"] is False
    assert hook_step["evidence"]["dependency_valid"] is False
    assert hook_step["evidence"]["reasons"] == [
        "ungrounded_code_line",
        "invalid_dependency",
    ]


def test_llm_consensus_never_becomes_a_verified_process_label():
    gold = _trace(hooks=[_hook("filtered", "hash-filtered")])
    consistency = [
        _trace(hooks=[_hook("filtered_alt", "hash-filtered")]),
        _trace(hooks=[_hook("filtered_again", "hash-filtered")]),
        _trace(hooks=[_hook("other", "hash-other")]),
    ]

    report = build_process_report(
        source="llm_gen",
        gold_trace=gold,
        consistency_traces=consistency,
        verifier_verdict=True,
    )

    hook_step, submit_step = report["steps"]
    assert hook_step["label"] == 1.0
    assert hook_step["label_kind"] == "heuristic"
    assert hook_step["evidence"]["consensus_matches"] == 2
    assert hook_step["evidence"]["consensus_total"] == 3
    assert submit_step["label_kind"] == "verified"


def test_llm_hook_without_consensus_remains_unlabeled():
    report = build_process_report(
        source="llm_gen",
        gold_trace=_trace(hooks=[_hook("filtered", "hash-filtered")]),
        consistency_traces=[],
        verifier_verdict=True,
    )

    hook_step, submit_step = report["steps"]
    assert hook_step["label"] is None
    assert hook_step["label_kind"] == "unlabeled"
    assert submit_step["label"] == 1.0
    assert submit_step["label_kind"] == "verified"


def test_one_consensus_match_is_enough_for_hook_heuristic():
    report = build_process_report(
        source="llm_gen",
        gold_trace=_trace(hooks=[_hook("filtered", "hash-filtered")]),
        consistency_traces=[
            _trace(hooks=[_hook("same_value", "hash-filtered")]),
            _trace(hooks=[_hook("other_value", "hash-other")]),
        ],
        verifier_verdict=True,
    )

    hook_step = report["steps"][0]
    assert hook_step["evidence"]["consensus_matches"] == 1
    assert hook_step["label"] == 1.0
    assert hook_step["label_kind"] == "heuristic"


def test_only_accepted_submission_receives_verifier_label():
    trace = {
        "turns": [
            {
                "turn_index": 0,
                "reasoning": "Rejected submission",
                "code": "submit(1)",
                "execution": {
                    "success": True,
                    "stdout": _submission_stdout(1),
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 1,
                },
            },
            {
                "turn_index": 1,
                "reasoning": "Accepted submission",
                "code": "submit(2)",
                "execution": {
                    "success": True,
                    "stdout": _submission_stdout(2),
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 2,
                },
            },
        ],
        "final_answer": 2,
        "final_answer_hash": hash_artifact(2),
        "success": True,
    }

    report = build_process_report(
        source="llm_gen",
        gold_trace=trace,
        consistency_traces=[],
        verifier_verdict=True,
    )

    rejected, accepted = report["steps"]
    assert rejected["value"] == 1
    assert rejected["value_hash"] == hash_artifact(1)
    assert rejected["label"] is None
    assert rejected["label_kind"] == "unlabeled"
    assert rejected["label_source"] == "rejected_submission"
    assert rejected["evidence"]["final_verified"] is None
    assert accepted["value"] == 2
    assert accepted["value_hash"] == hash_artifact(2)
    assert accepted["label"] == 1.0
    assert accepted["label_kind"] == "verified"


def test_submit_consensus_is_counted_per_submission_hash():
    trace = {
        "turns": [
            {
                "turn_index": 0,
                "reasoning": "Rejected submission",
                "code": "submit(1)",
                "execution": {
                    "success": True,
                    "stdout": _submission_stdout(1),
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 1,
                },
            },
            {
                "turn_index": 1,
                "reasoning": "Accepted submission",
                "code": "submit(7)",
                "execution": {
                    "success": True,
                    "stdout": _submission_stdout(7),
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 7,
                },
            },
        ],
        "final_answer": 7,
        "final_answer_hash": hash_artifact(7),
        "success": True,
    }
    consistency_traces = [
        _trace(hooks=[], answer=7),
        _trace(hooks=[], answer=7),
        _trace(hooks=[], answer=7, success=False),
    ]

    report = build_process_report(
        source="llm_gen",
        gold_trace=trace,
        consistency_traces=consistency_traces,
        verifier_verdict=True,
    )

    rejected, accepted = report["steps"]
    assert rejected["evidence"]["consensus_matches"] == 0
    assert rejected["evidence"]["consensus_total"] == 2
    assert accepted["evidence"]["consensus_matches"] == 2
    assert accepted["evidence"]["consensus_total"] == 2


def test_future_code_cannot_ground_an_earlier_hook():
    trace = {
        "turns": [
            {
                "turn_index": 0,
                "reasoning": "Claim a future value",
                "code": "early = 1\nhook(early, 'later = 2', name='later')",
                "execution": {
                    "success": True,
                    "stdout": "",
                    "stderr": "",
                    "hooks": [
                        _hook(
                            "later",
                            "hash-later",
                            code_line="later = 2",
                        )
                    ],
                    "submitted_answer": None,
                },
            },
            {
                "turn_index": 1,
                "reasoning": "Actually compute it",
                "code": "later = 2\nsubmit(later)",
                "execution": {
                    "success": True,
                    "stdout": _submission_stdout(2),
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 2,
                },
            },
        ],
        "final_answer": 2,
        "final_answer_hash": hash_artifact(2),
        "success": True,
    }

    report = build_process_report(
        source="template",
        gold_trace=trace,
        consistency_traces=[],
        verifier_verdict=True,
    )

    hook_step = report["steps"][0]
    assert hook_step["evidence"]["code_line_grounded"] is False
    assert hook_step["label"] == 0.0
    assert hook_step["label_kind"] == "heuristic"


def test_later_statement_in_same_cell_cannot_ground_hook():
    trace = _trace(
        hooks=[
            _hook(
                "later",
                "hash-later",
                code_line="later = 2",
                event_line=2,
            )
        ]
    )
    trace["turns"][0]["code"] = (
        "early = 1\nhook(early, 'later = 2', name='later')\nlater = 2\nsubmit(later)"
    )
    trace["turns"][0]["execution"]["submitted_answer"] = 2
    trace["turns"][0]["execution"]["stdout"] = _submission_stdout(2)
    trace["final_answer"] = 2
    trace["final_answer_hash"] = hash_artifact(2)

    report = build_process_report(
        source="template",
        gold_trace=trace,
        consistency_traces=[],
        verifier_verdict=True,
    )

    assert report["steps"][0]["evidence"]["code_line_grounded"] is False


def test_runtime_hook_captures_repeated_caller_line():
    namespace = {}
    exec(get_setup_code(), namespace)
    code = (
        "def emit(value):\n"
        "    hook(value, 'value = current', name='current')\n"
        "for current in range(2):\n"
        "    emit(current)"
    )
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        exec(compile(code, "<cell>", "exec"), namespace, namespace)

    records = [
        json.loads(line[line.find("{") :])
        for line in output.getvalue().splitlines()
        if "📍 Hook:" in line
    ]
    assert [record["event_line"] for record in records] == [2, 2]


def test_missing_hook_event_provenance_is_unlabeled_diagnostic():
    report = build_process_report(
        source="template",
        gold_trace=_trace(hooks=[_hook("filtered", "hash-filtered", event_line=None)]),
        consistency_traces=[],
        verifier_verdict=True,
    )

    hook_step = report["steps"][0]
    assert hook_step["label"] is None
    assert hook_step["label_kind"] == "unlabeled"
    assert hook_step["label_source"] == "event_provenance_unavailable"
    assert hook_step["evidence"]["code_line_grounded"] is False
    assert hook_step["evidence"]["reasons"] == ["missing_or_ambiguous_event_provenance"]


def test_missing_hook_provenance_state_is_unlabeled_diagnostic():
    hook = _hook("filtered", "hash-filtered")
    del hook["event_provenance_reason"]

    report = build_process_report(
        source="template",
        gold_trace=_trace(hooks=[hook]),
        consistency_traces=[],
        verifier_verdict=True,
    )

    hook_step = report["steps"][0]
    assert hook_step["label"] is None
    assert hook_step["label_kind"] == "unlabeled"
    assert hook_step["evidence"]["reasons"] == ["missing_event_provenance_state"]


@pytest.mark.asyncio
async def test_execution_preserves_structured_hooks_before_stdout_truncation():
    hook_payload = {
        "__csv_agent_hook__": True,
        "variable_name": "value",
        "code_line": "value = 1",
        "value": 1,
        "value_hash": "hash-value",
        "depends_on": [],
        "description": None,
        "event_line": 2,
    }
    raw_output = (
        "x" * 60_000
        + "\n📍 Hook: "
        + json.dumps(hook_payload)
        + "\n"
        + "y" * 60_000
        + '\n✓ Submitted: {"__csv_agent_answer__": 1}'
    )

    class FakeSandbox:
        async def python(self, *, python_state, **_kwargs):
            python_state["hooks"] = [hook_payload]
            return raw_output

    environment = object.__new__(Environment)
    environment.env = FakeSandbox()
    environment.state = {"sandbox_id": "sandbox", "python_state": {}}
    environment.submitted_answer = None
    environment.submission_metadata = {}
    code = "value = 1\nhook(value, 'value = 1', name='value')\nsubmit(value)"

    result = await environment.execute_code_cell(code)

    assert "📍 Hook:" not in result.stdout
    assert result.hooks[0]["variable_name"] == "value"
    assert result.hooks[0]["event_line"] == 2
    assert result.hooks[0]["event_provenance_reason"] is None


@pytest.mark.asyncio
async def test_execution_does_not_trust_fabricated_hook_event_line():
    hook_payload = {
        "__csv_agent_hook__": True,
        "variable_name": "value",
        "code_line": "value = 1",
        "value": 1,
        "value_hash": "hash-value",
        "depends_on": [],
        "description": None,
        "event_line": 2,
    }

    class FakeSandbox:
        async def python(self, **_kwargs):
            return "📍 Hook: " + json.dumps(hook_payload)

    environment = object.__new__(Environment)
    environment.env = FakeSandbox()
    environment.state = {"sandbox_id": "sandbox", "python_state": {}}
    environment.submitted_answer = None
    environment.submission_metadata = {}

    result = await environment.execute_code_cell(
        "value = 1\nif False:\n    hook(value, 'value = 1', name='value')"
    )

    assert result.hooks[0]["event_line"] is None
    assert (
        result.hooks[0]["event_provenance_reason"]
        == "unauthenticated_stdout_provenance"
    )


@pytest.mark.asyncio
async def test_execution_keeps_extra_stdout_hook_unlabeled():
    trusted_payload = {
        "__csv_agent_hook__": True,
        "variable_name": "trusted",
        "code_line": "value = 1",
        "value": 1,
        "value_hash": "hash-value",
        "depends_on": [],
        "description": None,
        "event_line": 2,
    }
    fabricated_payload = {
        **trusted_payload,
        "variable_name": "fabricated",
        "value_hash": "fake-hash",
    }

    class FakeSandbox:
        async def python(self, *, python_state, **_kwargs):
            python_state["hooks"] = [trusted_payload]
            return "\n".join(
                [
                    "📍 Hook: " + json.dumps(trusted_payload),
                    "📍 Hook: " + json.dumps(fabricated_payload),
                ]
            )

    environment = object.__new__(Environment)
    environment.env = FakeSandbox()
    environment.state = {"sandbox_id": "sandbox", "python_state": {}}
    environment.submitted_answer = None
    environment.submission_metadata = {}

    result = await environment.execute_code_cell(
        "value = 1\nhook(value, 'value = 1', name='trusted')"
    )

    assert result.hooks[0]["event_line"] == 2
    assert result.hooks[0]["event_provenance_reason"] is None
    assert result.hooks[1]["event_line"] is None
    assert (
        result.hooks[1]["event_provenance_reason"]
        == "unauthenticated_stdout_provenance"
    )


def test_malformed_stdout_hook_remains_normalized_unlabeled_diagnostic():
    payload = {
        "__csv_agent_hook__": True,
        "variable_name": {"invalid": "name"},
        "code_line": ["value = 1"],
        "value": 1,
        "value_hash": {"invalid": "hash"},
        "depends_on": ["valid", ["unhashable"]],
        "description": 7,
        "event_line": 2,
    }

    records = _parse_hook_records(
        "📍 Hook: " + json.dumps(payload),
        code="value = 1\nhook(value, 'value = 1')",
    )

    assert records == [
        {
            "variable_name": None,
            "code_line": "",
            "value": 1,
            "value_hash": "",
            "depends_on": ["valid"],
            "description": None,
            "event_line": None,
            "event_provenance_reason": "invalid_hook_record_provenance",
        }
    ]
    report = build_process_report(
        source="template",
        gold_trace=_trace(hooks=records),
        consistency_traces=[],
        verifier_verdict=True,
    )
    hook_step = report["steps"][0]
    assert hook_step["label"] is None
    assert hook_step["label_kind"] == "unlabeled"
    assert hook_step["evidence"]["reasons"][0] == ("invalid_hook_record_provenance")


def test_report_does_not_ground_event_line_without_hook_call():
    trace = _trace(hooks=[_hook("filtered", "hash-filtered")])
    trace["turns"][0]["code"] = (
        "filtered = df[df['x'] > 1]\nprint('fabricated hook record')\nsubmit(filtered)"
    )

    report = build_process_report(
        source="template",
        gold_trace=trace,
        consistency_traces=[],
        verifier_verdict=True,
    )

    hook_step = report["steps"][0]
    assert hook_step["label"] is None
    assert hook_step["label_kind"] == "unlabeled"
    assert hook_step["evidence"]["code_line_grounded"] is False
    assert hook_step["evidence"]["reasons"] == ["missing_or_ambiguous_event_provenance"]


def test_report_keeps_unauthenticated_stdout_hook_unlabeled():
    report = build_process_report(
        source="template",
        gold_trace=_trace(
            hooks=[
                _hook(
                    "filtered",
                    "hash-filtered",
                    event_line=None,
                    event_provenance_reason="unauthenticated_stdout_provenance",
                )
            ]
        ),
        consistency_traces=[],
        verifier_verdict=True,
    )

    hook_step = report["steps"][0]
    assert hook_step["label"] is None
    assert hook_step["label_kind"] == "unlabeled"
    assert hook_step["label_source"] == "event_provenance_unavailable"
    assert hook_step["evidence"]["reasons"] == ["unauthenticated_stdout_provenance"]


def test_unauthenticated_hooks_do_not_contaminate_trusted_evidence():
    untrusted = _hook(
        "fabricated_dependency",
        "shared-hash",
        event_line=None,
        event_provenance_reason="unauthenticated_stdout_provenance",
    )
    trusted = _hook(
        "trusted",
        "shared-hash",
        depends_on=["fabricated_dependency"],
    )
    consistency_untrusted = _hook(
        "fabricated_consensus",
        "shared-hash",
        event_line=None,
        event_provenance_reason="unauthenticated_stdout_provenance",
    )

    report = build_process_report(
        source="llm_gen",
        gold_trace=_trace(hooks=[untrusted, trusted]),
        consistency_traces=[_trace(hooks=[consistency_untrusted])],
        verifier_verdict=True,
    )

    diagnostic, trusted_step = report["steps"][:2]
    assert diagnostic["label"] is None
    assert trusted_step["evidence"]["dependency_valid"] is False
    assert trusted_step["evidence"]["duplicate"] is False
    assert trusted_step["evidence"]["consensus_matches"] == 0


def test_trusted_hook_order_precedes_unmatched_stdout_diagnostics():
    first = {
        "__csv_agent_hook__": True,
        **_hook(
            "first",
            "hash-first",
            code_line="first = 1",
            value=1,
            event_line=2,
        ),
    }
    second = {
        "__csv_agent_hook__": True,
        **_hook(
            "second",
            "hash-second",
            code_line="second = 2",
            value=2,
            event_line=4,
        ),
    }
    fabricated = {
        "__csv_agent_hook__": True,
        **_hook(
            "fabricated",
            "hash-fabricated",
            code_line="second = 2",
            value=2,
            event_line=4,
        ),
    }
    output = "\n".join(
        [
            "📍 Hook: " + json.dumps(second),
            "📍 Hook: " + json.dumps(fabricated),
            "📍 Hook: " + json.dumps(first),
        ]
    )

    records = _parse_hook_records(
        output,
        code=(
            "first = 1\n"
            "hook(first, 'first = 1', name='first')\n"
            "second = 2\n"
            "hook(second, 'second = 2', name='second')"
        ),
        trusted_records=[first, second],
    )

    assert [record["variable_name"] for record in records] == [
        "first",
        "second",
        "fabricated",
    ]
    assert records[2]["event_line"] is None
    assert records[2]["event_provenance_reason"] == "unauthenticated_stdout_provenance"


def test_trusted_hook_survives_suppressed_stdout():
    trusted = {
        "__csv_agent_hook__": True,
        **_hook(
            "trusted",
            "hash-trusted",
            code_line="value = 1",
            value=1,
            event_line=2,
        ),
    }

    records = _parse_hook_records(
        "",
        code="value = 1\nhook(value, 'value = 1', name='trusted')",
        trusted_records=[trusted],
    )

    assert len(records) == 1
    assert records[0]["variable_name"] == "trusted"
    assert records[0]["event_line"] == 2
    assert records[0]["event_provenance_reason"] is None


def test_same_value_hash_is_duplicate_even_when_hook_is_renamed():
    report = build_process_report(
        source="template",
        gold_trace=_trace(
            hooks=[
                _hook("first_name", "same-hash"),
                _hook("renamed_copy", "same-hash"),
            ]
        ),
        consistency_traces=[],
        verifier_verdict=True,
    )

    first_hook, second_hook = report["steps"][:2]
    assert first_hook["evidence"]["duplicate"] is False
    assert second_hook["evidence"]["duplicate"] is True
    assert second_hook["label"] == 0.0
    assert second_hook["label_kind"] == "heuristic"


def test_hookless_trace_still_has_verified_terminal_observation():
    report = build_process_report(
        source="procedural",
        gold_trace=_trace(hooks=[]),
        consistency_traces=[],
        verifier_verdict=True,
    )

    assert report["summary"]["verified_steps"] == 1
    assert report["summary"]["heuristic_steps"] == 0
    assert report["steps"][0]["step_type"] == "submit"
    assert report["steps"][0]["label_kind"] == "verified"
    assert report["steps"][0]["evidence"]["consensus_matches"] == 0
    assert report["steps"][0]["evidence"]["consensus_total"] == 0


def test_unknown_verifier_verdict_leaves_terminal_observation_unlabeled():
    report = build_process_report(
        source="procedural",
        gold_trace=_trace(hooks=[]),
        consistency_traces=[],
        verifier_verdict=None,
    )

    terminal = report["steps"][0]
    assert terminal["label"] is None
    assert terminal["label_kind"] == "unlabeled"
    assert terminal["label_source"] == "verifier_unavailable"
    assert terminal["evidence"]["final_verified"] is None
    assert report["summary"]["verified_steps"] == 0
    assert report["summary"]["unlabeled_steps"] == 1


@pytest.mark.parametrize("answer", [0, False, ""])
def test_trace_builder_hashes_falsy_answers(answer):
    response = f"Solve\n```python\nsubmit({answer!r})\n```"
    code = extract_python_cells(response)[0]
    result = SimpleNamespace(
        code=code,
        success=True,
        stdout="✓ Submitted: " + json.dumps({"__csv_agent_answer__": answer}),
        stderr="",
        hooks=[],
        submitted_answer=answer,
    )
    state = SimpleNamespace(
        submitted_answer=answer,
        execution_turns=[
            {
                "response": response,
                "code_cells": [code],
                "execution_results": [result],
            }
        ],
    )

    trace = build_trace_dict(state)

    assert trace["success"] is True
    assert trace["final_answer_hash"] == hash_artifact(answer)


@pytest.mark.asyncio
async def test_trace_builder_preserves_turns_pruned_from_conversation_history():
    environment = object.__new__(Environment)
    environment.conversation = ConversationHistory(
        system_prompt="system", max_messages=10
    )
    environment.execution_turns = []
    environment.submitted_answer = None
    environment.submission_metadata = {}
    environment.is_completed = False
    environment.current_turn = 0

    async def execute_cells(code_cells):
        turn_index = len(environment.execution_turns)
        terminal = turn_index == 5
        answer = 6 if terminal else None
        if terminal:
            environment.submitted_answer = answer
        result = SimpleNamespace(
            code=code_cells[0],
            success=True,
            stdout=(
                "✓ Submitted: " + json.dumps({"__csv_agent_answer__": answer})
                if terminal
                else ""
            ),
            stderr="",
            hooks=[],
            submitted_answer=answer,
        )
        return [result], answer

    environment._execute_cells = execute_cells
    environment._build_execution_feedback = lambda _cells, _results: "feedback"
    environment._validate_format = lambda _answer: (True, None)
    environment._validate_hooks = lambda: (True, None)

    for turn_index in range(6):
        code = "submit(6)" if turn_index == 5 else f"value_{turn_index} = {turn_index}"
        response = f"Reasoning for turn {turn_index}\n```python\n{code}\n```"
        await environment.process_turn(response)
        environment.current_turn += 1

    retained_assistant_messages = [
        message
        for message in environment.conversation.to_openai_messages()
        if message["role"] == "assistant"
    ]
    assert len(retained_assistant_messages) == 5

    trace = build_trace_dict(environment)

    assert len(trace["turns"]) == 6
    assert trace["turns"][0]["code"] == "value_0 = 0"
    assert trace["turns"][-1]["execution"]["submitted_answer"] == 6


@pytest.mark.parametrize("fence", ["py", "Python"])
def test_trace_builder_reuses_canonical_code_fence_extraction(fence: str):
    response = f"Explain it clearly\n```{fence}\nsubmit(7)\n```"
    code = extract_python_cells(response)[0]
    result = SimpleNamespace(
        code=code,
        success=True,
        stdout='✓ Submitted: {"__csv_agent_answer__": 7}',
        stderr="",
        hooks=[],
        submitted_answer=7,
    )
    state = SimpleNamespace(
        submitted_answer=7,
        execution_turns=[
            {
                "response": response,
                "code_cells": [code],
                "execution_results": [result],
            }
        ],
    )

    trace = build_trace_dict(state)

    assert trace["turns"][0]["code"] == "submit(7)"
    assert trace["turns"][0]["reasoning"] == "Explain it clearly"


@pytest.mark.parametrize("mismatch", ["recorded", "executed"])
def test_trace_builder_rejects_mismatched_turn_provenance(mismatch: str):
    response = "Explain it clearly\n```python\nsubmit(7)\n```"
    code = extract_python_cells(response)[0]
    recorded_code = "submit(8)" if mismatch == "recorded" else code
    executed_code = "submit(8)" if mismatch == "executed" else code
    result = SimpleNamespace(
        code=executed_code,
        success=True,
        stdout='✓ Submitted: {"__csv_agent_answer__": 7}',
        stderr="",
        hooks=[],
        submitted_answer=7,
    )
    state = SimpleNamespace(
        submitted_answer=7,
        execution_turns=[
            {
                "response": response,
                "code_cells": [recorded_code],
                "execution_results": [result],
            }
        ],
    )

    with pytest.raises(ValueError, match="match response"):
        build_trace_dict(state)


@pytest.mark.parametrize("answer", [0, False, ""])
def test_submit_consensus_falls_back_to_hashing_falsy_answers(answer):
    gold_trace = _trace(hooks=[], answer=answer)
    gold_trace["final_answer_hash"] = None
    consistency_trace = _trace(hooks=[], answer=answer)
    consistency_trace["final_answer_hash"] = None

    report = build_process_report(
        source="llm_gen",
        gold_trace=gold_trace,
        consistency_traces=[consistency_trace],
        verifier_verdict=True,
    )

    terminal = report["steps"][0]
    assert terminal["value_hash"] == hash_artifact(answer)
    assert terminal["evidence"]["consensus_matches"] == 1
    assert terminal["evidence"]["consensus_total"] == 1


@pytest.mark.parametrize(
    "code",
    [
        "submit(7)\nafter = 1",
        "submit(7)\nsubmit(8)",
        "submit(7)\nhook(7, 'value = 7')",
    ],
)
def test_process_report_rejects_nonterminal_or_multiple_submissions(code: str):
    trace = _trace(hooks=[])
    trace["turns"][0]["code"] = code

    with pytest.raises(ValueError, match="submission|submitted operation"):
        build_process_report(
            source="template",
            gold_trace=trace,
            consistency_traces=[],
            verifier_verdict=True,
        )


@pytest.mark.parametrize(
    ("submitted", "final_answer"),
    [(False, 0), (0, False), (1, 1.0), (1.0, 1)],
)
def test_process_report_rejects_type_distinct_accepted_submission(
    submitted, final_answer
):
    trace = _trace(hooks=[], answer=final_answer)
    trace["turns"][0]["execution"]["submitted_answer"] = submitted
    trace["turns"][0]["execution"]["stdout"] = _submission_stdout(submitted)

    with pytest.raises(ValueError, match="final answer is not the accepted submission"):
        build_process_report(
            source="template",
            gold_trace=trace,
            consistency_traces=[],
            verifier_verdict=True,
        )


def test_process_report_rejects_distinct_values_with_same_rounded_hash():
    trace = _trace(hooks=[], answer=1.004)
    trace["turns"][0]["execution"]["submitted_answer"] = 1.001
    trace["turns"][0]["execution"]["stdout"] = _submission_stdout(1.001)
    assert hash_artifact(1.001) == hash_artifact(1.004)

    with pytest.raises(ValueError, match="final answer is not the accepted submission"):
        build_process_report(
            source="template",
            gold_trace=trace,
            consistency_traces=[],
            verifier_verdict=True,
        )


@pytest.mark.parametrize(
    ("logged_answer", "captured_answer"),
    [(8, 7), (False, 0)],
)
def test_process_report_rejects_submission_record_mismatch(
    logged_answer, captured_answer
):
    trace = _trace(hooks=[], answer=captured_answer)
    trace["turns"][0]["execution"]["stdout"] = "✓ Submitted: " + json.dumps(
        {"__csv_agent_answer__": logged_answer}
    )

    with pytest.raises(ValueError, match="does not match captured answer"):
        build_process_report(
            source="template",
            gold_trace=trace,
            consistency_traces=[],
            verifier_verdict=True,
        )


def test_process_report_rejects_captured_submission_without_logged_record():
    trace = _trace(hooks=[], answer=7)
    trace["turns"][0]["execution"]["stdout"] = ""

    with pytest.raises(ValueError, match="submission record is missing"):
        build_process_report(
            source="template",
            gold_trace=trace,
            consistency_traces=[],
            verifier_verdict=True,
        )


def test_process_report_rejects_turns_appended_after_accepted_submission():
    trace = _trace(hooks=[], answer=7)
    trace["turns"].append(
        {
            "turn_index": 1,
            "reasoning": "Unexpected later execution",
            "code": "later = 1",
            "execution": {
                "success": True,
                "stdout": "",
                "stderr": "",
                "hooks": [],
                "submitted_answer": None,
            },
        }
    )

    with pytest.raises(ValueError, match="accepted submission is not in the final turn"):
        build_process_report(
            source="template",
            gold_trace=trace,
            consistency_traces=[],
            verifier_verdict=True,
        )


def test_process_report_rejects_failed_accepted_submission_execution():
    trace = _trace(hooks=[], answer=7)
    trace["turns"][0]["execution"]["success"] = False

    with pytest.raises(ValueError, match="accepted submission execution failed"):
        build_process_report(
            source="template",
            gold_trace=trace,
            consistency_traces=[],
            verifier_verdict=True,
        )


@pytest.mark.parametrize("answer", ["contains ✓ Submitted: marker", "📍 Hook:"])
def test_submission_payload_protocol_text_is_not_a_second_event(answer):
    trace = _trace(hooks=[], answer=answer)
    trace["turns"][0]["execution"]["stdout"] = "✓ Submitted: " + json.dumps(
        {"__csv_agent_answer__": answer}
    )

    report = build_process_report(
        source="template",
        gold_trace=trace,
        consistency_traces=[],
        verifier_verdict=True,
    )

    assert report["steps"][0]["label_kind"] == "verified"
