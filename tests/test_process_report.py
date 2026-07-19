"""Tests for diagnostic process reports."""

from csv_spec import hash_artifact
from src.datagen.process_report import build_process_report


def _trace(*, hooks: list[dict], answer=7, success=True) -> dict:
    return {
        "turns": [
            {
                "turn_index": 0,
                "reasoning": "Solve",
                "code": (
                    "filtered = df[df['x'] > 1]\n"
                    "total = filtered['y'].sum()\n"
                    "submit(total)"
                ),
                "execution": {
                    "success": success,
                    "stdout": "",
                    "stderr": "",
                    "hooks": hooks,
                    "submitted_answer": answer,
                },
            }
        ],
        "final_answer": answer,
        "final_answer_hash": f"answer-{answer}",
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
                    "stdout": "",
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
                    "stdout": "",
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 2,
                },
            },
        ],
        "final_answer": 2,
        "final_answer_hash": "accepted-answer-hash",
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
    assert accepted["value_hash"] == "accepted-answer-hash"
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
                    "stdout": "",
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
                    "stdout": "",
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 7,
                },
            },
        ],
        "final_answer": 7,
        "final_answer_hash": "answer-7",
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
                "code": "early = 1",
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
                    "stdout": "",
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 2,
                },
            },
        ],
        "final_answer": 2,
        "final_answer_hash": "answer-2",
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
