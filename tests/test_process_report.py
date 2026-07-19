"""Tests for diagnostic process reports."""

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
        verified=True,
        majority_count=1,
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
        verified=True,
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
        verified=True,
        majority_count=2,
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
        verified=True,
        majority_count=1,
    )

    hook_step, submit_step = report["steps"]
    assert hook_step["label"] is None
    assert hook_step["label_kind"] == "unlabeled"
    assert submit_step["label"] == 1.0
    assert submit_step["label_kind"] == "verified"


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
        verified=True,
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
        verified=True,
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
        verified=True,
    )

    assert report["summary"]["verified_steps"] == 1
    assert report["summary"]["heuristic_steps"] == 0
    assert report["steps"][0]["step_type"] == "submit"
    assert report["steps"][0]["label_kind"] == "verified"
