"""Tests for canonical PRM process reports."""

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


def test_template_verified_trace_labels_hooks_and_submit_gold():
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
        "gold_steps": 2,
        "strong_steps": 0,
        "weak_steps": 0,
        "unlabeled_steps": 0,
        "positive_steps": 2,
        "negative_steps": 0,
    }
    assert [step["confidence"] for step in report["steps"]] == ["gold", "gold"]
    assert [step["label"] for step in report["steps"]] == [1.0, 1.0]


def test_bad_hook_evidence_gets_negative_label():
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
    assert hook_step["confidence"] == "gold"
    assert hook_step["evidence"]["code_line_grounded"] is False
    assert hook_step["evidence"]["dependency_valid"] is False
    assert hook_step["evidence"]["reasons"] == [
        "ungrounded_code_line",
        "invalid_dependency",
    ]


def test_llm_hook_requires_consensus_for_strong_positive_label():
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

    hook_step = report["steps"][0]
    submit_step = report["steps"][1]
    assert hook_step["label"] == 1.0
    assert hook_step["confidence"] == "strong"
    assert hook_step["evidence"]["consensus_matches"] == 2
    assert hook_step["evidence"]["consensus_total"] == 3
    assert submit_step["confidence"] == "strong"


def test_llm_hook_without_consensus_is_not_exportable_by_default():
    report = build_process_report(
        source="llm_gen",
        gold_trace=_trace(hooks=[_hook("filtered", "hash-filtered")]),
        consistency_traces=[],
        verified=True,
        majority_count=1,
    )

    hook_step = report["steps"][0]
    submit_step = report["steps"][1]
    assert hook_step["label"] is None
    assert hook_step["confidence"] == "unlabeled"
    assert submit_step["label"] == 1.0
    assert submit_step["confidence"] == "weak"
