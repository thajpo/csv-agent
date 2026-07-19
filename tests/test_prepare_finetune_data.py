"""Tests for training-format conversion."""

import copy

import pytest

from src.training.prepare_finetune_data import convert_episodes, to_prm_samples


def _episode() -> dict:
    return {
        "episode_id": "ep-prm-001",
        "csv_source": "data.csv",
        "verified": False,
        "question": {
            "question_text": "Compute the answer",
            "hint": "Use the filtered rows",
        },
        "gold_trace": {
            "turns": [
                {
                    "turn_index": 0,
                    "reasoning": "Filter and aggregate",
                    "code": "filtered = df[df['x'] > 1]\nsubmit(7)",
                    "execution": {
                        "success": True,
                        "stdout": "ok",
                        "stderr": "",
                        "hooks": [
                            {
                                "variable_name": "filtered",
                                "code_line": "filtered = df[df['x'] > 1]",
                                "value": {"rows": 3},
                                "value_hash": "hash-filtered",
                                "description": None,
                                "depends_on": [],
                            },
                            {
                                "variable_name": "unused",
                                "code_line": "unused = 1",
                                "value": 1,
                                "value_hash": "hash-unused",
                                "description": None,
                                "depends_on": [],
                            },
                        ],
                        "submitted_answer": 7,
                    },
                }
            ],
            "final_answer": 7,
            "final_answer_hash": "answer-7",
            "success": True,
        },
        "process_report": {
            "summary": {
                "total_steps": 3,
                "labeled_steps": 2,
                "verified_steps": 1,
                "heuristic_steps": 1,
                "unlabeled_steps": 1,
                "positive_steps": 2,
                "negative_steps": 0,
            },
            "steps": [
                {
                    "step_index": 0,
                    "turn_index": 0,
                    "hook_index": 0,
                    "step_type": "hook",
                    "code_line": "filtered = df[df['x'] > 1]",
                    "variable_name": "filtered",
                    "semantic_role": "filter",
                    "value": {"rows": 3},
                    "value_hash": "hash-filtered",
                    "description": None,
                    "depends_on": [],
                    "label": 1.0,
                    "label_kind": "heuristic",
                    "label_source": "trace_consensus_heuristic",
                    "evidence": {"consensus_matches": 2},
                },
                {
                    "step_index": 1,
                    "turn_index": 0,
                    "hook_index": None,
                    "step_type": "submit",
                    "code_line": "submit(...)",
                    "variable_name": "answer",
                    "semantic_role": "final_answer",
                    "value": 7,
                    "value_hash": "answer-7",
                    "description": "Final submitted answer",
                    "depends_on": [],
                    "label": 1.0,
                    "label_kind": "verified",
                    "label_source": "terminal_verifier",
                    "evidence": {"final_verified": True},
                },
                {
                    "step_index": 2,
                    "turn_index": 0,
                    "hook_index": 1,
                    "step_type": "hook",
                    "code_line": "unused = 1",
                    "variable_name": "unused",
                    "semantic_role": None,
                    "value": 1,
                    "value_hash": "hash-unused",
                    "description": None,
                    "depends_on": [],
                    "label": None,
                    "label_kind": "unlabeled",
                    "label_source": "insufficient_evidence",
                    "evidence": {},
                },
            ],
        },
    }


def test_prm_export_defaults_to_externally_verified_steps():
    samples = to_prm_samples(_episode())

    assert len(samples) == 1
    sample = samples[0]
    assert sample["episode_id"] == "ep-prm-001"
    assert sample["step_index"] == 1
    assert sample["step_type"] == "submit"
    assert sample["label"] == 1.0
    assert sample["label_kind"] == "verified"
    assert sample["evidence"] == {"final_verified": True}


def test_prm_export_includes_hook_heuristics_only_when_requested():
    samples = convert_episodes(
        [_episode()],
        "prm",
        include_heuristic_hooks=True,
    )

    assert [sample["step_index"] for sample in samples] == [0, 1]
    assert [sample["label_kind"] for sample in samples] == [
        "heuristic",
        "verified",
    ]


def test_prm_export_rejects_missing_process_report():
    episode = _episode()
    del episode["process_report"]

    with pytest.raises(ValueError, match="missing required process_report"):
        to_prm_samples(episode)


def test_prm_export_rejects_malformed_process_report():
    episode = _episode()
    del episode["process_report"]["steps"][0]["step_type"]

    with pytest.raises(ValueError, match="malformed process_report"):
        to_prm_samples(episode)


def test_prm_export_rejects_verified_hook():
    episode = _episode()
    episode["process_report"]["steps"][0]["label_kind"] = "verified"

    with pytest.raises(ValueError, match="verified labels are only valid on submit"):
        to_prm_samples(episode)


def test_prm_export_rejects_nonnumeric_label():
    episode = _episode()
    episode["process_report"]["steps"][0]["label"] = "1.0"

    with pytest.raises(ValueError, match="malformed process_report"):
        to_prm_samples(episode)


@pytest.mark.parametrize(
    ("step_index", "label", "label_kind", "message"),
    [
        (2, 0.0, "unlabeled", "unlabeled steps must not have a numeric label"),
        (0, None, "heuristic", "labeled steps require a numeric label"),
    ],
)
def test_prm_export_rejects_label_kind_mismatches(
    step_index: int,
    label: float | None,
    label_kind: str,
    message: str,
):
    episode = _episode()
    step = episode["process_report"]["steps"][step_index]
    step["label"] = label
    step["label_kind"] = label_kind

    with pytest.raises(ValueError, match=message):
        to_prm_samples(episode)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step_index", 99),
        ("turn_index", 99),
    ],
)
def test_prm_export_rejects_out_of_range_indices(field: str, value: int):
    episode = _episode()
    episode["process_report"]["steps"][0][field] = value

    with pytest.raises(ValueError, match=field):
        to_prm_samples(episode)


def test_prm_export_rejects_out_of_range_hook_index():
    episode = _episode()
    episode["process_report"]["steps"][0]["hook_index"] = 99

    with pytest.raises(ValueError, match="hook_index"):
        to_prm_samples(episode)


def test_prm_export_allows_valid_report_without_eligible_labels():
    episode = _episode()
    unlabeled_step = copy.deepcopy(episode["process_report"]["steps"][2])
    unlabeled_step["step_index"] = 0
    episode["process_report"]["steps"] = [unlabeled_step]
    episode["process_report"]["summary"] = {
        "total_steps": 1,
        "labeled_steps": 0,
        "verified_steps": 0,
        "heuristic_steps": 0,
        "unlabeled_steps": 1,
        "positive_steps": 0,
        "negative_steps": 0,
    }

    assert to_prm_samples(episode) == []
