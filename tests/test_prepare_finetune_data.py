"""Tests for training-format conversion."""

import copy

import pytest

from src.datagen.process_report import build_process_report
from src.training.prepare_finetune_data import convert_episodes, to_prm_samples


def _episode() -> dict:
    gold_trace = {
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
    }
    consistency_trace = {
        "turns": [
            {
                "turn_index": 0,
                "reasoning": "Check independently",
                "code": "filtered = df[df['x'] > 1]\nsubmit(7)",
                "execution": {
                    "success": True,
                    "stdout": "ok",
                    "stderr": "",
                    "hooks": [
                        copy.deepcopy(gold_trace["turns"][0]["execution"]["hooks"][0])
                    ],
                    "submitted_answer": 7,
                },
            }
        ],
        "final_answer": 7,
        "final_answer_hash": "answer-7",
        "success": True,
    }
    episode = {
        "episode_id": "ep-prm-001",
        "csv_source": "data.csv",
        "source": "llm_gen",
        "verified": True,
        "question": {
            "source": "llm_gen",
            "question_text": "Compute the answer",
            "hint": "Use the filtered rows",
            "ground_truth_hash": "answer-7",
            "ground_truth_hashes": ["answer-7"],
        },
        "gold_trace": gold_trace,
        "consistency_traces": [consistency_trace],
        "triangulation": {
            "n_consistency_runs": 1,
            "n_consistency_succeeded": 1,
            "majority_answer_hash": "answer-7",
            "majority_count": 1,
            "gold_matches_majority": True,
        },
    }
    episode["process_report"] = build_process_report(
        source="llm_gen",
        gold_trace=gold_trace,
        consistency_traces=[consistency_trace],
        verifier_verdict=True,
    )
    return episode


def test_prm_export_defaults_to_externally_verified_steps():
    samples = to_prm_samples(_episode())

    assert len(samples) == 1
    sample = samples[0]
    assert sample["episode_id"] == "ep-prm-001"
    assert sample["step_index"] == 2
    assert sample["step_type"] == "submit"
    assert sample["label"] == 1.0
    assert sample["label_kind"] == "verified"
    assert sample["evidence"]["final_verified"] is True
    assert sample["evidence"]["consensus_matches"] == 1
    assert sample["evidence"]["consensus_total"] == 1


def test_prm_export_includes_hook_heuristics_only_when_requested():
    samples = convert_episodes(
        [_episode()],
        "prm",
        include_heuristic_hooks=True,
    )

    assert [sample["step_index"] for sample in samples] == [0, 1, 2]
    assert [sample["label_kind"] for sample in samples] == [
        "heuristic",
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

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


def test_prm_export_rejects_nonnumeric_label():
    episode = _episode()
    episode["process_report"]["steps"][0]["label"] = "1.0"

    with pytest.raises(ValueError, match="malformed process_report"):
        to_prm_samples(episode)


@pytest.mark.parametrize(
    ("step_index", "label", "label_kind"),
    [
        (1, 0.0, "unlabeled"),
        (0, None, "heuristic"),
    ],
)
def test_prm_export_rejects_label_kind_mismatches(
    step_index: int,
    label: float | None,
    label_kind: str,
):
    episode = _episode()
    step = episode["process_report"]["steps"][step_index]
    step["label"] = label
    step["label_kind"] = label_kind

    with pytest.raises(ValueError, match="does not match the canonical report"):
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

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


def test_prm_export_rejects_out_of_range_hook_index():
    episode = _episode()
    episode["process_report"]["steps"][0]["hook_index"] = 99

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_prm_export_rejects_noncanonical_trace_coverage(mutation: str):
    episode = _episode()
    if mutation == "missing":
        episode["process_report"]["steps"].pop(1)
    else:
        episode["process_report"]["steps"].insert(
            1, copy.deepcopy(episode["process_report"]["steps"][0])
        )

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


def test_prm_export_rejects_observations_out_of_canonical_order():
    episode = _episode()
    steps = episode["process_report"]["steps"]
    steps[1], steps[2] = steps[2], steps[1]
    for step_index, step in enumerate(steps):
        step["step_index"] = step_index

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


@pytest.mark.parametrize(
    ("step_index", "field", "value"),
    [
        (0, "variable_name", "fabricated"),
        (0, "value_hash", "fabricated"),
        (2, "value", 99),
        (2, "value_hash", "fabricated"),
    ],
)
def test_prm_export_rejects_observations_not_bound_to_trace(
    step_index: int,
    field: str,
    value: object,
):
    episode = _episode()
    episode["process_report"]["steps"][step_index][field] = value

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


def test_prm_export_rejects_verified_label_on_rejected_submission():
    episode = _episode()
    accepted_turn = copy.deepcopy(episode["gold_trace"]["turns"][0])
    accepted_turn["turn_index"] = 1
    rejected_turn = {
        "turn_index": 0,
        "reasoning": "Try the wrong answer",
        "code": "submit(3)",
        "execution": {
            "success": True,
            "stdout": "",
            "stderr": "",
            "hooks": [],
            "submitted_answer": 3,
        },
    }
    episode["gold_trace"]["turns"] = [rejected_turn, accepted_turn]
    episode["process_report"] = build_process_report(
        source="llm_gen",
        gold_trace=episode["gold_trace"],
        consistency_traces=[],
        verifier_verdict=True,
    )
    rejected = episode["process_report"]["steps"][0]
    rejected["label"] = 1.0
    rejected["label_kind"] = "verified"
    rejected["label_source"] = "terminal_verifier"
    rejected["evidence"]["final_verified"] = True

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


def test_prm_export_rejects_terminal_label_that_disagrees_with_verdict():
    episode = _episode()
    terminal = episode["process_report"]["steps"][2]
    terminal["evidence"]["final_verified"] = False

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


@pytest.mark.parametrize("field", ["consensus_matches", "consensus_total"])
def test_prm_export_rejects_submit_consensus_not_derived_from_traces(field: str):
    episode = _episode()
    terminal = episode["process_report"]["steps"][2]
    terminal["evidence"][field] = 99

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


def test_prm_export_rejects_summary_not_derived_from_steps():
    episode = _episode()
    episode["process_report"]["summary"]["total_steps"] = 99

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("step", "semantic_role", "fabricated"),
        ("step", "label_source", "fabricated"),
        ("evidence", "code_line_grounded", False),
        ("evidence", "dependency_valid", False),
        ("evidence", "duplicate", True),
        ("evidence", "consensus_matches", 0),
        ("evidence", "consensus_total", 0),
        ("evidence", "reasons", ["fabricated"]),
    ],
)
def test_prm_export_rejects_mutated_hook_metadata(
    section: str, field: str, value: object
):
    episode = _episode()
    hook = episode["process_report"]["steps"][0]
    target = hook if section == "step" else hook["evidence"]
    target[field] = value

    with pytest.raises(ValueError, match="does not match the canonical report"):
        to_prm_samples(episode, include_heuristic_hooks=True)


def test_prm_export_requires_verifier_provenance():
    episode = _episode()
    del episode["triangulation"]

    with pytest.raises(ValueError, match="triangulation must be an object"):
        to_prm_samples(episode)


def test_prm_export_allows_valid_report_without_eligible_labels():
    episode = _episode()
    episode["verified"] = False
    episode["gold_trace"]["turns"][0]["execution"]["hooks"] = []
    episode["consistency_traces"] = []
    episode["triangulation"] = {
        "n_consistency_runs": 0,
        "n_consistency_succeeded": 0,
        "majority_answer_hash": None,
        "majority_count": 0,
        "gold_matches_majority": False,
    }
    episode["process_report"] = build_process_report(
        source="llm_gen",
        gold_trace=episode["gold_trace"],
        consistency_traces=[],
        verifier_verdict=None,
    )

    assert to_prm_samples(episode) == []
