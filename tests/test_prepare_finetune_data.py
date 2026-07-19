"""Tests for training-format conversion."""

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
                        "hooks": [],
                        "submitted_answer": 7,
                    },
                }
            ],
            "final_answer": 7,
            "final_answer_hash": "answer-7",
            "success": True,
        },
        "process_report": {
            "summary": {},
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
