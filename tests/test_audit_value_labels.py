"""Tests for mechanical prefix-label failure auditing."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.experiments.audit_value_labels import audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _continuation(
    verdict: bool, *, answer=None, execution_success: bool = True
) -> dict:
    return {
        "verifier_verdict": verdict,
        "error": None,
        "trace": {
            "final_answer": answer,
            "turns": [
                {
                    "execution": {
                        "success": execution_success,
                        "submitted_answer": answer,
                    }
                }
            ],
        },
    }


def test_audit_separates_runtime_failures_and_rejected_answers(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    _write_jsonl(
        tasks_dir / "train.jsonl",
        [
            {
                "episode_id": "episode-1",
                "question": {
                    "template_name": "summary",
                    "question_text": "Count rows.",
                    "ground_truth": 10,
                },
            }
        ],
    )
    _write_jsonl(
        tmp_path / "train-values.jsonl",
        [
            {
                "prefix": {"prefix_id": "prefix-1", "episode_id": "episode-1"},
                "continuations": [
                    _continuation(True, answer=10),
                    _continuation(False, answer=11),
                    _continuation(False, answer=11),
                    _continuation(False),
                    _continuation(False, execution_success=False),
                    _continuation(True, answer=10),  # held out
                ],
            }
        ],
    )
    output = tmp_path / "audit.json"

    result = audit(
        Namespace(
            tasks_dir=tasks_dir,
            values_dir=tmp_path,
            output=output,
            splits=["train"],
            holdout_continuations=1,
        )
    )

    train = result["splits"]["train"]
    assert train["label_continuations"] == 5
    assert train["accepted"] == 1
    assert train["failure_modes"] == {
        "execution_error": 1,
        "no_terminal_submission": 1,
        "rejected_submission": 2,
    }
    assert train["unique_rejected_answer_clusters"] == 1
    assert train["rejected_answer_clusters"][0]["count"] == 2
    assert json.loads(output.read_text()) == result


def test_audit_rejects_missing_boolean_label(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    _write_jsonl(
        tasks_dir / "train.jsonl",
        [{"episode_id": "episode-1", "question": {}}],
    )
    continuation = _continuation(False)
    continuation["verifier_verdict"] = None
    _write_jsonl(
        tmp_path / "train-values.jsonl",
        [
            {
                "prefix": {"prefix_id": "prefix-1", "episode_id": "episode-1"},
                "continuations": [continuation],
            }
        ],
    )

    with pytest.raises(ValueError, match="no boolean verifier verdict"):
        audit(
            Namespace(
                tasks_dir=tasks_dir,
                values_dir=tmp_path,
                output=tmp_path / "audit.json",
                splits=["train"],
                holdout_continuations=0,
            )
        )
