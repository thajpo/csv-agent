"""Audit mechanical failure modes in verifier-labeled prefix continuations.

This command deliberately does not decide whether a rejected answer is
semantically wrong. It separates objective runtime outcomes and emits distinct
rejected-answer clusters for human adjudication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize prefix-value failures without inspecting test data."
    )
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--values-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation"]
    )
    parser.add_argument("--holdout-continuations", type=int, default=2)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _answer_key(answer: Any) -> str:
    payload = json.dumps(answer, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _failure_mode(continuation: dict[str, Any]) -> str:
    trace = continuation.get("trace") or {}
    if trace.get("final_answer") is not None:
        return "rejected_submission"
    if continuation.get("error"):
        return "collector_error"
    turns = trace.get("turns") or []
    if not turns:
        return "no_trace"
    execution = turns[-1].get("execution") or {}
    if not execution.get("success", False):
        return "execution_error"
    return "no_terminal_submission"


def audit_split(
    *,
    split: str,
    tasks_path: Path,
    values_path: Path,
    holdout_continuations: int,
) -> dict[str, Any]:
    tasks = {row["episode_id"]: row for row in _load_jsonl(tasks_path)}
    records = _load_jsonl(values_path)
    outcomes: Counter[str] = Counter()
    template_failures: dict[str, Counter[str]] = {}
    rejected_clusters: Counter[tuple[str, str]] = Counter()
    rejected_answers: dict[tuple[str, str], Any] = {}

    for record in records:
        episode_id = record["prefix"]["episode_id"]
        if episode_id not in tasks:
            raise ValueError(f"value record references unknown episode {episode_id}")
        question = tasks[episode_id]["question"]
        template_name = question.get("template_name") or "<unknown>"
        continuations = record["continuations"]
        label_count = len(continuations) - holdout_continuations
        if label_count <= 0:
            raise ValueError(
                f"record {record['prefix']['prefix_id']} has no label continuations"
            )

        for continuation in continuations[:label_count]:
            verdict = continuation.get("verifier_verdict")
            if not isinstance(verdict, bool):
                raise ValueError("label continuation has no boolean verifier verdict")
            mode = "accepted" if verdict else _failure_mode(continuation)
            outcomes[mode] += 1
            if verdict:
                continue
            template_failures.setdefault(template_name, Counter())[mode] += 1
            if mode == "rejected_submission":
                answer = continuation["trace"]["final_answer"]
                key = (episode_id, _answer_key(answer))
                rejected_clusters[key] += 1
                rejected_answers[key] = answer

    clusters = []
    for (episode_id, answer_hash), count in sorted(rejected_clusters.items()):
        question = tasks[episode_id]["question"]
        clusters.append(
            {
                "split": split,
                "episode_id": episode_id,
                "template_name": question.get("template_name"),
                "count": count,
                "question": question.get("question_text"),
                "expected_answer": question.get("ground_truth"),
                "actual_answer": rejected_answers[(episode_id, answer_hash)],
                "actual_answer_hash": answer_hash,
            }
        )

    total = sum(outcomes.values())
    accepted = outcomes["accepted"]
    return {
        "task_count": len(tasks),
        "candidate_count": len(records),
        "label_continuations": total,
        "accepted": accepted,
        "acceptance_rate": accepted / total if total else 0.0,
        "failure_modes": {
            key: value for key, value in sorted(outcomes.items()) if key != "accepted"
        },
        "unique_rejected_answer_clusters": len(clusters),
        "episodes_with_rejected_submissions": len(
            {cluster["episode_id"] for cluster in clusters}
        ),
        "template_failures": {
            template: dict(sorted(counts.items()))
            for template, counts in sorted(template_failures.items())
        },
        "rejected_answer_clusters": clusters,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.holdout_continuations < 0:
        raise ValueError("holdout continuations cannot be negative")
    split_results = {
        split: audit_split(
            split=split,
            tasks_path=args.tasks_dir / f"{split}.jsonl",
            values_path=args.values_dir / f"{split}-values.jsonl",
            holdout_continuations=args.holdout_continuations,
        )
        for split in args.splits
    }
    combined_modes: Counter[str] = Counter()
    for result in split_results.values():
        combined_modes.update(result["failure_modes"])
    result = {
        "holdout_continuations_excluded": args.holdout_continuations,
        "splits": split_results,
        "combined": {
            "task_count": sum(item["task_count"] for item in split_results.values()),
            "candidate_count": sum(
                item["candidate_count"] for item in split_results.values()
            ),
            "label_continuations": sum(
                item["label_continuations"] for item in split_results.values()
            ),
            "accepted": sum(item["accepted"] for item in split_results.values()),
            "failure_modes": dict(sorted(combined_modes.items())),
            "unique_rejected_answer_clusters": sum(
                item["unique_rejected_answer_clusters"]
                for item in split_results.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    result = audit(parse_args())
    print(json.dumps(result["combined"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
