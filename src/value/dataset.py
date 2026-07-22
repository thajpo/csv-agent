"""Leakage-safe inputs for training a trajectory value model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from csv_spec import PrefixValueRecord, TrajectoryPrefix


@dataclass(frozen=True)
class ValueExample:
    """One partial attempt and its repeated-continuation outcome counts."""

    prefix_id: str
    episode_id: str
    dataset_id: str
    text: str
    successes: int
    attempts: int
    turns_consumed: int
    turns_left: int
    execution_failed: bool
    output_chars: int
    selection_verdicts: tuple[bool, ...] = ()

    @property
    def target(self) -> float:
        return self.successes / self.attempts

    @property
    def held_out_target(self) -> float:
        if not self.selection_verdicts:
            raise ValueError("example has no held-out continuation outcomes")
        return sum(self.selection_verdicts) / len(self.selection_verdicts)


def render_prefix(prefix: TrajectoryPrefix) -> str:
    """Render only information available to the actor at this boundary."""
    sections = ["[SYSTEM]", prefix.system_prompt, "[WORK SO FAR]"]
    if prefix.conversation_messages:
        for message in prefix.conversation_messages:
            sections.extend([f"[{message['role'].upper()}]", message["content"]])
    else:
        sections.append("No actions have been taken.")
    sections.extend(["[TURNS LEFT]", str(prefix.max_turns - prefix.consumed_turns)])
    return "\n".join(sections)


def _dataset_id(csv_source: str) -> str:
    path = Path(csv_source)
    return path.parent.name or path.stem


def example_from_record(
    record: PrefixValueRecord, *, holdout_continuations: int = 1
) -> ValueExample:
    """Create a trainer row while reserving final rollouts for selection tests."""
    if holdout_continuations < 0:
        raise ValueError("holdout_continuations cannot be negative")
    if len(record.continuations) <= holdout_continuations:
        raise ValueError("record does not contain enough labeling continuations")

    split_at = len(record.continuations) - holdout_continuations
    label_outcomes = record.continuations[:split_at]
    held_out = record.continuations[split_at:]
    if any(outcome.verifier_verdict is None for outcome in label_outcomes):
        raise ValueError("labeling continuation is missing a verifier verdict")
    if any(outcome.verifier_verdict is None for outcome in held_out):
        raise ValueError("selection continuation is missing a verifier verdict")

    successes = sum(outcome.verifier_verdict is True for outcome in label_outcomes)
    prefix = record.prefix
    executions = [turn["execution"] for turn in prefix.turns]
    output_chars = sum(
        len(execution.get("stdout", "")) + len(execution.get("stderr", ""))
        for execution in executions
    )
    return ValueExample(
        prefix_id=prefix.prefix_id,
        episode_id=prefix.episode_id,
        dataset_id=_dataset_id(prefix.csv_source),
        text=render_prefix(prefix),
        successes=successes,
        attempts=len(label_outcomes),
        turns_consumed=prefix.consumed_turns,
        turns_left=prefix.max_turns - prefix.consumed_turns,
        execution_failed=any(
            not execution.get("success", False) for execution in executions
        ),
        output_chars=output_chars,
        selection_verdicts=tuple(
            outcome.verifier_verdict is True for outcome in held_out
        ),
    )


def load_value_examples(
    path: Path, *, holdout_continuations: int = 1
) -> list[ValueExample]:
    examples: list[ValueExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = PrefixValueRecord.model_validate_json(line)
                examples.append(
                    example_from_record(
                        record, holdout_continuations=holdout_continuations
                    )
                )
            except Exception as error:
                raise ValueError(
                    f"invalid value record at {path}:{line_number}"
                ) from error
    if not examples:
        raise ValueError(f"no value examples found in {path}")
    return examples


def assert_dataset_disjoint(
    train: Iterable[ValueExample],
    validation: Iterable[ValueExample],
    test: Iterable[ValueExample],
) -> None:
    groups = {
        "train": {example.dataset_id for example in train},
        "validation": {example.dataset_id for example in validation},
        "test": {example.dataset_id for example in test},
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlaps = {
        f"{left}/{right}": sorted(groups[left] & groups[right])
        for left, right in pairs
        if groups[left] & groups[right]
    }
    if overlaps:
        raise ValueError(f"CSV datasets occur in multiple splits: {overlaps}")


def simple_features(example: ValueExample) -> list[float]:
    """Signals that the learned text model must beat to be interesting."""
    return [
        float(example.turns_consumed),
        float(example.turns_left),
        float(example.execution_failed),
        math.log1p(example.output_chars),
    ]
