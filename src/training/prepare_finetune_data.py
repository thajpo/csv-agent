"""
Prepare fine-tuning data from episodes.

Converts structured episode data to various training formats:
- Standard SFT: Classic conversation format
- Interleaved SFT: Code + state prediction (Lucas Beyer style)
- PRM: Process reward model samples

Usage:
    uv run python -m src.training.prepare_finetune_data \
        --input data/episodes/train.jsonl \
        --format sft-standard \
        --output data/training/train_sft.jsonl
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from csv_spec import ProcessReportDict, hash_artifact
from pydantic import TypeAdapter, ValidationError


_PROCESS_REPORT_ADAPTER = TypeAdapter(ProcessReportDict)


def load_episodes(input_path: str, verified_only: bool = True) -> list[dict[str, Any]]:
    """Load episodes from JSONL file."""
    episodes = []
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                episode = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                continue

            if verified_only and not episode.get("verified", False):
                continue

            episodes.append(episode)

    return episodes


def build_system_prompt(episode: dict[str, Any]) -> str:
    """Build system prompt from episode metadata."""
    csv_source = episode.get("csv_source", "data.csv")
    return f"""You are a data analysis assistant. You have access to a pandas DataFrame loaded from '{csv_source}'.

Available functions:
- submit(answer): Submit your final answer
- hook(value, name="..."): Checkpoint intermediate values

Write Python code to answer the question. Always call submit() with your final answer."""


def to_sft_standard(episode: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert episode to standard SFT format.

    Format: System prompt + alternating user/assistant messages.
    User messages contain question (first) or execution output (subsequent).
    Assistant messages contain reasoning + code.
    """
    question = episode.get("question", {})
    gold_trace = episode.get("gold_trace") or episode.get("teacher_gold_trace", {})
    turns = gold_trace.get("turns", [])

    if not turns:
        return None

    question_text = question.get("question_text", "")
    hint = question.get("hint")

    messages = []

    messages.append({"role": "system", "content": build_system_prompt(episode)})

    first_user_content = question_text
    if hint:
        first_user_content += f"\n\nHint: {hint}"

    messages.append({"role": "user", "content": first_user_content})

    for turn in turns:
        reasoning = turn.get("reasoning", "")
        code = turn.get("code", "")
        execution = turn.get("execution", {})
        stdout = execution.get("stdout", "")
        stderr = execution.get("stderr", "")

        assistant_content = ""
        if reasoning:
            assistant_content += reasoning + "\n\n"
        if code:
            assistant_content += f"```python\n{code}\n```"

        messages.append({"role": "assistant", "content": assistant_content.strip()})

        user_content = ""
        if stdout:
            user_content += f"[stdout]:\n{stdout}"
        if stderr:
            if user_content:
                user_content += "\n"
            user_content += f"[stderr]:\n{stderr}"

        if user_content:
            messages.append({"role": "user", "content": user_content.strip()})

    if messages[-1]["role"] == "user":
        messages.pop()

    return {"messages": messages}


def to_sft_interleaved(episode: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert episode to interleaved SFT format (state prediction).

    Format: After each code block, model predicts interpreter state
    before seeing actual output. Trains model to reason about execution.
    """
    question = episode.get("question", {})
    gold_trace = episode.get("gold_trace") or episode.get("teacher_gold_trace", {})
    turns = gold_trace.get("turns", [])

    if not turns:
        return None

    question_text = question.get("question_text", "")
    hint = question.get("hint")

    messages = []

    messages.append(
        {
            "role": "system",
            "content": build_system_prompt(episode)
            + "\n\nAfter writing code, predict what values will be computed before seeing the output.",
        }
    )

    first_user_content = question_text
    if hint:
        first_user_content += f"\n\nHint: {hint}"

    messages.append({"role": "user", "content": first_user_content})

    for turn in turns:
        reasoning = turn.get("reasoning", "")
        code = turn.get("code", "")
        execution = turn.get("execution", {})
        hooks = execution.get("hooks", [])
        stdout = execution.get("stdout", "")
        submitted = execution.get("submitted_answer")

        assistant_content = ""
        if reasoning:
            assistant_content += reasoning + "\n\n"
        if code:
            assistant_content += f"```python\n{code}\n```"

        messages.append({"role": "assistant", "content": assistant_content.strip()})

        if hooks:
            messages.append(
                {"role": "user", "content": "Predict the intermediate values:"}
            )

            predictions = []
            for hook in hooks:
                var_name = hook.get("variable_name", "result")
                value = hook.get("value")
                predictions.append(f"{var_name} = {json.dumps(value)}")

            messages.append({"role": "assistant", "content": "\n".join(predictions)})

        if submitted is not None:
            messages.append(
                {"role": "user", "content": "Predict the submitted answer:"}
            )
            messages.append({"role": "assistant", "content": json.dumps(submitted)})

        if stdout:
            messages.append({"role": "user", "content": f"[actual output]:\n{stdout}"})

    if messages[-1]["role"] == "user":
        messages.pop()

    return {"messages": messages}


def to_prm_samples(
    episode: dict[str, Any],
    include_heuristic_hooks: bool = False,
) -> list[dict[str, Any]]:
    """
    Convert episode to PRM (Process Reward Model) samples.

    Terminal verifier labels are exported by default. Hook heuristics are
    diagnostic candidates and require explicit opt-in.
    """
    question = episode.get("question", {})
    gold_trace = episode.get("gold_trace") or episode.get("teacher_gold_trace", {})
    episode_id = episode.get("episode_id", "<unknown>")
    if "process_report" not in episode:
        raise ValueError(
            f"Episode {episode_id!r} is missing required process_report; "
            "regenerate episodes before PRM conversion"
        )
    try:
        process_report = _PROCESS_REPORT_ADAPTER.validate_python(
            episode["process_report"], strict=True
        )
    except ValidationError as exc:
        raise ValueError(
            f"Episode {episode_id!r} has malformed process_report; "
            "regenerate episodes before PRM conversion"
        ) from exc
    _validate_process_report_semantics(
        process_report=process_report,
        gold_trace=gold_trace,
        consistency_traces=episode.get("consistency_traces", []),
        episode_id=episode_id,
    )
    process_steps = process_report["steps"]
    turns = gold_trace["turns"]

    if not turns or not process_steps:
        return []

    question_text = question.get("question_text", "")
    hint = question.get("hint", "")

    samples = []
    prefix_messages = []

    prefix_messages.append({"role": "system", "content": build_system_prompt(episode)})

    user_content = question_text
    if hint:
        user_content += f"\n\nHint: {hint}"
    prefix_messages.append({"role": "user", "content": user_content})

    steps_by_turn: dict[int, list[dict[str, Any]]] = {}
    for step in process_steps:
        if step.get("label") is None:
            continue
        label_kind = step.get("label_kind")
        if label_kind == "heuristic" and not include_heuristic_hooks:
            continue
        if label_kind not in {"verified", "heuristic"}:
            continue
        steps_by_turn.setdefault(int(step.get("turn_index", 0)), []).append(step)

    for turn_idx, turn in enumerate(turns):
        code = turn.get("code", "")
        execution = turn.get("execution", {})
        stdout = execution.get("stdout", "")

        for step in steps_by_turn.get(turn_idx, []):
            samples.append(
                {
                    "prefix": json.dumps(prefix_messages),
                    "turn_index": turn_idx,
                    "step_index": step.get("step_index"),
                    "step_type": step.get("step_type"),
                    "code_line": step.get("code_line", ""),
                    "variable_name": step.get("variable_name"),
                    "semantic_role": step.get("semantic_role"),
                    "value": step.get("value"),
                    "value_hash": step.get("value_hash"),
                    "label": step.get("label"),
                    "label_kind": step.get("label_kind"),
                    "label_source": step.get("label_source"),
                    "evidence": step.get("evidence", {}),
                    "episode_id": episode.get("episode_id"),
                }
            )

        prefix_messages.append(
            {"role": "assistant", "content": f"```python\n{code}\n```"}
        )
        if stdout:
            prefix_messages.append({"role": "user", "content": f"[stdout]:\n{stdout}"})

    return samples


def _validate_process_report_semantics(
    *,
    process_report: dict[str, Any],
    gold_trace: Any,
    consistency_traces: Any,
    episode_id: Any,
) -> None:
    prefix = f"Episode {episode_id!r} has inconsistent process_report"

    if not isinstance(gold_trace, dict):
        raise ValueError(f"{prefix}: gold_trace must be an object")
    turns = gold_trace.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"{prefix}: gold_trace.turns must be a list")
    trace_success = gold_trace.get("success")
    if not isinstance(trace_success, bool):
        raise ValueError(f"{prefix}: gold_trace.success must be a boolean")
    if not isinstance(consistency_traces, list):
        raise ValueError(f"{prefix}: consistency_traces must be a list")

    successful_consistency_hashes: list[str | None] = []
    for trace_index, trace in enumerate(consistency_traces):
        if not isinstance(trace, dict) or not isinstance(trace.get("success"), bool):
            raise ValueError(
                f"{prefix}: consistency_traces[{trace_index}] must contain a "
                "boolean success field"
            )
        if trace["success"]:
            successful_consistency_hashes.append(trace.get("final_answer_hash"))

    expected_observations: list[tuple[str, int, int | None, Any]] = []
    submitted_step_indices: list[int] = []
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(
                f"{prefix}: gold_trace.turns[{turn_index}] must be an object"
            )
        if turn.get("turn_index") != turn_index:
            raise ValueError(
                f"{prefix}: gold_trace.turns[{turn_index}].turn_index must be "
                f"{turn_index}"
            )
        execution = turn.get("execution")
        if not isinstance(execution, dict):
            raise ValueError(
                f"{prefix}: gold_trace.turns[{turn_index}].execution must be an object"
            )
        hooks = execution.get("hooks")
        if not isinstance(hooks, list):
            raise ValueError(
                f"{prefix}: gold_trace.turns[{turn_index}].execution.hooks must be "
                "a list"
            )
        for hook_index, hook in enumerate(hooks):
            if not isinstance(hook, dict):
                raise ValueError(
                    f"{prefix}: gold_trace.turns[{turn_index}].execution.hooks"
                    f"[{hook_index}] must be an object"
                )
            expected_observations.append(("hook", turn_index, hook_index, hook))

        submitted_answer = execution.get("submitted_answer")
        if submitted_answer is not None:
            expected_observations.append(("submit", turn_index, None, submitted_answer))
            submitted_step_indices.append(len(expected_observations) - 1)

    if trace_success and not submitted_step_indices:
        raise ValueError(
            f"{prefix}: successful gold_trace must contain a submitted answer"
        )

    process_steps = process_report["steps"]
    if len(process_steps) != len(expected_observations):
        raise ValueError(
            f"{prefix}: steps must cover trace hooks and submissions one-to-one; "
            f"expected {len(expected_observations)}, got {len(process_steps)}"
        )

    accepted_step_index = (
        submitted_step_indices[-1] if trace_success and submitted_step_indices else None
    )

    for expected_step_index, (step, expected) in enumerate(
        zip(process_steps, expected_observations, strict=True)
    ):
        expected_step_type, expected_turn_index, expected_hook_index, source_value = (
            expected
        )
        step_index = step["step_index"]
        if step_index != expected_step_index:
            raise ValueError(
                f"{prefix}: steps[{expected_step_index}].step_index must be "
                f"{expected_step_index}, got {step_index}"
            )

        turn_index = step["turn_index"]
        if turn_index != expected_turn_index:
            raise ValueError(
                f"{prefix}: steps[{step_index}].turn_index must be "
                f"{expected_turn_index}, got {turn_index}"
            )

        step_type = step["step_type"]
        if step_type != expected_step_type:
            raise ValueError(
                f"{prefix}: steps[{step_index}].step_type must be "
                f"{expected_step_type!r}, got {step_type!r}"
            )
        label_kind = step["label_kind"]
        label = step["label"]

        if label_kind == "verified" and step_type != "submit":
            raise ValueError(
                f"{prefix}: steps[{step_index}] verified labels are only valid "
                "on submit steps"
            )
        if label_kind == "unlabeled" and label is not None:
            raise ValueError(
                f"{prefix}: steps[{step_index}] unlabeled steps must not have a "
                "numeric label"
            )
        if label_kind != "unlabeled" and label is None:
            raise ValueError(
                f"{prefix}: steps[{step_index}] labeled steps require a numeric label"
            )
        if label is not None and not math.isfinite(label):
            raise ValueError(
                f"{prefix}: steps[{step_index}].label must be a finite number"
            )

        hook_index = step["hook_index"]
        if step_type == "submit":
            if hook_index != expected_hook_index:
                raise ValueError(
                    f"{prefix}: steps[{step_index}] submit steps must not have a "
                    "hook_index"
                )
            _validate_submit_observation(
                step=step,
                submitted_answer=source_value,
                accepted=step_index == accepted_step_index,
                gold_trace=gold_trace,
                trace_success=trace_success,
                successful_consistency_hashes=successful_consistency_hashes,
                prefix=prefix,
            )
            continue

        if hook_index != expected_hook_index:
            raise ValueError(
                f"{prefix}: steps[{step_index}].hook_index must be "
                f"{expected_hook_index}, got {hook_index}"
            )
        _validate_hook_observation(
            step=step,
            hook=source_value,
            prefix=prefix,
        )

    expected_summary = _summarize_process_steps(process_steps)
    if process_report["summary"] != expected_summary:
        raise ValueError(
            f"{prefix}: summary does not match the report steps; "
            f"expected {expected_summary!r}"
        )


def _validate_hook_observation(
    *,
    step: dict[str, Any],
    hook: dict[str, Any],
    prefix: str,
) -> None:
    expected_fields = {
        "code_line": hook.get("code_line", ""),
        "variable_name": hook.get("variable_name"),
        "value": hook.get("value"),
        "value_hash": hook.get("value_hash"),
        "description": hook.get("description"),
        "depends_on": hook.get("depends_on", []),
    }
    for field, expected_value in expected_fields.items():
        if step[field] != expected_value:
            raise ValueError(
                f"{prefix}: steps[{step['step_index']}].{field} does not match "
                "the source hook"
            )


def _validate_submit_observation(
    *,
    step: dict[str, Any],
    submitted_answer: Any,
    accepted: bool,
    gold_trace: dict[str, Any],
    trace_success: bool,
    successful_consistency_hashes: list[str | None],
    prefix: str,
) -> None:
    step_index = step["step_index"]
    if step["value"] != submitted_answer:
        raise ValueError(
            f"{prefix}: steps[{step_index}].value does not match the source submission"
        )

    if accepted:
        if gold_trace.get("final_answer") != submitted_answer:
            raise ValueError(
                f"{prefix}: accepted submission does not match gold_trace.final_answer"
            )
        expected_hash = gold_trace.get("final_answer_hash") or hash_artifact(
            submitted_answer
        )
    else:
        expected_hash = hash_artifact(submitted_answer)
    if step["value_hash"] != expected_hash:
        raise ValueError(
            f"{prefix}: steps[{step_index}].value_hash does not match the source "
            "submission"
        )

    expected_identity = {
        "code_line": "submit(...)",
        "variable_name": "answer",
        "description": "Final submitted answer",
        "depends_on": [],
        "semantic_role": "final_answer",
    }
    for field, expected_value in expected_identity.items():
        if step[field] != expected_value:
            raise ValueError(
                f"{prefix}: steps[{step_index}].{field} is not canonical for a "
                "submission"
            )

    evidence = step["evidence"]
    required_evidence = {
        "final_verified",
        "trace_success",
        "consensus_matches",
        "consensus_total",
    }
    missing_evidence = sorted(required_evidence.difference(evidence))
    if missing_evidence:
        raise ValueError(
            f"{prefix}: steps[{step_index}].evidence is missing "
            f"{', '.join(missing_evidence)}"
        )
    if evidence["trace_success"] is not trace_success:
        raise ValueError(
            f"{prefix}: steps[{step_index}].evidence.trace_success does not match "
            "gold_trace.success"
        )

    expected_consensus_matches = sum(
        1
        for answer_hash in successful_consistency_hashes
        if answer_hash == expected_hash
    )
    if evidence["consensus_matches"] != expected_consensus_matches:
        raise ValueError(
            f"{prefix}: steps[{step_index}].evidence.consensus_matches must be "
            f"{expected_consensus_matches}"
        )
    if evidence["consensus_total"] != len(successful_consistency_hashes):
        raise ValueError(
            f"{prefix}: steps[{step_index}].evidence.consensus_total must be "
            f"{len(successful_consistency_hashes)}"
        )

    if not accepted:
        expected_label = (None, "unlabeled", "rejected_submission", None)
    elif evidence["final_verified"] is None:
        expected_label = (None, "unlabeled", "verifier_unavailable", None)
    else:
        verdict = evidence["final_verified"]
        expected_label = (
            1.0 if verdict else 0.0,
            "verified",
            "terminal_verifier",
            verdict,
        )

    actual_label = (
        step["label"],
        step["label_kind"],
        step["label_source"],
        evidence["final_verified"],
    )
    if actual_label != expected_label:
        status = "accepted" if accepted else "rejected"
        raise ValueError(
            f"{prefix}: steps[{step_index}] has label metadata inconsistent with "
            f"the {status} submission"
        )


def _summarize_process_steps(
    steps: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "total_steps": len(steps),
        "labeled_steps": sum(1 for step in steps if step["label"] is not None),
        "verified_steps": sum(1 for step in steps if step["label_kind"] == "verified"),
        "heuristic_steps": sum(
            1 for step in steps if step["label_kind"] == "heuristic"
        ),
        "unlabeled_steps": sum(
            1 for step in steps if step["label_kind"] == "unlabeled"
        ),
        "positive_steps": sum(1 for step in steps if step["label"] == 1.0),
        "negative_steps": sum(1 for step in steps if step["label"] == 0.0),
    }


def convert_episodes(
    episodes: list[dict[str, Any]],
    format_type: str,
    include_heuristic_hooks: bool = False,
) -> list[dict[str, Any]]:
    """Convert episodes to specified training format."""
    results = []

    for episode in episodes:
        if format_type == "sft-standard":
            result = to_sft_standard(episode)
            if result:
                results.append(result)

        elif format_type == "sft-interleaved":
            result = to_sft_interleaved(episode)
            if result:
                results.append(result)

        elif format_type == "prm":
            samples = to_prm_samples(
                episode,
                include_heuristic_hooks=include_heuristic_hooks,
            )
            results.extend(samples)

        else:
            raise ValueError(f"Unknown format: {format_type}")

    return results


def save_jsonl(data: list[dict[str, Any]], output_path: str) -> None:
    """Save data to JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert episodes to training formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input episodes JSONL file"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sft-standard", "sft-interleaved", "prm"],
        default="sft-standard",
        help="Output format: sft-standard, sft-interleaved, or prm",
    )
    parser.add_argument("--output", type=str, help="Path to output JSONL file")
    parser.add_argument(
        "--include-unverified", action="store_true", help="Include unverified episodes"
    )
    parser.add_argument(
        "--include-heuristic-hooks",
        action="store_true",
        help=(
            "Include non-verified hook heuristics in PRM output. "
            "Disabled by default because these are experimental baselines."
        ),
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/training/{Path(args.input).stem}_{args.format.replace('-', '_')}.jsonl"

    print(f"Loading episodes from {args.input}...")
    episodes = load_episodes(args.input, verified_only=not args.include_unverified)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("Warning: No episodes to process!")
        return

    print(f"Converting to {args.format} format...")
    results = convert_episodes(
        episodes,
        args.format,
        include_heuristic_hooks=args.include_heuristic_hooks,
    )
    print(f"Generated {len(results)} training samples")

    save_jsonl(results, args.output)

    print(f"\nDone! Output: {args.output}")


if __name__ == "__main__":
    main()
