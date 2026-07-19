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
from pathlib import Path
from typing import Any, cast

from csv_spec import ProcessReportDict, TraceDict
from pydantic import TypeAdapter, ValidationError

from src.datagen.process_report import (
    EpisodeSource,
    build_process_report,
    validate_trace_submissions,
)
from src.datagen.shared.verification import (
    derive_ground_truth_verification,
    derive_llm_verification,
    trace_answer_hash,
    validate_float_tolerance,
)


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
        episode=episode,
        process_report=process_report,
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
    episode: dict[str, Any],
    process_report: dict[str, Any],
    episode_id: Any,
) -> None:
    prefix = f"Episode {episode_id!r} has inconsistent process_report"
    source, gold_trace, consistency_traces, verifier_verdict = _canonical_report_inputs(
        episode=episode, prefix=prefix
    )
    expected_report = build_process_report(
        source=source,
        gold_trace=gold_trace,
        consistency_traces=consistency_traces,
        verifier_verdict=verifier_verdict,
    )
    if not _canonical_json_equal(process_report, expected_report):
        raise ValueError(
            f"{prefix}: report does not match the canonical report derived from "
            "episode provenance; regenerate episodes before PRM conversion"
        )


def _canonical_json_equal(left: Any, right: Any) -> bool:
    options = {
        "allow_nan": True,
        "ensure_ascii": False,
        "separators": (",", ":"),
        "sort_keys": True,
    }
    return json.dumps(left, **options) == json.dumps(right, **options)


def _canonical_report_inputs(
    *, episode: dict[str, Any], prefix: str
) -> tuple[EpisodeSource, TraceDict, list[TraceDict], bool | None]:
    source = episode.get("source")
    if source not in {"llm_gen", "template", "procedural"}:
        raise ValueError(f"{prefix}: source is missing or unsupported")

    question = episode.get("question")
    if not isinstance(question, dict):
        raise ValueError(f"{prefix}: question must be an object")
    question_source = question.get("source")
    if question_source is not None and question_source != source:
        raise ValueError(f"{prefix}: question.source does not match episode source")

    verified = episode.get("verified")
    if type(verified) is not bool:
        raise ValueError(f"{prefix}: verified must be a boolean")

    gold_trace = _validate_trace(
        trace=episode.get("gold_trace"),
        path="gold_trace",
        prefix=prefix,
    )
    raw_consistency_traces = episode.get("consistency_traces")
    if not isinstance(raw_consistency_traces, list):
        raise ValueError(f"{prefix}: consistency_traces must be a list")
    consistency_traces = [
        _validate_trace(
            trace=trace,
            path=f"consistency_traces[{trace_index}]",
            prefix=prefix,
        )
        for trace_index, trace in enumerate(raw_consistency_traces)
    ]

    triangulation = episode.get("triangulation")
    if not isinstance(triangulation, dict):
        raise ValueError(f"{prefix}: triangulation must be an object")
    required_fields = {
        "n_consistency_runs",
        "n_consistency_succeeded",
        "majority_answer_hash",
        "majority_count",
        "gold_matches_majority",
        "float_tolerance",
    }
    if not required_fields.issubset(triangulation):
        raise ValueError(f"{prefix}: triangulation provenance is incomplete")

    n_runs = triangulation["n_consistency_runs"]
    n_succeeded = triangulation["n_consistency_succeeded"]
    majority_count = triangulation["majority_count"]
    majority_hash = triangulation["majority_answer_hash"]
    gold_matches = triangulation["gold_matches_majority"]
    float_tolerance = triangulation["float_tolerance"]
    if type(n_runs) is not int or type(n_succeeded) is not int:
        raise ValueError(f"{prefix}: triangulation counts must be integers")
    if type(majority_count) is not int or majority_count < 0:
        raise ValueError(f"{prefix}: triangulation.majority_count is invalid")
    if majority_hash is not None and (
        not isinstance(majority_hash, str) or not majority_hash
    ):
        raise ValueError(f"{prefix}: triangulation.majority_answer_hash is invalid")
    if type(gold_matches) is not bool:
        raise ValueError(
            f"{prefix}: triangulation.gold_matches_majority must be a boolean"
        )
    try:
        float_tolerance = validate_float_tolerance(float_tolerance)
    except ValueError as exc:
        raise ValueError(f"{prefix}: triangulation.float_tolerance is invalid") from exc

    succeeded = sum(1 for trace in consistency_traces if trace["success"])
    if n_runs != len(consistency_traces) or n_succeeded != succeeded:
        raise ValueError(f"{prefix}: triangulation counts do not match source traces")

    if source == "llm_gen":
        evidence = derive_llm_verification(
            gold_trace=gold_trace,
            consistency_traces=consistency_traces,
            float_tolerance=float_tolerance,
        )
    else:
        if consistency_traces:
            raise ValueError(
                f"{prefix}: ground-truth episodes cannot use consistency traces"
            )
        try:
            evidence = derive_ground_truth_verification(
                question=question,
                gold_trace=gold_trace,
                float_tolerance=float_tolerance,
            )
        except ValueError as exc:
            raise ValueError(f"{prefix}: {exc}") from exc

    if (
        majority_hash != evidence.majority_answer_hash
        or majority_count != evidence.majority_count
        or gold_matches != (evidence.verdict is True)
        or verified != (evidence.verdict is True)
    ):
        raise ValueError(
            f"{prefix}: triangulation metadata disagrees with source traces"
        )

    verifier_verdict = evidence.verdict

    return (
        cast(EpisodeSource, source),
        gold_trace,
        consistency_traces,
        verifier_verdict,
    )


def _validate_trace(*, trace: Any, path: str, prefix: str) -> TraceDict:
    if not isinstance(trace, dict):
        raise ValueError(f"{prefix}: {path} must be an object")
    required_fields = {"turns", "final_answer", "success"}
    if not required_fields.issubset(trace):
        raise ValueError(f"{prefix}: {path} provenance is incomplete")
    if type(trace["success"]) is not bool:
        raise ValueError(f"{prefix}: {path}.success must be a boolean")
    final_answer_hash = trace.get("final_answer_hash")
    if final_answer_hash is not None and not isinstance(final_answer_hash, str):
        raise ValueError(f"{prefix}: {path}.final_answer_hash is invalid")
    if not isinstance(trace["turns"], list):
        raise ValueError(f"{prefix}: {path}.turns must be a list")

    for turn_index, turn in enumerate(trace["turns"]):
        turn_path = f"{path}.turns[{turn_index}]"
        if not isinstance(turn, dict) or turn.get("turn_index") != turn_index:
            raise ValueError(f"{prefix}: {turn_path} is not in canonical order")
        if not isinstance(turn.get("code"), str):
            raise ValueError(f"{prefix}: {turn_path}.code must be a string")
        execution = turn.get("execution")
        if not isinstance(execution, dict):
            raise ValueError(f"{prefix}: {turn_path}.execution must be an object")
        if type(execution.get("success")) is not bool:
            raise ValueError(
                f"{prefix}: {turn_path}.execution.success must be a boolean"
            )
        if "submitted_answer" not in execution:
            raise ValueError(
                f"{prefix}: {turn_path}.execution.submitted_answer is required"
            )
        hooks = execution.get("hooks")
        if not isinstance(hooks, list):
            raise ValueError(f"{prefix}: {turn_path}.execution.hooks must be a list")
        for hook_index, hook in enumerate(hooks):
            hook_path = f"{turn_path}.execution.hooks[{hook_index}]"
            if not isinstance(hook, dict):
                raise ValueError(f"{prefix}: {hook_path} must be an object")
            hook_fields = {
                "code_line",
                "variable_name",
                "value",
                "value_hash",
                "description",
                "depends_on",
                "event_line",
                "event_provenance_reason",
            }
            if not hook_fields.issubset(hook):
                raise ValueError(f"{prefix}: {hook_path} provenance is incomplete")
            if not isinstance(hook["code_line"], str):
                raise ValueError(f"{prefix}: {hook_path}.code_line must be a string")
            if hook["variable_name"] is not None and not isinstance(
                hook["variable_name"], str
            ):
                raise ValueError(f"{prefix}: {hook_path}.variable_name is invalid")
            if not isinstance(hook["value_hash"], str):
                raise ValueError(f"{prefix}: {hook_path}.value_hash must be a string")
            if not isinstance(hook["depends_on"], list) or not all(
                isinstance(dep, str) for dep in hook["depends_on"]
            ):
                raise ValueError(f"{prefix}: {hook_path}.depends_on must be a list")
            if hook["description"] is not None and not isinstance(
                hook["description"], str
            ):
                raise ValueError(f"{prefix}: {hook_path}.description is invalid")
            event_line = hook["event_line"]
            if event_line is not None and (
                type(event_line) is not int
                or event_line < 1
                or event_line > len(turn["code"].splitlines())
            ):
                raise ValueError(f"{prefix}: {hook_path}.event_line is invalid")
            provenance_reason = hook["event_provenance_reason"]
            if provenance_reason is not None and (
                not isinstance(provenance_reason, str) or not provenance_reason
            ):
                raise ValueError(
                    f"{prefix}: {hook_path}.event_provenance_reason is invalid"
                )
            if provenance_reason is None and event_line is None:
                raise ValueError(
                    f"{prefix}: {hook_path} authenticated provenance is incomplete"
                )
    try:
        canonical_answer_hash = trace_answer_hash(cast(TraceDict, trace))
        validate_trace_submissions(cast(TraceDict, trace), path=path)
    except ValueError as exc:
        raise ValueError(f"{prefix}: {exc}") from exc

    if trace["success"]:
        if canonical_answer_hash is None:
            raise ValueError(f"{prefix}: {path} final answer hash is unavailable")
    elif trace["final_answer"] is not None:
        raise ValueError(f"{prefix}: unsuccessful {path} has a final answer")

    return cast(TraceDict, trace)


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
