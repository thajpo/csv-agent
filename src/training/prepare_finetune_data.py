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
from typing import Any


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


def to_prm_samples(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert episode to PRM (Process Reward Model) samples.

    Format: Each hook/step becomes a scored sample.
    Verified episodes get positive labels, unverified get negative.
    """
    question = episode.get("question", {})
    gold_trace = episode.get("gold_trace") or episode.get("teacher_gold_trace", {})
    turns = gold_trace.get("turns", [])
    verified = episode.get("verified", False)

    if not turns:
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

    for turn_idx, turn in enumerate(turns):
        code = turn.get("code", "")
        execution = turn.get("execution", {})
        hooks = execution.get("hooks", [])
        submitted = execution.get("submitted_answer")
        stdout = execution.get("stdout", "")

        for hook in hooks:
            samples.append(
                {
                    "prefix": json.dumps(prefix_messages),
                    "turn_index": turn_idx,
                    "step_type": "hook",
                    "code_line": hook.get("code_line", ""),
                    "variable_name": hook.get("variable_name"),
                    "value": hook.get("value"),
                    "value_hash": hook.get("value_hash"),
                    "label": 1.0 if verified else 0.0,
                    "episode_id": episode.get("episode_id"),
                }
            )

        if submitted is not None:
            samples.append(
                {
                    "prefix": json.dumps(prefix_messages),
                    "turn_index": turn_idx,
                    "step_type": "submit",
                    "code_line": "submit(...)",
                    "variable_name": "answer",
                    "value": submitted,
                    "value_hash": gold_trace.get("final_answer_hash"),
                    "label": 1.0 if verified else 0.0,
                    "episode_id": episode.get("episode_id"),
                }
            )

        prefix_messages.append(
            {"role": "assistant", "content": f"```python\n{code}\n```"}
        )
        if stdout:
            prefix_messages.append({"role": "user", "content": f"[stdout]:\n{stdout}"})

    return samples


def convert_episodes(
    episodes: list[dict[str, Any]],
    format_type: str,
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
            samples = to_prm_samples(episode)
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
    results = convert_episodes(episodes, args.format)
    print(f"Generated {len(results)} training samples")

    save_jsonl(results, args.output)

    print(f"\nDone! Output: {args.output}")


if __name__ == "__main__":
    main()
