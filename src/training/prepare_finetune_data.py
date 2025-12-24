"""
Prepare fine-tuning data for OpenAI and Anthropic APIs.

Extracts conversation_for_sft from episodes and formats for API upload.

Usage:
    uv run python -m src.training.prepare_finetune_data \
        --input episodes/train.jsonl \
        --provider openai \
        --output data/episodes/train_openai.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_episodes(input_path: str, verified_only: bool = True) -> list[dict[str, Any]]:
    """Load episodes from JSONL file.

    Args:
        input_path: Path to episodes JSONL file
        verified_only: If True, only load verified episodes (default: True)

    Returns:
        List of episode dictionaries
    """
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

            # Filter by verification status
            if verified_only and not episode.get("verified", False):
                continue

            episodes.append(episode)

    return episodes


def convert_to_api_format(
    episodes: list[dict[str, Any]],
    provider: str = "openai"
) -> list[dict[str, Any]]:
    """Convert episodes to API fine-tuning format.

    Extracts conversation_for_sft field and formats for the specified provider.

    Args:
        episodes: List of episode dictionaries
        provider: Either "openai" or "anthropic"

    Returns:
        List of formatted training examples

    Raises:
        ValueError: If provider is not supported or conversation_for_sft is missing
    """
    if provider not in ("openai", "anthropic"):
        raise ValueError(f"Unsupported provider: {provider}. Must be 'openai' or 'anthropic'.")

    formatted_data = []

    for episode in episodes:
        # Extract conversation_for_sft
        conversation = episode.get("conversation_for_sft")
        if not conversation:
            print(f"Warning: Episode {episode.get('episode_id', 'unknown')} missing conversation_for_sft, skipping")
            continue

        system_prompt = conversation.get("system_prompt", "")
        messages = conversation.get("messages", [])

        if not messages:
            print(f"Warning: Episode {episode.get('episode_id', 'unknown')} has empty messages, skipping")
            continue

        # Format based on provider
        if provider == "openai":
            # OpenAI format: {"messages": [{"role": "system", "content": "..."}, ...]}
            formatted_messages = []

            # Add system prompt as first message if present
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Add conversation messages
            formatted_messages.extend(messages)

            formatted_data.append({
                "messages": formatted_messages
            })

        elif provider == "anthropic":
            # Anthropic format: {"system": "...", "messages": [...]}
            # Note: Anthropic separates system prompt from messages
            formatted_data.append({
                "system": system_prompt,
                "messages": messages
            })

    return formatted_data


def save_jsonl(data: list[dict[str, Any]], output_path: str) -> None:
    """Save data to JSONL file.

    Args:
        data: List of dictionaries to save
        output_path: Path to output JSONL file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(data)} training examples to {output_path}")


def main():
    """CLI entry point for preparing fine-tuning data."""
    parser = argparse.ArgumentParser(
        description="Prepare fine-tuning data for OpenAI or Anthropic APIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input episodes JSONL file (e.g., episodes/train.jsonl)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="API provider format (openai or anthropic)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output JSONL file (defaults to data/episodes/train_{provider}.jsonl)"
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include unverified episodes (default: verified only)"
    )

    args = parser.parse_args()

    # Set default output path if not specified
    if args.output is None:
        args.output = f"data/episodes/train_{args.provider}.jsonl"

    print(f"Loading episodes from {args.input}...")
    episodes = load_episodes(args.input, verified_only=not args.include_unverified)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("Warning: No episodes to process!")
        return

    print(f"Converting to {args.provider} format...")
    formatted_data = convert_to_api_format(episodes, provider=args.provider)
    print(f"Converted {len(formatted_data)} episodes to {args.provider} format")

    print(f"Saving to {args.output}...")
    save_jsonl(formatted_data, args.output)

    print("\nDone!")
    print(f"Summary:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Provider: {args.provider}")
    print(f"  Training examples: {len(formatted_data)}")


if __name__ == "__main__":
    main()
