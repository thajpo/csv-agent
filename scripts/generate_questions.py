#!/usr/bin/env python3
"""
Generate questions from a CSV dataset.

Usage:
    uv run python scripts/generate_questions.py
    uv run python scripts/generate_questions.py --csv path/to/other.csv
"""

import argparse
import asyncio
import json
from pathlib import Path
import yaml

from src.datagen.question_gen import explore_and_generate_questions


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Generate questions from a CSV dataset")
    parser.add_argument("--csv", default=config.get("csv"), help="Path to CSV file")
    parser.add_argument("--model", default=config.get("question_gen_model"), help="Model to use")
    parser.add_argument("--max-turns", type=int, default=config.get("question_gen_max_turns", 20), help="Max exploration turns")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    # Create output directory based on dataset name
    csv_path = Path(args.csv)
    dataset_name = csv_path.stem
    output_path = Path(args.output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating questions from: {args.csv}")
    print(f"Using model: {args.model}")
    print(f"Output directory: {output_path}")
    print()

    # Run question generation
    questions, trace = asyncio.run(
        explore_and_generate_questions(
            csv_path=str(csv_path),
            model=args.model,
            max_turns=args.max_turns,
            output_dir=str(output_path),
        )
    )

    # Save questions
    questions_file = output_path / "questions.json"
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\n✓ Generated {len(questions)} questions")
    print(f"✓ Saved to: {questions_file}")


if __name__ == "__main__":
    main()
