#!/usr/bin/env python3
"""
Generate training data by validating questions via teacher triangulation.

Usage:
    uv run python scripts/generate_training_data.py --questions outputs/data/questions.json
"""

import argparse
import asyncio
import json
from pathlib import Path
import yaml

from src.datagen.teacher import batch_triangulate
from src.core.prompts import generate_data_overview


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Generate training data via teacher triangulation")
    parser.add_argument("--csv", default=config.get("csv"), help="Path to CSV file")
    parser.add_argument("--questions", required=True, help="Path to questions.json")
    parser.add_argument("--model", default=config.get("teacher_model"), help="Model to use")
    parser.add_argument("--n-consistency", type=int, default=config.get("n_consistency", 3), help="Number of consistency traces")
    parser.add_argument("--max-turns", type=int, default=config.get("max_turns", 10), help="Max turns per trace")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    # Load questions
    with open(args.questions) as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from: {args.questions}")
    print(f"Using model: {args.model}")
    print(f"Consistency traces: {args.n_consistency}")
    print()

    # Setup output directory
    csv_path = Path(args.csv)
    dataset_name = csv_path.stem
    output_path = Path(args.output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate data overview for prompts
    data_overview = generate_data_overview(str(csv_path))

    # Run triangulation
    results = asyncio.run(
        batch_triangulate(
            csv_path=str(csv_path),
            questions=questions,
            model=args.model,  # Required positional arg (3rd)
            n_consistency=args.n_consistency,
            data_overview=data_overview,
            max_turns=args.max_turns,
        )
    )

    # Process results
    all_traces = []
    verified_traces = []

    for q_dict, gold_trace, gold_conv, sys_prompt, consistency_results, verified in results:
        trace_record = {
            "question": q_dict,
            "gold_trace": {
                "code_cells": gold_trace.code_cells,
                "final_answer": gold_trace.final_answer,
                "execution_success": gold_trace.execution_success,
            },
            "conversation": gold_conv,
            "system_prompt": sys_prompt,
            "verified": verified,
        }
        all_traces.append(trace_record)

        if verified:
            verified_traces.append(trace_record)

    # Save all traces
    traces_file = output_path / "traces.json"
    with open(traces_file, "w") as f:
        json.dump(all_traces, f, indent=2)

    # Save verified traces (training data)
    training_file = output_path / "training_data.json"
    with open(training_file, "w") as f:
        json.dump(verified_traces, f, indent=2)

    print(f"\n✓ Processed {len(results)} questions")
    print(f"✓ Verified: {len(verified_traces)}/{len(results)}")
    print(f"✓ All traces saved to: {traces_file}")
    print(f"✓ Training data saved to: {training_file}")


if __name__ == "__main__":
    main()
