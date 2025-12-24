"""
CLI script for evaluating CSV agent models.

Usage:
    uv run python -m scripts.evaluate_model \\
        --model openai/gpt-4o-mini \\
        --episodes episodes/test.jsonl \\
        --output eval_results/report.md

    uv run python -m scripts.evaluate_model \\
        --model openai/gpt-4o-mini \\
        --episodes data/fixtures/mock_episodes.jsonl \\
        --csv data/mock/data.csv \\
        --format json \\
        --output eval_results/report.json
"""

import argparse
import asyncio
import sys
from pathlib import Path

from src.eval.evaluator import Evaluator
from src.eval.report import generate_report


async def main():
    """Main entry point for evaluation script."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Evaluate CSV agent model on test episodes"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., 'openai/gpt-4o-mini')",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
        help="Path to episodes JSONL file (e.g., 'episodes/test.jsonl')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results/report.md",
        help="Path to output report (default: eval_results/report.md)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV path override (if None, uses episode's csv_source)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Report format (default: markdown)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns per episode (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent evaluations (default: 5)",
    )
    parser.add_argument(
        "--float-tol",
        type=float,
        default=0.1,
        help="Float comparison tolerance (default: 0.1)",
    )

    args = parser.parse_args()

    # Validate inputs
    episodes_path = Path(args.episodes)
    if not episodes_path.exists():
        print(f"Error: Episodes file not found: {args.episodes}", file=sys.stderr)
        sys.exit(1)

    # Print configuration
    print("=" * 60)
    print("CSV Agent Model Evaluation")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Episodes:    {args.episodes}")
    print(f"Output:      {args.output}")
    print(f"Format:      {args.format}")
    print(f"Max Turns:   {args.max_turns}")
    print(f"Concurrency: {args.concurrency}")
    if args.csv:
        print(f"CSV Override: {args.csv}")
    print("=" * 60)
    print()

    # Create evaluator
    sampling_args = {
        "temperature": args.temperature,
        "max_tokens": 6000,
        "top_p": 1.0,
    }

    evaluator = Evaluator(
        model=args.model,
        csv_path=args.csv,
        max_turns=args.max_turns,
        sampling_args=sampling_args,
        float_tol=args.float_tol,
    )

    # Load episodes
    print(f"Loading episodes from {args.episodes}...")
    episodes = evaluator.load_episodes(args.episodes)
    print(f"Loaded {len(episodes)} episodes")
    print()

    # Run evaluation
    print(f"Evaluating {len(episodes)} episodes (concurrency={args.concurrency})...")
    results = await evaluator.evaluate_batch(episodes, concurrency=args.concurrency)
    print("Evaluation complete!")
    print()

    # Compute metrics
    print("Computing metrics...")
    metrics = evaluator.compute_metrics(results)

    # Print summary to console
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Accuracy:               {metrics.accuracy:.1%} ({metrics.total_correct}/{metrics.total_episodes})")
    print(f"Execution Success Rate: {metrics.execution_success_rate:.1%} ({metrics.total_executed}/{metrics.total_episodes})")
    print(f"Average Turns:          {metrics.avg_turns:.1f}")
    print(f"Average Time:           {metrics.avg_elapsed_seconds:.1f}s")

    if metrics.accuracy_by_difficulty:
        print()
        print("Accuracy by Difficulty:")
        for difficulty in sorted(metrics.accuracy_by_difficulty.keys()):
            accuracy = metrics.accuracy_by_difficulty[difficulty]
            total = metrics.episodes_by_difficulty[difficulty]
            correct = metrics.correct_by_difficulty.get(difficulty, 0)
            print(f"  {difficulty:12s}: {accuracy:5.1%} ({correct}/{total})")

    print("=" * 60)
    print()

    # Generate report
    print(f"Generating {args.format} report at {args.output}...")
    generate_report(
        metrics=metrics,
        results=results,
        output_path=args.output,
        format=args.format,
        model=args.model,
        episodes_path=args.episodes,
    )
    print(f"Report saved to {args.output}")
    print()
    print("Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
