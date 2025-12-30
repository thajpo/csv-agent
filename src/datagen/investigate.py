#!/usr/bin/env python3
"""
Failure investigation CLI for rapid iteration.

Quick diagnostic tool to understand WHY synthetic questions fail triangulation.
Designed for fast feedback loops during template development.

Usage:
    # Run diagnostic batch on synthetic questions (quick test)
    uv run python -m src.datagen.investigate --batch --limit 10

    # Analyze results from a batch run
    uv run python -m src.datagen.investigate --analyze results.jsonl

    # Show category breakdown from a diagnostic batch
    uv run python -m src.datagen.investigate --summary results.jsonl
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from csv_spec import FailureCategory


console = Console()


def analyze_diagnostics_file(path: Path) -> dict:
    """Load and analyze diagnostics from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return {"error": "No records found"}

    # Category breakdown
    categories = Counter()
    by_category: dict[str, list] = {}
    by_template: dict[str, Counter] = {}
    entropy_values = []

    for r in records:
        cat = r.get("failure_category", "unknown")
        categories[cat] += 1
        by_category.setdefault(cat, []).append(r)

        template = r.get("template_name", "unknown")
        by_template.setdefault(template, Counter())[cat] += 1

        if "entropy" in r:
            entropy_values.append(r["entropy"])

    return {
        "total": len(records),
        "categories": dict(categories),
        "by_category": by_category,
        "by_template": by_template,
        "avg_entropy": sum(entropy_values) / len(entropy_values) if entropy_values else 0,
    }


def print_summary_table(analysis: dict) -> None:
    """Print a summary table of failure categories."""
    table = Table(title="Failure Category Breakdown")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("Interpretation", style="dim")

    interpretations = {
        "good": "Verified successfully",
        "ambiguous": "Multiple valid interpretations",
        "too_hard": "Consistent wrong answer",
        "hint_necessary": "Needs hint to solve",
        "execution_failure": "Code didn't run",
    }

    total = analysis["total"]
    for cat, count in sorted(analysis["categories"].items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        interp = interpretations.get(cat, "")
        style = "green" if cat == "good" else "yellow" if cat == "ambiguous" else "red"
        table.add_row(cat.upper(), str(count), f"{pct:.1f}%", interp, style=style)

    console.print(table)
    console.print(f"\n[dim]Average entropy: {analysis['avg_entropy']:.2f}[/dim]")


def print_template_breakdown(analysis: dict) -> None:
    """Print per-template category breakdown."""
    table = Table(title="Failure by Template")
    table.add_column("Template", style="bold")
    table.add_column("Total", justify="right")
    table.add_column("Good", justify="right", style="green")
    table.add_column("Ambig", justify="right", style="yellow")
    table.add_column("Hard", justify="right", style="red")
    table.add_column("Hint", justify="right", style="cyan")
    table.add_column("Fail%", justify="right")

    for template, counts in sorted(
        analysis["by_template"].items(),
        key=lambda x: -sum(x[1].values())
    )[:15]:  # Top 15 templates
        total = sum(counts.values())
        good = counts.get("good", 0)
        ambig = counts.get("ambiguous", 0)
        hard = counts.get("too_hard", 0)
        hint = counts.get("hint_necessary", 0)
        fail_pct = 100 * (total - good) / total if total > 0 else 0

        table.add_row(
            template[:40],
            str(total),
            str(good),
            str(ambig),
            str(hard),
            str(hint),
            f"{fail_pct:.0f}%",
        )

    console.print(table)


def print_examples(analysis: dict, category: str, n: int = 3) -> None:
    """Print example failures for a category."""
    examples = analysis["by_category"].get(category, [])[:n]

    if not examples:
        console.print(f"[dim]No examples for category: {category}[/dim]")
        return

    console.print(f"\n[bold]Example {category.upper()} failures:[/bold]")

    for i, ex in enumerate(examples, 1):
        question = ex.get("question", "")[:100]
        reasoning = ex.get("classification_reasoning", "")
        template = ex.get("template_name", "unknown")

        panel = Panel(
            f"[dim]Template:[/dim] {template}\n"
            f"[dim]Question:[/dim] {question}...\n"
            f"[dim]Reason:[/dim] {reasoning}",
            title=f"Example {i}",
            border_style="dim",
        )
        console.print(panel)


async def run_diagnostic_batch(
    limit: int = 10,
    output_path: Path | None = None,
    template_filter: str | None = None,
) -> Path:
    """Run a small diagnostic batch and save results."""
    from src.datagen.teacher import batch_triangulate
    from src.datagen.pipeline_ui import EpisodeGenUI
    from src.core.config import config
    from src.core.prompts import generate_data_overview

    # Find a dataset with questions
    questions_dir = Path("data/questions_synthetic")
    if not questions_dir.exists():
        console.print("[red]No synthetic questions found. Run question generation first.[/red]")
        raise SystemExit(1)

    # Get first available dataset
    datasets = list(questions_dir.iterdir())
    if not datasets:
        console.print("[red]No datasets with questions found.[/red]")
        raise SystemExit(1)

    dataset_dir = datasets[0]
    questions_file = dataset_dir / "questions.json"
    csv_path = Path("data/kaggle") / dataset_dir.name / "data.csv"

    if not csv_path.exists():
        console.print(f"[red]CSV not found: {csv_path}[/red]")
        raise SystemExit(1)

    # Load questions (handle both dict and list formats)
    with open(questions_file) as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data

    # Filter by template if specified
    if template_filter:
        questions = [q for q in questions if template_filter.lower() in q.get("template_name", "").lower()]
        if not questions:
            console.print(f"[red]No questions matching template filter: {template_filter}[/red]")
            raise SystemExit(1)

    questions = questions[:limit]
    console.print(f"[bold]Running diagnostic batch on {len(questions)} questions from {dataset_dir.name}[/bold]")

    # Generate data overview
    data_overview = generate_data_overview(str(csv_path))

    # Run triangulation with diagnostics
    ui = EpisodeGenUI()

    results = await batch_triangulate(
        csv_path=str(csv_path),
        questions=questions,
        model=config.teacher_model,
        n_consistency=3,
        dataset_description="",
        data_overview=data_overview,
        max_turns=10,
        ui=ui,
        include_diagnostics=True,
    )

    # Save results
    output_path = output_path or Path(f"data/investigate_{dataset_dir.name}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for r in results:
            if r.diagnostics:
                record = {
                    "question": (r.question.get("question_text") or r.question.get("question", ""))[:200],
                    "template_name": r.question.get("template_name"),
                    "difficulty": r.question.get("difficulty"),
                    "verified": r.verified,
                    **r.diagnostics,
                }
                # Flatten answer_distribution for easier analysis
                if "answer_distribution" in record:
                    dist = record.pop("answer_distribution")
                    record["cluster_count"] = dist.get("cluster_count", 0)
                    record["entropy"] = dist.get("entropy", 0)
                    record["majority_confidence"] = dist.get("majority_confidence", 0)
                    record["successful_traces"] = dist.get("successful_traces", 0)
                    record["total_traces"] = dist.get("total_traces", 0)
                f.write(json.dumps(record) + "\n")

    console.print(f"[green]Saved diagnostics to: {output_path}[/green]")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Investigate synthetic question failures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run quick diagnostic batch
    uv run python -m src.datagen.investigate --batch --limit 5

    # Analyze results
    uv run python -m src.datagen.investigate --analyze data/investigate_*.jsonl

    # Focus on specific template
    uv run python -m src.datagen.investigate --batch --template MAX_VARIANCE --limit 5
        """,
    )

    parser.add_argument(
        "--batch", action="store_true",
        help="Run a diagnostic batch on synthetic questions"
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Number of questions to process (default: 10)"
    )
    parser.add_argument(
        "--template", type=str,
        help="Filter questions by template name (substring match)"
    )
    parser.add_argument(
        "--analyze", type=Path,
        help="Analyze diagnostics from a JSONL file"
    )
    parser.add_argument(
        "--summary", type=Path,
        help="Show summary statistics from a JSONL file"
    )
    parser.add_argument(
        "--examples", type=str,
        help="Show example failures for a category (e.g., 'ambiguous')"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output path for diagnostic results"
    )

    args = parser.parse_args()

    if args.batch:
        output_path = asyncio.run(run_diagnostic_batch(
            limit=args.limit,
            output_path=args.output,
            template_filter=args.template,
        ))
        # Auto-analyze the results
        analysis = analyze_diagnostics_file(output_path)
        console.print()
        print_summary_table(analysis)

    elif args.analyze:
        analysis = analyze_diagnostics_file(args.analyze)
        print_summary_table(analysis)
        console.print()
        print_template_breakdown(analysis)

        if args.examples:
            print_examples(analysis, args.examples)

    elif args.summary:
        analysis = analyze_diagnostics_file(args.summary)
        print_summary_table(analysis)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
