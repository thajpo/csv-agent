#!/usr/bin/env python3
"""
Smoke test for csv-agent pipeline.

Usage:
    uv run python scripts/smoke_test.py                    # Full pipeline
    uv run python scripts/smoke_test.py --stage questions  # Question gen only
    uv run python scripts/smoke_test.py --stage triangulation  # Triangulation only
    uv run python scripts/smoke_test.py --n-questions 3    # More questions per CSV
    uv run python scripts/smoke_test.py --csvs 2           # Use 2 CSVs instead of 3
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


DEFAULT_CSVS = [
    "data/mock/data.csv",
    "data/fixtures/smoke/breast_cancer/data.csv",
    "data/fixtures/smoke/student_performance/data.csv",
]

SMOKE_TEST_DIR = Path("smoke_test")
console = Console()


def clear_smoke_test_dir():
    if SMOKE_TEST_DIR.exists():
        shutil.rmtree(SMOKE_TEST_DIR)
    SMOKE_TEST_DIR.mkdir(parents=True)
    console.print(f"[dim]Cleared {SMOKE_TEST_DIR}/[/dim]")


def print_header(title: str):
    console.print()
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False))


def print_subheader(text: str):
    console.print(f"\n[bold yellow]{text}[/bold yellow]")


def get_csv_name(csv_path: str) -> str:
    p = Path(csv_path)
    if p.name == "data.csv":
        return p.parent.name
    return p.stem


async def run_synthetic_questions(
    csv_path: str, output_dir: Path, n_questions: int
) -> dict | None:
    from src.datagen.synthetic.generator import (
        CompositionalQuestionGenerator,
        load_dataset_description,
    )

    dataset_name = get_csv_name(csv_path)
    print_subheader(f"Synthetic Questions: {dataset_name}")

    if not Path(csv_path).exists():
        console.print(f"[red]CSV not found: {csv_path}[/red]")
        return None

    try:
        dataset_description = load_dataset_description(csv_path)
        console.print(
            f"[dim]Description: {dataset_description[:80]}...[/dim]"
            if dataset_description
            else "[dim]No description[/dim]"
        )

        generator = CompositionalQuestionGenerator(
            csv_path=csv_path,
            dataset_description=dataset_description,
        )

        await generator.setup()
        result = await generator.generate(n_questions=n_questions)
        await generator.cleanup()

        questions_dir = output_dir / "questions_synthetic" / dataset_name
        questions_dir.mkdir(parents=True, exist_ok=True)
        questions_file = questions_dir / "questions.json"

        with open(questions_file, "w") as f:
            json.dump(result, f, indent=2)

        console.print(f"[green]Generated {len(result['questions'])} questions[/green]")
        console.print(f"[dim]Saved to: {questions_file}[/dim]")

        if result["questions"]:
            console.print("\n[bold]Sample questions:[/bold]")
            for i, q in enumerate(result["questions"][:3], 1):
                console.print(f"  {i}. {q['question'][:100]}...")
                console.print(
                    f"     [dim]Difficulty: {q.get('difficulty', 'N/A')} | Template: {q.get('template_name', 'N/A')}[/dim]"
                )

        return result

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return None


async def run_llm_questions(
    csv_path: str, output_dir: Path, max_turns: int = 5
) -> dict | None:
    from src.datagen.question_gen import explore_and_generate_questions
    from src.datagen.synthetic.generator import load_dataset_description
    from src.core.config import config

    dataset_name = get_csv_name(csv_path)
    print_subheader(f"LLM Questions: {dataset_name}")

    if not Path(csv_path).exists():
        console.print(f"[red]CSV not found: {csv_path}[/red]")
        return None

    try:
        dataset_description = load_dataset_description(csv_path)
        console.print(
            f"[dim]Description: {dataset_description[:80]}...[/dim]"
            if dataset_description
            else "[dim]No description[/dim]"
        )

        questions_dir = output_dir / "questions_llm" / dataset_name
        questions_dir.mkdir(parents=True, exist_ok=True)

        questions, trace = await explore_and_generate_questions(
            csv_path=csv_path,
            model=config.question_gen_model,
            max_turns=max_turns,
            temperature=config.sampling_args.temperature,
            max_tokens=config.sampling_args.max_tokens,
            output_dir=str(questions_dir),
            dataset_description=dataset_description,
        )

        console.print(
            f"[green]Generated {len(questions)} questions in {len(trace.turns)} turns[/green]"
        )

        if questions:
            console.print("\n[bold]Sample questions:[/bold]")
            for i, q in enumerate(questions[:3], 1):
                console.print(f"  {i}. {q['question'][:100]}...")
                console.print(
                    f"     [dim]Difficulty: {q.get('difficulty', 'N/A')} | Steps: {q.get('n_steps', 'N/A')}[/dim]"
                )

        return {"questions": questions, "trace": trace}

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return None


async def run_triangulation(
    csv_path: str,
    questions: list[dict],
    output_dir: Path,
    n_consistency: int = 3,
) -> list | None:
    from src.datagen.teacher import triangulate_teacher
    from src.datagen.ui import EpisodeGenUI
    from src.datagen.synthetic.generator import load_dataset_description
    from src.core.config import config
    from src.core.prompts import generate_data_overview

    dataset_name = get_csv_name(csv_path)
    print_subheader(f"Triangulation: {dataset_name}")

    if not questions:
        console.print("[yellow]No questions to triangulate[/yellow]")
        return None

    try:
        ui = EpisodeGenUI()
        dataset_description = load_dataset_description(csv_path)
        data_overview = generate_data_overview(csv_path)

        results = []
        for i, q in enumerate(questions, 1):
            console.print(
                f"\n[bold]Question {i}/{len(questions)}:[/bold] {q['question'][:80]}..."
            )

            (
                gold_trace,
                gold_conversation,
                system_prompt,
                consistency_results,
                verified,
                timing_metadata,
                majority_hash,
                majority_count,
            ) = await triangulate_teacher(
                csv_path=csv_path,
                question=q["question"],
                hint=q.get("hint", ""),
                n_steps=q.get("n_steps"),
                difficulty=q.get("difficulty"),
                n_consistency=n_consistency,
                model=config.teacher_model,
                dataset_description=dataset_description,
                data_overview=data_overview,
                max_turns=config.max_turns,
                sampling_args=config.sampling_args.model_dump(),
                ui=ui,
                float_tol=config.float_tolerance,
            )

            status = "[green]VERIFIED[/green]" if verified else "[red]FAILED[/red]"
            console.print(f"  Result: {status}")
            console.print(f"  [dim]Gold answer: {gold_trace['final_answer']}[/dim]")
            console.print(
                f"  [dim]Majority count: {majority_count}/{n_consistency}[/dim]"
            )
            console.print(
                f"  [dim]Timing: {timing_metadata['total_elapsed']:.1f}s total[/dim]"
            )

            results.append(
                {
                    "question": q,
                    "verified": verified,
                    "gold_answer": gold_trace["final_answer"],
                    "majority_count": majority_count,
                    "timing": timing_metadata,
                }
            )

        results_dir = output_dir / "triangulation" / dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / "results.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        verified_count = sum(1 for r in results if r["verified"])
        console.print(
            f"\n[bold]Summary:[/bold] {verified_count}/{len(results)} verified"
        )
        console.print(f"[dim]Saved to: {results_file}[/dim]")

        return results

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return None


async def run_full_pipeline(
    csv_paths: list[str],
    output_dir: Path,
    n_questions: int,
    n_consistency: int,
    include_llm: bool = False,
):
    print_header("FULL PIPELINE SMOKE TEST")

    all_results = []

    for csv_path in csv_paths:
        dataset_name = get_csv_name(csv_path)
        console.print(f"\n[bold magenta]{'=' * 60}[/bold magenta]")
        console.print(f"[bold magenta]Dataset: {dataset_name}[/bold magenta]")
        console.print(f"[bold magenta]{'=' * 60}[/bold magenta]")

        synth_result = await run_synthetic_questions(csv_path, output_dir, n_questions)

        llm_result = None
        if include_llm:
            llm_result = await run_llm_questions(csv_path, output_dir, max_turns=5)

        if synth_result and synth_result["questions"]:
            questions_to_test = synth_result["questions"][:n_questions]
            tri_results = await run_triangulation(
                csv_path, questions_to_test, output_dir, n_consistency
            )
            all_results.append(
                {
                    "dataset": dataset_name,
                    "csv_path": csv_path,
                    "synthetic_questions": len(synth_result["questions"])
                    if synth_result
                    else 0,
                    "llm_questions": len(llm_result["questions"]) if llm_result else 0,
                    "triangulation_results": tri_results,
                }
            )

    print_header("SMOKE TEST SUMMARY")

    table = Table(title="Results by Dataset")
    table.add_column("Dataset", style="cyan")
    table.add_column("Synth Q's", justify="right")
    table.add_column("LLM Q's", justify="right")
    table.add_column("Verified", justify="right")
    table.add_column("Total", justify="right")

    for r in all_results:
        tri = r.get("triangulation_results") or []
        verified = sum(1 for t in tri if t.get("verified"))
        table.add_row(
            r["dataset"],
            str(r["synthetic_questions"]),
            str(r["llm_questions"]),
            str(verified),
            str(len(tri)),
        )

    console.print(table)
    console.print(f"\n[dim]All outputs saved to: {output_dir}/[/dim]")


async def run_questions_only(
    csv_paths: list[str],
    output_dir: Path,
    n_questions: int,
    include_llm: bool = False,
):
    print_header("QUESTION GENERATION SMOKE TEST")

    for csv_path in csv_paths:
        dataset_name = get_csv_name(csv_path)
        console.print(f"\n[bold magenta]Dataset: {dataset_name}[/bold magenta]")

        await run_synthetic_questions(csv_path, output_dir, n_questions)

        if include_llm:
            await run_llm_questions(csv_path, output_dir, max_turns=5)


async def run_triangulation_only(
    csv_paths: list[str],
    output_dir: Path,
    n_questions: int,
    n_consistency: int,
):
    print_header("TRIANGULATION SMOKE TEST")

    for csv_path in csv_paths:
        dataset_name = get_csv_name(csv_path)
        console.print(f"\n[bold magenta]Dataset: {dataset_name}[/bold magenta]")

        synth_result = await run_synthetic_questions(csv_path, output_dir, n_questions)

        if synth_result and synth_result["questions"]:
            questions_to_test = synth_result["questions"][:n_questions]
            await run_triangulation(
                csv_path, questions_to_test, output_dir, n_consistency
            )


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for csv-agent pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["questions", "triangulation", "full"],
        default="full",
        help="Pipeline stage to run (default: full)",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=1,
        help="Number of questions per CSV (default: 1)",
    )
    parser.add_argument(
        "--n-consistency",
        type=int,
        default=3,
        help="Number of consistency traces (default: 3)",
    )
    parser.add_argument(
        "--csvs",
        type=int,
        default=3,
        help="Number of CSVs to use (default: 3)",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Include LLM question generation (slower)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="*",
        default=None,
        help="Specific CSV paths to use (overrides --csvs)",
    )

    args = parser.parse_args()

    if args.csv:
        csv_paths = args.csv
    else:
        csv_paths = DEFAULT_CSVS[: args.csvs]

    missing = [p for p in csv_paths if not Path(p).exists()]
    if missing:
        console.print(f"[red]Missing CSVs: {missing}[/red]")
        console.print("[yellow]Try running from project root or check paths[/yellow]")
        return 1

    clear_smoke_test_dir()

    console.print(
        Panel(
            f"[bold]Stage:[/bold] {args.stage}\n"
            f"[bold]CSVs:[/bold] {len(csv_paths)}\n"
            f"[bold]Questions/CSV:[/bold] {args.n_questions}\n"
            f"[bold]Consistency traces:[/bold] {args.n_consistency}\n"
            f"[bold]Include LLM:[/bold] {args.include_llm}",
            title="Smoke Test Configuration",
            expand=False,
        )
    )

    for i, csv in enumerate(csv_paths, 1):
        console.print(f"  {i}. {csv}")

    try:
        if args.stage == "questions":
            asyncio.run(
                run_questions_only(
                    csv_paths, SMOKE_TEST_DIR, args.n_questions, args.include_llm
                )
            )
        elif args.stage == "triangulation":
            asyncio.run(
                run_triangulation_only(
                    csv_paths, SMOKE_TEST_DIR, args.n_questions, args.n_consistency
                )
            )
        else:
            asyncio.run(
                run_full_pipeline(
                    csv_paths,
                    SMOKE_TEST_DIR,
                    args.n_questions,
                    args.n_consistency,
                    args.include_llm,
                )
            )

        console.print("\n[bold green]Smoke test complete![/bold green]")
        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
