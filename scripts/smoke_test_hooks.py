#!/usr/bin/env python3
"""
Smoke test for hook validation.

Runs a single teacher trace and verifies hooks are being:
1. Required (prompt enforcement)
2. Grounded (code_line matches executed code)
3. Sufficient (count >= n_steps)

Usage:
    uv run python scripts/smoke_test_hooks.py
    uv run python scripts/smoke_test_hooks.py --csv path/to/data.csv
"""
import asyncio
import argparse
from pathlib import Path

from src.datagen.teacher import execute_teacher_trace
from src.datagen.ui import EpisodeGenUI
from src.core.prompts import generate_data_overview
from src.core.types import Question


async def main(csv_path: str, model: str):
    ui = EpisodeGenUI()
    ui.print_header()

    # Load CSV and generate overview
    csv_path = Path(csv_path)
    if not csv_path.exists():
        ui.console.print(f"[red]CSV not found: {csv_path}[/red]")
        return

    ui.console.print(f"[cyan]Loading CSV:[/cyan] {csv_path}")
    data_overview = generate_data_overview(str(csv_path))

    # Create a simple test question
    question = Question(
        question_text="What is the mean value of the first numeric column?",
        hint="Use df.select_dtypes('number').iloc[:, 0].mean()",
        difficulty="EASY",
        n_steps=2,  # Expect at least 2 hooks
    )

    ui.console.print(f"\n[cyan]Test Question:[/cyan] {question.question_text}")
    ui.console.print(f"[cyan]Expected hooks:[/cyan] ~{question.n_steps}")
    ui.console.print(f"[cyan]Model:[/cyan] {model}\n")

    # Run teacher trace
    ui.console.print("[yellow]Running teacher trace...[/yellow]\n")

    trace, conversation, system_prompt, elapsed = await execute_teacher_trace(
        csv_path=str(csv_path),
        question=question.question_text,
        model=model,
        hint=question.hint,
        n_steps=question.n_steps,
        difficulty=question.difficulty,
        mode="teacher-tutor",
        dataset_description=f"CSV file: {csv_path.name}",
        data_overview=data_overview,
        max_turns=10,
        ui=ui,
    )

    # Report results
    ui.console.print("\n" + "=" * 60)
    ui.console.print("[bold cyan]HOOK VALIDATION RESULTS[/bold cyan]")
    ui.console.print("=" * 60)

    hooks = trace.hooks
    code_cells = trace.code_cells

    ui.console.print(f"\n[cyan]Hooks found:[/cyan] {len(hooks)}")
    ui.console.print(f"[cyan]Code cells executed:[/cyan] {len(code_cells)}")

    if hooks:
        ui.console.print("\n[bold]Hook Details:[/bold]")
        for i, hook in enumerate(hooks, 1):
            ui.console.print(f"\n  [green]Hook {i}:[/green]")
            ui.console.print(f"    name: {hook.variable_name}")
            ui.console.print(f"    code_line: {hook.code_line[:60]}...")
            ui.console.print(f"    depends_on: {hook.depends_on}")

            # Check grounding
            all_code = "\n".join(code_cells)
            normalized_line = " ".join(hook.code_line.split())
            normalized_code = " ".join(all_code.split())
            is_grounded = normalized_line in normalized_code
            status = "[green]✓ GROUNDED[/green]" if is_grounded else "[red]✗ UNGROUNDED[/red]"
            ui.console.print(f"    status: {status}")
    else:
        ui.console.print("\n[red]⚠️  NO HOOKS FOUND - enforcement may have failed[/red]")

    # Summary
    ui.console.print("\n" + "-" * 60)
    if len(hooks) >= question.n_steps:
        ui.console.print(f"[green]✓ PASS: Got {len(hooks)} hooks (expected ~{question.n_steps})[/green]")
    else:
        ui.console.print(f"[red]✗ FAIL: Got {len(hooks)} hooks (expected ~{question.n_steps})[/red]")

    if trace.final_answer is not None:
        ui.console.print(f"[cyan]Final answer:[/cyan] {trace.final_answer}")
    else:
        ui.console.print("[yellow]No answer submitted[/yellow]")

    ui.console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for hook validation")
    parser.add_argument(
        "--csv",
        default="data/csv/data.csv",
        help="Path to CSV file (default: data/csv/data.csv)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model to use (default: openai/gpt-4o-mini)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.csv, args.model))
