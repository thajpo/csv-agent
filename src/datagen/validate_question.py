"""
Validate a single question end-to-end with full visibility.

This enables tight debugging loops - test one question without running
the full pipeline.

Usage:
    # From a questions file:
    uv run python -m src.datagen.validate_question \
        --csv data/csv/data.csv \
        --questions-file data/questions_synthetic/marketing-data/questions.json \
        --index 0

    # Custom question:
    uv run python -m src.datagen.validate_question \
        --csv data/csv/data.csv \
        --question "What is the mean of Age?" \
        --hint "Use df['Age'].mean()"

    # Show full trace details:
    uv run python -m src.datagen.validate_question \
        --csv data/csv/data.csv \
        --questions-file ... \
        --show-trace --show-code
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.datagen.teacher import execute_teacher_trace
from src.datagen.pipeline_ui import EpisodeGenUI
from src.core.prompts import generate_data_overview
from src.core.config import config
from csv_spec import hash_artifact


console = Console()


def load_question_from_file(questions_file: str, index: int) -> dict:
    """Load a specific question from a questions file."""
    with open(questions_file) as f:
        data = json.load(f)

    questions = data.get("questions", data if isinstance(data, list) else [])

    if index >= len(questions):
        raise ValueError(
            f"Index {index} out of range (have {len(questions)} questions)"
        )

    return questions[index]


async def validate_question(
    csv_path: str,
    question: str,
    hint: str | None = None,
    expected_answer: any = None,
    expected_hash: str | None = None,
    show_trace: bool = False,
    show_code: bool = False,
    save_trace: str | None = None,
) -> bool:
    """
    Validate a single question by running teacher trace.

    Returns True if validation passed.
    """
    ui = EpisodeGenUI()

    # Load dataset info
    data_overview = generate_data_overview(csv_path)

    # Get dataset description from meta.json if available
    csv_path_obj = Path(csv_path)
    meta_path = csv_path_obj.parent / "meta.json"
    dataset_description = ""
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                dataset_description = meta.get("description", "")
        except Exception:
            pass

    console.print(
        Panel(
            f"[bold]Question:[/bold] {question}\n\n"
            f"[dim]Hint:[/dim] {hint or 'None'}\n"
            f"[dim]CSV:[/dim] {csv_path}\n"
            f"[dim]Model:[/dim] {config.teacher_model}",
            title="Validation Setup",
        )
    )

    console.print("\n[dim]Running teacher trace...[/dim]\n")

    try:
        trace, conversation, system_prompt, elapsed = await execute_teacher_trace(
            csv_path=csv_path,
            question=question,
            model=config.teacher_model,
            hint=hint,
            dataset_description=dataset_description,
            data_overview=data_overview,
            max_turns=config.max_turns,
            sampling_args={
                "temperature": config.sampling_args.temperature,
                "max_tokens": config.sampling_args.max_tokens,
            },
            ui=ui,
        )

        success = trace.get("success", False)
        final_answer = trace.get("final_answer")
        final_hash = trace.get("final_answer_hash")

        # Check against expected
        answer_matches = True
        if expected_hash:
            answer_matches = final_hash == expected_hash
        elif expected_answer is not None:
            # Try direct comparison
            answer_matches = final_answer == expected_answer
            # Try hash comparison
            if not answer_matches:
                expected_h = hash_artifact(expected_answer)
                answer_matches = final_hash == expected_h

        # Display results
        if success and answer_matches:
            status = "[bold green]SUCCESS[/bold green]"
        elif success and not answer_matches:
            status = "[bold yellow]ANSWER MISMATCH[/bold yellow]"
        else:
            status = "[bold red]FAILED[/bold red]"

        console.print(
            Panel(
                f"{status}\n\n"
                f"[dim]Elapsed:[/dim] {elapsed:.1f}s\n"
                f"[dim]Turns:[/dim] {len(trace.get('turns', []))}\n"
                f"[dim]Success:[/dim] {trace.get('success', False)}",
                title="Result",
            )
        )

        # Show answer comparison
        if expected_answer is not None or expected_hash:
            table = Table(title="Answer Comparison")
            table.add_column("", style="dim")
            table.add_column("Value")
            table.add_column("Hash", style="dim")

            table.add_row(
                "Expected",
                str(expected_answer)[:50] if expected_answer else "N/A",
                (expected_hash or "")[:20],
            )
            table.add_row(
                "Actual",
                str(final_answer)[:50] if final_answer else "N/A",
                (final_hash or "")[:20],
            )
            console.print(table)

        # Show trace details
        if show_trace or show_code:
            console.print("\n[bold]Trace Details[/bold]\n")

            for i, turn in enumerate(trace.get("turns", []), 1):
                console.print(f"[bold]Turn {i}[/bold]")

                if show_trace and turn.get("reasoning"):
                    console.print(f"[dim]Reasoning:[/dim] {turn['reasoning'][:300]}...")

                if show_code and turn.get("code"):
                    console.print(
                        Syntax(
                            turn["code"], "python", theme="monokai", line_numbers=True
                        )
                    )

                execution = turn.get("execution", {})
                if execution.get("stdout"):
                    stdout = execution["stdout"]
                    # Truncate but show submit line
                    if len(stdout) > 500:
                        stdout = stdout[:400] + "\n...\n" + stdout[-100:]
                    console.print(Panel(stdout, title="stdout", style="green"))

                if execution.get("stderr"):
                    console.print(
                        Panel(execution["stderr"][:300], title="stderr", style="red")
                    )

                hooks = execution.get("hooks", [])
                if hooks:
                    console.print(f"[dim]Hooks ({len(hooks)}):[/dim]")
                    for h in hooks[:5]:
                        val = str(h.get("value", "?"))[:60]
                        console.print(f"  • {h.get('variable_name', '?')}: {val}")

                console.print()

        # Show final answer
        console.print(
            Panel(
                f"[bold]{final_answer}[/bold]",
                title="Final Answer",
                style="green" if (success and answer_matches) else "red",
            )
        )

        # Save trace
        if save_trace:
            output_path = Path(save_trace)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "question": question,
                        "hint": hint,
                        "csv_path": csv_path,
                        "trace": trace,
                        "elapsed": elapsed,
                        "expected_answer": expected_answer,
                        "expected_hash": expected_hash,
                        "answer_matches": answer_matches,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            console.print(f"\n[green]Trace saved to {output_path}[/green]")

        return success and answer_matches

    except Exception as e:
        console.print(f"[red]ERROR: {e}[/red]")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate a single question",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="CSV file path")

    # Question source (either file+index or direct question)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--questions-file", help="Questions JSON file")
    source.add_argument("--question", help="Direct question text")

    parser.add_argument("--index", type=int, default=0, help="Question index in file")
    parser.add_argument("--hint", help="Hint text (for direct question)")

    # Display options
    parser.add_argument("--show-trace", action="store_true", help="Show reasoning")
    parser.add_argument("--show-code", action="store_true", help="Show code cells")
    parser.add_argument("--save-trace", help="Save trace to JSON file")

    args = parser.parse_args()

    # Load question
    if args.questions_file:
        q = load_question_from_file(args.questions_file, args.index)
        question = q.get("question", q.get("question_text", ""))
        hint = q.get("hint")
        expected_answer = q.get("ground_truth") or q.get("_ground_truth")
        expected_hash = q.get("ground_truth_hash")
    else:
        question = args.question
        hint = args.hint
        expected_answer = None
        expected_hash = None

    # Run validation
    success = asyncio.run(
        validate_question(
            csv_path=args.csv,
            question=question,
            hint=hint,
            expected_answer=expected_answer,
            expected_hash=expected_hash,
            show_trace=args.show_trace,
            show_code=args.show_code,
            save_trace=args.save_trace,
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
