"""
LLM-based question generator for CSV datasets.

This script uses an LLM to:
1. Explore a dataset using Jupyter kernel
2. Document exploration observations
3. Generate questions with varying difficulty levels (EASY, MEDIUM, HARD, VERY_HARD)

Usage:
    python -m scripts.question_gen --csv data.csv --output questions.json
"""

import argparse
import json
import re
import sys
import yaml
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

from src.kernel import JupyterKernel
from src.model import APILLM
from src.conversation import ConversationManager, Turn, CodeCellResult
from src.types import ExplorationTurn, ExplorationTrace
from src.prompts import EXPLORATION_SYSTEM_PROMPT, EXPLORATION_CONTINUE_MSG

# Create rich console
console = Console()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    return {}


def extract_python_cells(response: str) -> list[str]:
    """Extract ```python...``` code blocks from response."""
    pattern = r'```python\n(.*?)```'
    return re.findall(pattern, response, re.DOTALL)


def try_parse_questions(response: str) -> list[dict] | None:
    """
    Try to parse questions from model response.

    Looks for JSON block with structure:
    {
        "questions": [
            {"question": ..., "hint": ..., "n_steps": ..., "difficulty": ...},
            ...
        ]
    }

    Returns:
        List of question dicts if found and valid, None otherwise
    """
    # Try to find ```json...``` block
    json_pattern = r'```json\n(.*?)```'
    matches = re.findall(json_pattern, response, re.DOTALL)

    if not matches:
        return None

    try:
        data = json.loads(matches[0])
        if "questions" in data and isinstance(data["questions"], list):
            # Validate structure
            questions = data["questions"]
            for q in questions:
                if not all(key in q for key in ["question", "hint", "n_steps", "difficulty"]):
                    return None
            return questions
    except json.JSONDecodeError:
        return None

    return None


def build_execution_feedback(results: list[CodeCellResult]) -> str:
    """Build feedback message from execution results."""
    if not results:
        return "No code blocks found. Write Python code in ```python blocks."

    feedback_parts = []
    for i, result in enumerate(results, 1):
        if result.success:
            feedback_parts.append(f"✓ Cell {i} executed successfully")
            if result.stdout.strip():
                feedback_parts.append(f"Output:\n{result.stdout}")
        else:
            feedback_parts.append(f"✗ Cell {i} failed")
            feedback_parts.append(f"Error:\n{result.stderr}")

    return "\n\n".join(feedback_parts)


def force_question_generation(llm: APILLM, conversation: ConversationManager) -> list[dict]:
    """
    If model hasn't generated questions by max_turns, force it with a direct prompt.

    Returns:
        List of question dicts
    """
    # Add forcing message
    force_turn = Turn(
        turn_number=conversation.get_active_turn_count(),
        timestamp=datetime.now(),
        model_response="",
        done_signal=False,
        feedback_message="You've explored enough. Now generate the 13 questions in JSON format as specified in the system prompt.",
        reasoning=None
    )
    conversation.add_turn(force_turn)

    # Get response
    messages = conversation.to_openai_messages()
    response = llm(messages)

    # Try to parse
    questions = try_parse_questions(response)

    if not questions:
        raise RuntimeError("Model failed to generate valid questions even after forcing. Check model output.")

    return questions


def explore_and_generate_questions(
    csv_path: str,
    model: str = "meta-llama/llama-3.2-3b-instruct:free",
    max_turns: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    output_dir: str = "."
) -> tuple[list[dict], ExplorationTrace]:
    """
    LLM explores dataset and generates questions.

    Args:
        csv_path: Path to CSV file
        model: Model identifier for APILLM
        max_turns: Max exploration turns before forcing question generation
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        output_dir: Where to save outputs

    Returns:
        (questions, exploration_trace)
    """
    console.print(f"\n[bold green]Question Generation[/bold green] Starting exploration of {csv_path}")
    console.print(f"[bold blue]Model:[/bold blue] {model}")
    console.print(f"[bold blue]Max turns:[/bold blue] {max_turns}\n")

    # 1. Setup
    kernel = JupyterKernel(timeout=120, csv_path=csv_path)
    llm = APILLM(model=model, sampling_args={"temperature": temperature, "max_tokens": max_tokens})
    conversation = ConversationManager(
        system_prompt=EXPLORATION_SYSTEM_PROMPT,
        max_active_turns=50,  # Keep full exploration in context
        max_context_tokens=100_000
    )

    # 2. Multi-turn exploration loop
    exploration_turns = []
    questions_generated = None

    for turn_num in range(max_turns):
        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold cyan]TURN {turn_num + 1}/{max_turns}[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

        # Get model response
        messages = conversation.to_openai_messages()
        console.print("[yellow]Generating LLM response...[/yellow]")
        response = llm(messages)

        # Display LLM response in a panel
        console.print(Panel(
            Markdown(response),
            title="[bold yellow]LLM Response[/bold yellow]",
            border_style="yellow"
        ))

        # Extract code cells
        code_cells = extract_python_cells(response)
        console.print(f"\n[bold blue]Found {len(code_cells)} code block(s)[/bold blue]")

        # Execute code
        execution_results = []
        for i, code in enumerate(code_cells, 1):
            console.print(f"\n[bold magenta]Executing Cell {i}[/bold magenta]")

            # Display code with syntax highlighting
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=f"[bold magenta]Code Cell {i}[/bold magenta]", border_style="magenta"))

            result = kernel.execute(code)
            execution_results.append(CodeCellResult(
                code=code,
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr if not result.success else ""
            ))

            if result.success:
                console.print("[bold green]✓ Execution Successful[/bold green]")
                if result.stdout.strip():
                    # Show actual output in a clearly marked panel
                    console.print(Panel(
                        result.stdout,
                        title="[bold green]ACTUAL OUTPUT FROM CODE EXECUTION[/bold green]",
                        border_style="green",
                        padding=(1, 2)
                    ))
                else:
                    console.print("[dim]No output[/dim]")
            else:
                console.print(f"[bold red]✗ Execution Failed[/bold red]")
                console.print(Panel(
                    result.error_message,
                    title="[bold red]ERROR[/bold red]",
                    border_style="red"
                ))

        # Save turn
        turn = ExplorationTurn(
            turn_number=turn_num,
            reasoning=response,
            code_cells=code_cells,
            execution_results=execution_results,
            timestamp=datetime.now()
        )
        exploration_turns.append(turn)

        # Check if model generated questions
        questions_generated = try_parse_questions(response)
        if questions_generated:
            console.print(f"\n[bold green]✓ Success! Model generated {len(questions_generated)} questions![/bold green]")
            break

        # Build feedback
        feedback = build_execution_feedback(execution_results)
        feedback += EXPLORATION_CONTINUE_MSG

        # Add turn to conversation
        conversation_turn = Turn(
            turn_number=turn_num,
            timestamp=datetime.now(),
            model_response=response,
            code_cells=code_cells,
            execution_results=execution_results,
            done_signal=False,
            feedback_message=feedback,
            reasoning=None
        )
        conversation.add_turn(conversation_turn)

    # 3. Validate we got questions
    if not questions_generated:
        console.print("\n[bold yellow]⚠ Warning: Model didn't generate questions. Forcing...[/bold yellow]")
        questions_generated = force_question_generation(llm, conversation)

    # 4. Create trace
    trace = ExplorationTrace(
        csv_path=csv_path,
        turns=exploration_turns,
        questions_generated=questions_generated,
        total_turns=len(exploration_turns),
        timestamp=datetime.now()
    )

    # 5. Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save questions.json
    questions_file = output_path / "questions.json"
    with open(questions_file, 'w') as f:
        json.dump(questions_generated, f, indent=2)
    console.print(f"\n[bold green]💾 Saved questions → {questions_file}[/bold green]")

    # Save exploration trace
    trace_file = output_path / "exploration_trace.json"
    with open(trace_file, 'w') as f:
        json.dump(trace.model_dump(), f, indent=2, default=str)
    console.print(f"[bold green]💾 Saved exploration trace → {trace_file}[/bold green]")

    # Cleanup
    kernel.shutdown()

    return questions_generated, trace


def main():
    # Parse preliminary args to get config path
    parser_prelim = argparse.ArgumentParser(description="Generate questions using LLM exploration", add_help=False)
    parser_prelim.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    args_prelim, _ = parser_prelim.parse_known_args()

    # Load config
    config = load_config(args_prelim.config)

    # Parse all arguments with config defaults
    parser = argparse.ArgumentParser(description="Generate questions using LLM exploration")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--csv", default=config.get("csv", "data.csv"), help="Path to CSV file")
    parser.add_argument("--output", "-o", default="questions.json", help="Output JSON file for questions")
    parser.add_argument("--exploration-output", default="exploration_trace.json", help="Output JSON file for exploration trace")
    parser.add_argument("--max-turns", type=int, default=config.get("question_gen_max_turns", 20), help="Max exploration turns")
    parser.add_argument("--temperature", type=float, default=config.get("sampling_args", {}).get("temperature", 0.7), help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=config.get("sampling_args", {}).get("max_tokens", 2000), help="Max tokens per response")

    args = parser.parse_args()

    # Set output directory from output file path
    output_dir = str(Path(args.output).parent)
    if output_dir == ".":
        output_dir = "."

    try:
        model = config.get("question_gen_model") or args.model

        questions, trace = explore_and_generate_questions(
            csv_path=args.csv,
            model=model,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=output_dir
        )

        # Print summary
        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print("[bold cyan]SUMMARY[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold]Total questions:[/bold] {len(questions)}")

        # Count by difficulty
        difficulty_counts = {}
        for q in questions:
            diff = q.get("difficulty", "UNKNOWN")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        console.print("\n[bold]By difficulty:[/bold]")
        for diff, count in sorted(difficulty_counts.items()):
            console.print(f"  [cyan]{diff}:[/cyan] {count}")

        console.print(f"\n[bold]Sample questions:[/bold]")
        for i, q in enumerate(questions[:3], 1):
            console.print(Panel(
                f"[bold]{q['question']}[/bold]\n\n"
                f"[dim]Steps:[/dim] {q['n_steps']}\n"
                f"[dim]Hint:[/dim] {q['hint']}",
                title=f"[bold cyan]Question {i} - {q['difficulty']}[/bold cyan]",
                border_style="cyan"
            ))

        return 0

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]", style="bold red")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
