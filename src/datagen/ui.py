"""
Rich UI components for data generation pipelines.

Contains specialized UI classes for:
- QuestionGenUI: Question generation output
- EpisodeGenUI: Episode generation and triangulation output
"""
import json
import re
import textwrap
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from src.utils.ui import ConsoleUI


class QuestionGenUI(ConsoleUI):
    """Rich console output for question generation, extends base ConsoleUI."""
    
    def print_turn_header(self, turn_num: int, max_turns: int) -> None:
        """Print the turn separator and header."""
        self.print_section(f"TURN {turn_num + 1}/{max_turns}")
    
    def print_llm_response(self, response: str) -> None:
        """Display LLM response with appropriate formatting."""
        if response.strip().startswith('{') and '"questions"' in response:
            try:
                parsed_json = json.loads(response.strip())
                formatted_json = json.dumps(parsed_json, indent=2)
                syntax = Syntax(formatted_json, "json", theme="monokai", line_numbers=False)
                self.console.print(Panel(
                    syntax,
                    title="[bold yellow]LLM Response (JSON)[/bold yellow]",
                    border_style="yellow"
                ))
            except json.JSONDecodeError:
                self.print_markdown_panel(response, "LLM Response")
        else:
            self.print_markdown_panel(response, "LLM Response")
    
    def print_code_cell(self, cell_num: int, code: str) -> None:
        """Display a code cell with syntax highlighting."""
        self.console.print(f"\n[bold magenta]Executing Cell {cell_num}[/bold magenta]")
        self.print_code_panel(code, f"Code Cell {cell_num}")
    
    def print_saved_file(self, file_path) -> None:
        """Print file save confirmation."""
        label = "questions" if "questions" in str(file_path) else "exploration trace"
        self.console.print(f"[bold green]💾 Saved {label} → {file_path}[/bold green]")
    
    def print_summary_header(self) -> None:
        """Print summary section header."""
        self.print_section("SUMMARY")
    
    def print_question_panel(self, question_num: int, question: dict) -> None:
        """Print a question in a formatted panel."""
        self.console.print(Panel(
            f"[bold]{question['question']}[/bold]\n\n"
            f"[dim]Steps:[/dim] {question['n_steps']}\n"
            f"[dim]Hint:[/dim] {question['hint']}",
            title=f"[bold cyan]Question {question_num} - {question['difficulty']}[/bold cyan]",
            border_style="cyan"
        ))
    
    def print_code_blocks_found(self, count: int) -> None:
        """Print number of code blocks found."""
        self.console.print(f"\n[bold blue]Found {count} code block(s)[/bold blue]")
    
    def print_total_questions(self, count: int) -> None:
        """Print total question count."""
        self.print_key_value("Total questions", count)
    
    def print_difficulty_header(self) -> None:
        """Print difficulty section header."""
        self.console.print("\n[bold]By difficulty:[/bold]")
    
    def print_difficulty_count(self, difficulty: str, count: int) -> None:
        """Print a difficulty count."""
        self.console.print(f"  [cyan]{difficulty}:[/cyan] {count}")
    
    def print_sample_questions_header(self) -> None:
        """Print sample questions section header."""
        self.console.print("\n[bold]Sample questions:[/bold]")


class EpisodeGenUI:
    """Rich UI for episode generation pipeline."""

    def __init__(self):
        self.base = ConsoleUI()
        self.console = self.base.console

    def print_pipeline_header(
        self,
        n_questions: int,
        n_consistency: int,
        csv_path: str,
        model: str,
        float_tol: float,
        output_file: str
    ) -> None:
        """Print the main pipeline header with configuration."""
        total_calls = n_questions * (n_consistency + 1)

        self.console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
        self.console.print("[bold cyan]  EPISODE GENERATION PIPELINE[/bold cyan]")
        self.console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

        self.console.print("[bold]Configuration:[/bold]")
        self.base.print_key_value("  CSV Path", csv_path, indent=2)
        self.base.print_key_value("  Questions", f"{n_questions} questions loaded", indent=2)
        self.base.print_key_value("  Model", model, indent=2)
        self.base.print_key_value("  N Consistency", f"{n_consistency} runs per question", indent=2)
        self.base.print_key_value("  Float Tolerance", f"±{float_tol}", indent=2)
        self.base.print_key_value("  Total LLM Calls", f"{total_calls} ({n_questions} × {n_consistency + 1})", indent=2)

        self.console.print(f"\n[bold]Output:[/bold] {output_file}")
        self.console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

    def print_question_header(self, q_num: int, total: int, question: dict) -> None:
        """Print question section header."""
        difficulty = question["difficulty"]
        n_steps = question["n_steps"]
        q_text = question["question"]
        hint = question["hint"]

        self.console.print("\n[bold magenta]" + "┌" + "─" * 58 + "┐[/bold magenta]")
        self.console.print(f"[bold magenta]│ QUESTION {q_num}/{total} - {difficulty} ({n_steps} steps)" + " " * (58 - len(f" QUESTION {q_num}/{total} - {difficulty} ({n_steps} steps)")) + "│[/bold magenta]")
        self.console.print("[bold magenta]" + "├" + "─" * 58 + "┤[/bold magenta]")

        wrapped_q = textwrap.fill(q_text, width=56)
        for line in wrapped_q.split("\n"):
            padding = 56 - len(line)
            self.console.print(f"[bold magenta]│[/bold magenta] {line}" + " " * padding + " [bold magenta]│[/bold magenta]")

        if hint:
            self.console.print("[bold magenta]│" + " " * 58 + "│[/bold magenta]")
            self.console.print("[bold magenta]│[/bold magenta] [dim]Hint:[/dim]" + " " * 51 + "[bold magenta]│[/bold magenta]")
            wrapped_h = textwrap.fill(hint, width=50)
            for line in wrapped_h.split("\n"):
                padding = 56 - len(line) - 6
                self.console.print(f"[bold magenta]│[/bold magenta]       {line}" + " " * padding + " [bold magenta]│[/bold magenta]")

        self.console.print("[bold magenta]" + "└" + "─" * 58 + "┘[/bold magenta]\n")

    def print_trace_header(self, mode: str, hint: str | None = None) -> None:
        """Print trace execution header."""
        if mode == "gold":
            title = "GOLD TRACE (with hint)"
            style = "green"
        else:
            title = f"CONSISTENCY TRACE {mode} (no hint)"
            style = "yellow"

        self.console.print(f"\n[bold {style}]▼ {title}[/bold {style}]")
        if hint and mode == "gold":
            self.console.print(f"[dim]  Hint: {hint}[/dim]")

    def print_trace_start(self, mode: str) -> None:
        """Print trace start indicator."""
        if mode == "gold":
            self.console.print(f"[dim]  ⏱️  Starting gold trace execution...[/dim]")
        else:
            self.console.print(f"[dim]  ⏱️  Starting consistency trace {mode} execution...[/dim]")

    def print_api_call_start(self, turn_num: int, max_turns: int) -> None:
        """Print API call start indicator."""
        self.console.print(f"[dim]    → Turn {turn_num}/{max_turns}: Calling LLM API...[/dim]")

    def print_api_call_complete(self, turn_num: int, elapsed_seconds: float) -> None:
        """Print API call completion with timing."""
        self.console.print(f"[dim]    ✓ Turn {turn_num} response received ({elapsed_seconds:.1f}s)[/dim]")

    def print_turn(
        self,
        turn_num: int,
        max_turns: int,
        response: str,
        code_cells: list[str],
        execution_results: list[dict]
    ) -> None:
        """Print a single turn with LLM response and execution."""
        self.console.print(f"\n[bold cyan]  Turn {turn_num}/{max_turns}[/bold cyan]")

        reasoning = re.sub(r'```python.*?```', '', response, flags=re.DOTALL).strip()

        if reasoning:
            # Show full reasoning without truncation
            self.console.print(f"[dim]  {reasoning}[/dim]\n")

        for i, (code, result) in enumerate(zip(code_cells, execution_results), 1):
            syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
            panel = Panel(
                syntax,
                title=f"[bold]Code Cell {i}[/bold]",
                border_style="blue",
                padding=(0, 1)
            )
            self.console.print(panel)

            if result["success"]:
                self.base.print_success("Code executed successfully")
                stdout = result["stdout"]
                if stdout.strip():
                    self.console.print(f"[green]  Output:[/green] {stdout.strip()}")
            else:
                self.base.print_error("Execution failed")
                stderr = result["stderr"]
                if stderr.strip():
                    self.console.print(f"[red]  Error:[/red] {stderr.strip()}")

    def print_trace_complete(self, success: bool, final_answer: Any, turns: int, elapsed_seconds: float | None = None) -> None:
        """Print trace completion status."""
        if success:
            self.base.print_success(f"Answer submitted: {final_answer}")
            if elapsed_seconds is not None:
                self.console.print(f"[dim]  Completed in {turns} turn(s) • {elapsed_seconds:.1f}s total[/dim]")
            else:
                self.console.print(f"[dim]  Completed in {turns} turn(s)[/dim]")
        else:
            self.base.print_error("Failed to produce an answer")
            if elapsed_seconds is not None:
                self.console.print(f"[dim]  Failed after {elapsed_seconds:.1f}s[/dim]")

    def print_triangulation_result(
        self,
        gold_trace,
        consistency_traces: list,
        verified: bool,
        float_tol: float
    ) -> None:
        """Print triangulation summary with verification status."""
        from src.datagen.teacher import answers_match, get_majority_answer

        self.console.print("\n[bold cyan]┌─ TRIANGULATION " + "─" * 42 + "┐[/bold cyan]")

        self.console.print("[bold cyan]│[/bold cyan]")
        self.console.print("[bold cyan]│[/bold cyan] [bold]Gold Trace:[/bold]")
        # Handle both dict (TraceDict) and object access patterns
        gold_answer = gold_trace.get("final_answer") if isinstance(gold_trace, dict) else gold_trace.final_answer
        gold_hash = gold_trace.get("final_answer_hash") if isinstance(gold_trace, dict) else gold_trace.final_answer_hash
        self.console.print(f"[bold cyan]│[/bold cyan]   Answer: {gold_answer}")
        self.console.print(f"[bold cyan]│[/bold cyan]   Hash:   {gold_hash}")

        self.console.print("[bold cyan]│[/bold cyan]")
        self.console.print("[bold cyan]│[/bold cyan] [bold]Consistency Results:[/bold]")

        for i, trace in enumerate(consistency_traces, 1):
            if trace is None:
                self.console.print(f"[bold cyan]│[/bold cyan]   Trace {i}: [red]FAILED[/red]")
                continue

            # Handle both dict (TraceDict) and object access patterns
            answer = trace.get("final_answer") if isinstance(trace, dict) else trace.final_answer
            answer_hash = trace.get("final_answer_hash") if isinstance(trace, dict) else trace.final_answer_hash

            matches = answers_match(
                gold_hash,
                answer_hash,
                gold_answer,
                answer,
                float_tol=float_tol
            )
            check = "✓" if matches else " "
            hash_display = f"{answer_hash[:8]}..." if isinstance(answer_hash, str) else "None"
            self.console.print(f"[bold cyan]│[/bold cyan]   Trace {i}: {answer}  (hash: {hash_display}) {check}")

        valid_answers = [
            (t.get("final_answer") if isinstance(t, dict) else t.final_answer) for t in consistency_traces
            if t is not None and (t.get("final_answer") if isinstance(t, dict) else t.final_answer) is not None
        ]
        majority_value, majority_count = get_majority_answer(valid_answers, float_tol=float_tol)

        self.console.print("[bold cyan]│[/bold cyan]")
        self.console.print(f"[bold cyan]│[/bold cyan] [bold]Majority:[/bold] {majority_value} ({majority_count}/{len(consistency_traces)} votes)")

        self.console.print("[bold cyan]│[/bold cyan]")
        self.console.print("[bold cyan]│[/bold cyan] [bold]Verification:[/bold]")
        self.console.print(f"[bold cyan]│[/bold cyan]   Gold value:     {gold_answer}")
        self.console.print(f"[bold cyan]│[/bold cyan]   Majority value: {majority_value}")
        self.console.print(f"[bold cyan]│[/bold cyan]   Tolerance:      ±{float_tol}")
        self.console.print("[bold cyan]│[/bold cyan]")

        if verified:
            self.console.print("[bold cyan]│[/bold cyan] [bold green]✓ VERIFIED[/bold green] (match within tolerance)")
        else:
            self.console.print("[bold cyan]│[/bold cyan] [bold red]✗ FAILED[/bold red] (no match)")

        self.console.print("[bold cyan]│[/bold cyan]")
        self.console.print("[bold cyan]" + "└" + "─" * 58 + "┘[/bold cyan]")

    def print_progress_summary(self, current: int, total: int, verified_count: int) -> None:
        """Print running progress summary."""
        rate = (verified_count / current * 100) if current > 0 else 0
        self.console.print(f"\n[bold]Progress:[/bold] {current}/{total} questions processed | {verified_count} verified ({rate:.1f}% rate)\n")
