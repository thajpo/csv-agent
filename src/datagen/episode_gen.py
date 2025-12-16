"""
Episode generation pipeline.

This script:
1. Loads questions from CSV (TODO: implement)
2. Runs teacher triangulation on each question
3. Saves verified episodes to disk

Usage:
    python -m src.authoring.episode_gen
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import uuid
import yaml
from typing import Any

from rich.panel import Panel
from rich.syntax import Syntax

from src.datagen.teacher import batch_triangulate
from src.core.prompts import generate_data_overview, DEFAULT_DATASET_DESCRIPTION
from src.core.types import Episode, EpisodeJSONL, Question, ExecutionTrace

from src.core.ui import ConsoleUI


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
        total_calls = n_questions * (n_consistency + 1)  # +1 for gold trace

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

    def print_question_header(
        self,
        q_num: int,
        total: int,
        question: dict
    ) -> None:
        """Print question section header."""
        difficulty = question.get("difficulty", "UNKNOWN")
        n_steps = question.get("n_steps", "?")
        q_text = question.get("question", "")
        hint = question.get("hint", "")

        self.console.print("\n[bold magenta]" + "┌" + "─" * 58 + "┐[/bold magenta]")
        self.console.print(f"[bold magenta]│ QUESTION {q_num}/{total} - {difficulty} ({n_steps} steps)" + " " * (58 - len(f" QUESTION {q_num}/{total} - {difficulty} ({n_steps} steps)")) + "│[/bold magenta]")
        self.console.print("[bold magenta]" + "├" + "─" * 58 + "┤[/bold magenta]")

        # Word wrap question text
        import textwrap
        wrapped_q = textwrap.fill(q_text, width=56)
        for line in wrapped_q.split("\n"):
            padding = 56 - len(line)
            self.console.print(f"[bold magenta]│[/bold magenta] {line}" + " " * padding + " [bold magenta]│[/bold magenta]")

        if hint:
            self.console.print("[bold magenta]│" + " " * 58 + "│[/bold magenta]")
            self.console.print("[bold magenta]│[/bold magenta] [dim]Hint:[/dim]" + " " * 51 + "[bold magenta]│[/bold magenta]")
            wrapped_h = textwrap.fill(hint, width=50)
            for line in wrapped_h.split("\n"):
                padding = 56 - len(line) - 6  # -6 for "      " prefix
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

        # Extract just the reasoning (non-code part)
        import re
        reasoning = re.sub(r'```python.*?```', '', response, flags=re.DOTALL).strip()

        if reasoning:
            # Show truncated reasoning
            max_len = 200
            if len(reasoning) > max_len:
                reasoning = reasoning[:max_len] + "..."
            self.console.print(f"[dim]  {reasoning}[/dim]\n")

        # Show code cells
        for i, (code, result) in enumerate(zip(code_cells, execution_results), 1):
            # Syntax highlight the code
            syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
            panel = Panel(
                syntax,
                title=f"[bold]Code Cell {i}[/bold]",
                border_style="blue",
                padding=(0, 1)
            )
            self.console.print(panel)

            # Show execution result
            if result.get("success"):
                self.base.print_success("Code executed successfully")
                stdout = result.get("stdout", "")
                if stdout.strip():
                    self.console.print(f"[green]  Output:[/green] {stdout.strip()}")
            else:
                self.base.print_error("Execution failed")
                stderr = result.get("stderr", "")
                if stderr.strip():
                    self.console.print(f"[red]  Error:[/red] {stderr.strip()}")

    def print_trace_complete(self, success: bool, final_answer: Any, turns: int) -> None:
        """Print trace completion status."""
        if success:
            self.base.print_success(f"Answer submitted: {final_answer}")
            self.console.print(f"[dim]  Completed in {turns} turn(s)[/dim]")
        else:
            self.base.print_error("Failed to produce an answer")

    def print_triangulation_result(
        self,
        gold_trace: ExecutionTrace,
        consistency_traces: list[ExecutionTrace],
        verified: bool,
        float_tol: float
    ) -> None:
        """Print triangulation summary with verification status."""
        from collections import Counter

        self.console.print("\n[bold cyan]┌─ TRIANGULATION " + "─" * 42 + "┐[/bold cyan]")

        # Gold answer
        self.console.print("[bold cyan]│[/bold cyan]")
        self.console.print("[bold cyan]│[/bold cyan] [bold]Gold Trace:[/bold]")
        self.console.print(f"[bold cyan]│[/bold cyan]   Answer: {gold_trace.final_answer}")
        self.console.print(f"[bold cyan]│[/bold cyan]   Hash:   {gold_trace.final_answer_hash}")

        # Consistency results
        self.console.print("[bold cyan]│[/bold cyan]")
        self.console.print("[bold cyan]│[/bold cyan] [bold]Consistency Results:[/bold]")

        for i, trace in enumerate(consistency_traces, 1):
            answer = trace.final_answer
            answer_hash = trace.final_answer_hash

            # Check if matches gold
            from src.datagen.teacher import answers_match
            matches = answers_match(
                gold_trace.final_answer_hash,
                answer_hash,
                gold_trace.final_answer,
                answer,
                float_tol=float_tol
            )
            check = "✓" if matches else " "
            hash_display = f"{answer_hash[:8]}..." if isinstance(answer_hash, str) else "None"
            self.console.print(f"[bold cyan]│[/bold cyan]   Trace {i}: {answer}  (hash: {hash_display}) {check}")

        # Majority calculation
        consistency_hashes = [
            t.final_answer_hash for t in consistency_traces
            if t.final_answer_hash is not None
        ]
        if consistency_hashes:
            hash_counts = Counter(consistency_hashes)
            majority_hash, majority_count = hash_counts.most_common(1)[0]
            majority_value = next(
                (t.final_answer for t in consistency_traces if t.final_answer_hash == majority_hash),
                None
            )
        else:
            majority_hash, majority_count, majority_value = None, 0, None

        self.console.print("[bold cyan]│[/bold cyan]")
        self.console.print(f"[bold cyan]│[/bold cyan] [bold]Majority:[/bold] {majority_value} ({majority_count}/{len(consistency_traces)} votes)")

        # Verification
        self.console.print("[bold cyan]│[/bold cyan]")
        self.console.print("[bold cyan]│[/bold cyan] [bold]Verification:[/bold]")
        self.console.print(f"[bold cyan]│[/bold cyan]   Gold value:     {gold_trace.final_answer}")
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


# Create global UI instance
ui = EpisodeGenUI()


def save_episode(episode: Episode, output_dir: Path) -> Path:
    """
    Save episode as JSON file.

    Args:
        episode: Episode to save
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename: {id}.json
    filepath = output_dir / f"{episode.id}.json"

    with open(filepath, 'w') as f:
        json.dump(episode.model_dump(), f, indent=2, default=str)

    return filepath


def load_questions(questions_path: str) -> list[dict]:
    """Load questions from JSON file."""
    with open(questions_path) as f:
        return json.load(f)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    with open(config_file) as f:
        return yaml.safe_load(f)


def main():
    # Load config
    config = load_config()

    # Extract config values (fail-fast on missing keys)
    csv_path = config["csv"]
    teacher_model = config["teacher_model"]
    max_turns = config["max_turns"]
    temperature = config["sampling_args"]["temperature"]
    max_tokens = config["sampling_args"]["max_tokens"]
    n_consistency = config["n_consistency"]
    verified_only = config["verified_only"]
    float_tol = config.get("float_tolerance", 0.1)

    # Load questions from question_gen.py output
    questions_file = config.get("questions_json", "question/questions.json")
    questions = load_questions(questions_file)

    # Output as single JSONL file
    output_jsonl = Path(config.get("episodes_jsonl", "episodes/episodes.jsonl"))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)



    # Display pipeline header
    ui.print_pipeline_header(
        n_questions=len(questions),
        n_consistency=n_consistency,
        csv_path=csv_path,
        model=teacher_model,
        float_tol=float_tol,
        output_file=str(output_jsonl)
    )

    # Generate data overview
    data_overview = generate_data_overview(csv_path)

    # Sampling args
    sampling_args = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Run batch triangulation with UI
    results = batch_triangulate(
        csv_path=csv_path,
        questions=questions,
        model=teacher_model,  # Required positional arg (3rd)
        n_consistency=n_consistency,
        dataset_description=DEFAULT_DATASET_DESCRIPTION,
        data_overview=data_overview,
        max_turns=max_turns,
        sampling_args=sampling_args,

        ui=ui,
        float_tol=float_tol,
    )

    # Convert to JSONL episodes and save
    episodes_jsonl = []
    episodes_verified = 0

    for q_dict, gold_trace, gold_conversation, system_prompt, consistency_results, verified in results:
        # Create Question object (question, metadata)
        question_obj = Question(
            question_text=q_dict["question"],
            hint=q_dict.get("hint"),
            difficulty=q_dict.get("difficulty"),
            n_steps=q_dict.get("n_steps"),
        )

        # Extract consistency traces (ignore conversations)
        consistency_traces = [trace for trace, _ in consistency_results]
        consistency_conversations = [conv for _, conv in consistency_results]

        # Create Episode object
        episode = Episode(
            id=str(uuid.uuid4()),
            question=question_obj,
            teacher_trace=gold_trace,
            consistency_traces=consistency_traces,
            verified=verified,
            timestamp=datetime.now(),
        )

        # Convert to JSONL format
        episode_jsonl = EpisodeJSONL.from_episode(
            episode=episode,
            gold_conversation=gold_conversation,
            system_prompt=system_prompt,
            consistency_conversations=consistency_conversations,
        )

        # Save if verified OR verified_only is False
        if verified or not verified_only:
            episodes_jsonl.append(episode_jsonl)
            if verified:
                episodes_verified += 1

    # Write JSONL file (one episode per line)
    with open(output_jsonl, 'w') as f:
        for ep in episodes_jsonl:
            f.write(json.dumps(ep.model_dump(), default=str) + '\n')

    # Display final summary
    ui.base.print_section("PIPELINE COMPLETE")
    ui.base.print_key_value("Output file", str(output_jsonl))
    ui.base.print_key_value("Total questions", len(questions))
    ui.base.print_key_value("Episodes saved", len(episodes_jsonl))
    ui.base.print_key_value("Episodes verified", episodes_verified)
    verification_rate = episodes_verified / len(questions) * 100 if questions else 0.0
    ui.base.print_key_value("Verification rate", f"{verification_rate:.1f}%")

    if verification_rate == 100:
        ui.base.print_success("All episodes verified!")
    elif verification_rate >= 80:
        ui.base.print_success(f"High verification rate: {verification_rate:.1f}%")
    elif verification_rate >= 50:
        ui.base.print_warning(f"Moderate verification rate: {verification_rate:.1f}%")
    else:
        ui.base.print_error(f"Low verification rate: {verification_rate:.1f}%")

    ui.base.print_empty_line()

    return 0


if __name__ == "__main__":
    sys.exit(main())
