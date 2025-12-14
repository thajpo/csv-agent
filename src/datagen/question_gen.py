"""
LLM-based question generator for CSV datasets.

This script uses an LLM to:
1. Explore a dataset using Jupyter kernel
2. Document exploration observations
3. Generate questions with varying difficulty levels (EASY, MEDIUM, HARD, VERY_HARD)

Configuration is loaded from config.yaml.
"""
import json
import re
import sys
import textwrap
import yaml
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

from src.core.kernel import JupyterKernel
from src.core.model import APILLM
from src.core.conversation import ConversationHistory, CodeCellResult
from src.datagen.types import ExplorationTurn, ExplorationTrace
from src.core.prompts import EXPLORATION_SYSTEM_PROMPT, MIN_EXPLORATION_TURNS, get_exploration_continue_msg


class QuestionGenUI:
    """Handles all Rich console output for question generation."""
    
    def __init__(self):
        self.console = Console()
    
    def print_header(self, title: str, **kwargs) -> None:
        """Print a formatted header."""
        self.console.print(f"\n[bold green]{title}[/bold green]", **kwargs)
    
    def print_info(self, label: str, value: str) -> None:
        """Print a key-value info line."""
        self.console.print(f"[bold blue]{label}:[/bold blue] {value}")
    
    def print_turn_header(self, turn_num: int, max_turns: int) -> None:
        """Print the turn separator and header."""
        self.console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        self.console.print(f"[bold cyan]TURN {turn_num + 1}/{max_turns}[/bold cyan]")
        self.console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")
    
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
                self.console.print(Panel(
                    Markdown(response),
                    title="[bold yellow]LLM Response[/bold yellow]",
                    border_style="yellow"
                ))
        else:
            self.console.print(Panel(
                Markdown(response),
                title="[bold yellow]LLM Response[/bold yellow]",
                border_style="yellow"
            ))
    
    def print_code_cell(self, cell_num: int, code: str) -> None:
        """Display a code cell with syntax highlighting."""
        self.console.print(f"\n[bold magenta]Executing Cell {cell_num}[/bold magenta]")
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        self.console.print(Panel(
            syntax,
            title=f"[bold magenta]Code Cell {cell_num}[/bold magenta]",
            border_style="magenta"
        ))
    
    def print_execution_success(self, stdout: str) -> None:
        """Display successful execution result."""
        self.console.print("[bold green]✓ Execution Successful[/bold green]")
        if stdout.strip():
            self.console.print(Panel(
                stdout,
                title="[bold green]ACTUAL OUTPUT FROM CODE EXECUTION[/bold green]",
                border_style="green",
                padding=(1, 2)
            ))
        else:
            self.console.print("[dim]No output[/dim]")
    
    def print_execution_failure(self, error_message: str) -> None:
        """Display execution failure."""
        self.console.print("[bold red]✗ Execution Failed[/bold red]")
        self.console.print(Panel(
            error_message,
            title="[bold red]ERROR[/bold red]",
            border_style="red"
        ))
    
    def print_status(self, message: str, style: str = "yellow") -> None:
        """Print a status message."""
        self.console.print(f"[{style}]{message}[/{style}]")
    
    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[bold green]✓ {message}[/bold green]")
    
    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[bold red]✗ {message}[/bold red]")
    
    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[bold yellow]⚠ {message}[/bold yellow]")
    
    def print_saved_file(self, file_path: Path) -> None:
        """Print file save confirmation."""
        label = "questions" if "questions" in str(file_path) else "exploration trace"
        self.console.print(f"[bold green]💾 Saved {label} → {file_path}[/bold green]")
    
    def print_summary_header(self) -> None:
        """Print summary section header."""
        self.console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        self.console.print("[bold cyan]SUMMARY[/bold cyan]")
        self.console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
    
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
        self.console.print(f"[bold]Total questions:[/bold] {count}")
    
    def print_difficulty_header(self) -> None:
        """Print difficulty section header."""
        self.console.print("\n[bold]By difficulty:[/bold]")
    
    def print_difficulty_count(self, difficulty: str, count: int) -> None:
        """Print a difficulty count."""
        self.console.print(f"  [cyan]{difficulty}:[/cyan] {count}")
    
    def print_sample_questions_header(self) -> None:
        """Print sample questions section header."""
        self.console.print("\n[bold]Sample questions:[/bold]")
    
    def print_empty_line(self) -> None:
        """Print an empty line."""
        self.console.print()


# Create global UI instance
ui = QuestionGenUI()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    return {}


def extract_python_cells(response: str) -> list[str]:
    """
    Extract python code blocks, trimming malformed fences.

    Accepts ```python``` or ```py``` fences and tolerates missing closing backticks.
    """
    pattern = r"```(?:python|py)\n([\s\S]*?)(?:```|$)"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    cleaned_blocks = []

    for block in matches:
        cleaned = textwrap.dedent(block).replace("\r\n", "\n").strip()
        # Strip any trailing stray backticks the regex may have captured
        cleaned = cleaned.rstrip("`").strip()
        if cleaned:
            cleaned_blocks.append(cleaned)

    return cleaned_blocks


def try_parse_questions(response: str) -> list[dict] | None:
    """
    Parse if found and valid, None otherwise
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


def force_question_generation(llm: APILLM, conversation: ConversationHistory) -> list[dict]:
    """
    If model hasn't generated questions by max_turns, force it with a direct prompt.

    Returns:
        List of question dicts
    """
    # Add forcing message as user feedback
    conversation.add_user_feedback(
        "You've explored enough. Now generate the 13 questions in JSON format as specified in the system prompt."
    )

    # Get response
    messages = conversation.to_openai_messages()
    response = llm(messages)

    # Try to parse
    questions = try_parse_questions(response)

    if not questions:
        raise RuntimeError("Model failed to generate valid questions even after forcing. Check model output.")

    return questions


async def explore_and_generate_questions(
    csv_path: str,
    model: str,
    max_turns: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 6000,  # Increased default to allow full question generation
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
    ui.print_header(f"Question Generation Starting exploration of {csv_path}")
    ui.print_info("Model", model)
    ui.print_info("Max turns", str(max_turns))
    ui.print_empty_line()

    # 1. Setup
    kernel = await JupyterKernel.create(timeout=120, workdir=None, csv_path=csv_path)
    llm = APILLM(model=model, sampling_args={"temperature": temperature, "max_tokens": max_tokens})
    conversation = ConversationHistory(
        system_prompt=EXPLORATION_SYSTEM_PROMPT,
        max_messages=100,  # Keep full exploration in context
        max_context_tokens=100_000
    )

    # 2. Multi-turn exploration loop
    exploration_turns = []
    questions_generated = None

    for turn_num in range(max_turns):
        ui.print_turn_header(turn_num + 1, max_turns)

        # Get model response
        messages = conversation.to_openai_messages()
        ui.print_status("Generating LLM response...")
        response = await llm(messages)

        # Display LLM response
        ui.print_llm_response(response)

        # Extract code cells
        code_cells = extract_python_cells(response)
        ui.print_code_blocks_found(len(code_cells))

        # Execute code
        execution_results = []
        for i, code in enumerate(code_cells, 1):
            ui.print_code_cell(i, code)

            result = await kernel.execute(code)
            execution_results.append(CodeCellResult(
                code=code,
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr if not result.success else ""
            ))

            if result.success:
                ui.print_execution_success(result.stdout)
            else:
                ui.print_execution_failure(result.error_message)

        # Save turn
        turn = ExplorationTurn(
            turn_number=turn_num,
            reasoning=response,
            code_cells=code_cells,
            execution_results=execution_results,
            timestamp=datetime.now()
        )
        exploration_turns.append(turn)

        # Check if model signaled completion with <DONE>
        if "<DONE>" in response or "</DONE>" in response:
            # Enforce minimum exploration turns
            if turn_num < MIN_EXPLORATION_TURNS:
                ui.print_warning(f"Model tried to finish too early (turn {turn_num + 1}/{MIN_EXPLORATION_TURNS} minimum)")
                ui.print_status("Rejecting early completion - continuing exploration")
                # Don't parse questions, force more exploration
            else:
                ui.print_success("Model signaled completion with <DONE>")
                questions_generated = try_parse_questions(response)
                if questions_generated:
                    ui.print_success(f"Successfully extracted {len(questions_generated)} questions!")
                    break
                else:
                    ui.print_error("Found <DONE> but couldn't parse questions from response")
                    # Continue to allow retry

        # Build feedback with turn-aware message
        feedback = build_execution_feedback(execution_results)
        feedback += get_exploration_continue_msg(turn_num, MIN_EXPLORATION_TURNS)

        # Add turn to conversation
        conversation.add_assistant_response(response)
        conversation.add_user_feedback(feedback)

    # 3. Validate we got questions
    if not questions_generated:
        ui.print_warning("Model didn't generate questions. Forcing...")
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
    ui.print_saved_file(questions_file)

    # Save exploration trace
    trace_file = output_path / "exploration_trace.json"
    with open(trace_file, 'w') as f:
        json.dump(trace.model_dump(), f, indent=2, default=str)
    ui.print_saved_file(trace_file)

    # Cleanup
    kernel.shutdown()

    return questions_generated, trace


def main():
    # Load config
    config = load_config("config.yaml")

    # Extract config values (fail-fast on missing keys)
    csv_path = config["csv"]
    output_file = config["output"]
    max_turns = config["question_gen_max_turns"]
    temperature = config["sampling_args"]["temperature"]
    max_tokens = config["sampling_args"]["max_tokens"]
    model = config["question_gen_model"]

    # Set output directory from output file path
    output_dir = str(Path(output_file).parent)
    if output_dir == ".":
        output_dir = "."

    try:
        questions, trace = explore_and_generate_questions(
            csv_path=csv_path,
            model=model,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            output_dir=output_dir
        )

        # Print summary
        ui.print_summary_header()
        ui.print_total_questions(len(questions))

        # Count by difficulty
        difficulty_counts = {}
        for q in questions:
            diff = q.get("difficulty", "UNKNOWN")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        ui.print_difficulty_header()
        for diff, count in sorted(difficulty_counts.items()):
            ui.print_difficulty_count(diff, count)

        ui.print_sample_questions_header()
        for i, q in enumerate(questions[:3], 1):
            ui.print_question_panel(i, q)

        return 0

    except Exception as e:
        ui.print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
