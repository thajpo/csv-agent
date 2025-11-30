"""
Rich terminal entrypoint for the CSV exploration agent.

Usage:
    python -m src.rich_pipeline
    python -m src.rich_pipeline --csv data.csv --max-turns 10
    python -m src.rich_pipeline --output episodes.jsonl
"""

import argparse
import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.tree import Tree
from rich import box

from src.environment.kernel import JupyterKernel
from src.llm import APILLM
from src.tools import parse_tool_call, run_tool
from src.prompts import BOOTSTRAP_CODE, build_prompt, DEFAULT_DATASET_DESCRIPTION


console = Console()

MAX_OUTPUT_CHARS = 10000


def extract_code_blocks(text: str) -> list[str]:
    """Extract all code between <code> and </code> tags."""
    pattern = r'<code>(.*?)</code>'
    return re.findall(pattern, text, re.DOTALL)


def extract_json_episodes(text: str) -> list[dict] | None:
    """
    Extract JSON episode array from the final output.
    Looks for ```json ... ``` block containing an array.
    """
    # Try to find JSON code block
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, list) and len(data) > 0:
                return data
        except json.JSONDecodeError:
            continue
    
    # Fallback: try to find raw JSON array
    try:
        # Find array start
        start = text.find('[')
        if start != -1:
            # Try to parse from there
            bracket_count = 0
            end = start
            for i, char in enumerate(text[start:], start):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end = i + 1
                        break
            
            candidate = text[start:end]
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def execute_tool_call(code: str, df: pd.DataFrame) -> tuple[str, str, bool]:
    """
    Parse and execute a tool call.
    
    Returns:
        (tool_name, output, success)
    """
    result = parse_tool_call(code)
    
    if isinstance(result, str):
        return ("error", result, False)
    
    tool_name, params = result
    output = run_tool(tool_name, df, params)
    return (tool_name, output, True)


def display_bootstrap(output: str):
    """Display bootstrap exploration results."""
    console.print()
    console.print(Panel(
        Text(output[:3000] + "..." if len(output) > 3000 else output, style="dim"),
        title="[bold cyan]📊 Initial Data Exploration[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    ))


def display_assistant_response(response: str, turn: int):
    """Display the LLM's response with nice formatting."""
    console.print()
    console.print(Panel(
        Markdown(response),
        title=f"[bold magenta]🤖 Assistant (Turn {turn})[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    ))


def display_tool_call(code: str, tool_name: str, output: str, success: bool, call_num: int):
    """Display a tool call and its result."""
    console.print()
    console.print(f"[bold yellow]⚡ Tool Call {call_num}[/bold yellow]")
    console.print(Panel(
        Syntax(code.strip(), "json", theme="monokai"),
        border_style="yellow",
        box=box.SIMPLE,
    ))
    
    if success:
        style = "green"
        icon = "✓"
    else:
        style = "red"
        icon = "✗"
    
    display_output = output
    if len(display_output) > 2000:
        display_output = display_output[:2000] + f"\n... ({len(output)} chars total)"
    
    console.print(Panel(
        Text(display_output),
        title=f"[{style}]{icon} {tool_name}[/{style}]",
        border_style=style,
        box=box.SIMPLE,
    ))


def display_episode(episode: dict, idx: int):
    """Display a single episode with its hooks and answers."""
    difficulty = episode.get("difficulty", "?")
    diff_color = {"MEDIUM": "cyan", "HARD": "yellow", "VERY_HARD": "red"}.get(difficulty, "white")
    
    # Episode header
    console.print()
    console.print(Panel(
        f"[bold]{episode.get('question_text', 'No question')}[/bold]",
        title=f"[bold {diff_color}]Episode {idx} [{difficulty}][/bold {diff_color}]",
        border_style=diff_color,
        box=box.ROUNDED,
    ))
    
    # Hooks tree
    hooks = episode.get("hooks", [])
    teacher_answers = episode.get("teacher_answers", {})
    
    if hooks:
        tree = Tree("[bold]Hooks[/bold]")
        for hook in hooks:
            hook_id = hook.get("id", "?")
            tool = hook.get("tool", "?")
            deps = hook.get("depends_on", [])
            answer = teacher_answers.get(hook_id, "?")
            
            # Format answer nicely
            if isinstance(answer, dict):
                answer_str = json.dumps(answer, indent=2)
            else:
                answer_str = str(answer)
            
            dep_str = f" [dim](deps: {', '.join(deps)})[/dim]" if deps else ""
            hook_branch = tree.add(f"[cyan]{hook_id}[/cyan] → [magenta]{tool}[/magenta]{dep_str}")
            
            # Show params
            params = hook.get("params", {})
            if params:
                params_str = json.dumps(params, indent=2)
                hook_branch.add(f"[dim]params:[/dim] {params_str}")
            
            # Show answer
            hook_branch.add(f"[green]answer:[/green] {answer_str}")
        
        console.print(tree)
    
    # Solution trace
    trace = episode.get("solution_trace", "")
    if trace:
        console.print(f"\n[dim]Solution trace:[/dim] {trace[:200]}{'...' if len(trace) > 200 else ''}")


def display_episodes_summary(episodes: list[dict]):
    """Display summary table of all episodes."""
    console.print()
    console.print(Panel(
        "[bold green]🎯 Episode Generation Complete![/bold green]",
        box=box.DOUBLE,
        border_style="green",
    ))
    
    if not episodes:
        console.print("[yellow]No episodes were generated.[/yellow]")
        return
    
    # Summary table
    table = Table(
        title=f"[bold]Generated Episodes ({len(episodes)} total)[/bold]",
        box=box.ROUNDED,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Difficulty", width=10)
    table.add_column("Hooks", width=6)
    table.add_column("Question", style="white", max_width=60)
    
    # Count by difficulty
    counts = {"MEDIUM": 0, "HARD": 0, "VERY_HARD": 0}
    
    for i, ep in enumerate(episodes, 1):
        diff = ep.get("difficulty", "?")
        diff_color = {"MEDIUM": "cyan", "HARD": "yellow", "VERY_HARD": "red"}.get(diff, "white")
        counts[diff] = counts.get(diff, 0) + 1
        
        hooks = ep.get("hooks", [])
        question = ep.get("question_text", "?")
        if len(question) > 57:
            question = question[:57] + "..."
        
        table.add_row(
            str(i),
            f"[{diff_color}]{diff}[/{diff_color}]",
            str(len(hooks)),
            question,
        )
    
    console.print(table)
    
    # Distribution check
    console.print(f"\n[dim]Distribution: MEDIUM={counts.get('MEDIUM', 0)}, HARD={counts.get('HARD', 0)}, VERY_HARD={counts.get('VERY_HARD', 0)}[/dim]")
    expected = {"MEDIUM": 3, "HARD": 3, "VERY_HARD": 4}
    if counts != expected:
        console.print(f"[yellow]⚠ Expected: MEDIUM=3, HARD=3, VERY_HARD=4[/yellow]")


def save_episodes(episodes: list[dict], output_path: str, metadata: dict):
    """Save episodes to JSONL file."""
    path = Path(output_path)
    
    with open(path, "w") as f:
        for ep in episodes:
            # Merge with metadata
            full_episode = {**metadata, **ep}
            f.write(json.dumps(full_episode) + "\n")
    
    console.print(f"\n[green]✓[/green] Saved {len(episodes)} episodes to [bold]{path}[/bold]")


def run_pipeline(
    csv_path: str = "data.csv",
    dataset_description: str = DEFAULT_DATASET_DESCRIPTION,
    max_turns: int = 10,
    output_path: str | None = None,
):
    """Run the full exploration pipeline with rich output."""
    
    # Header
    console.print()
    console.print(Panel(
        "[bold]CSV Exploration Agent[/bold]\n"
        f"[dim]Dataset: {csv_path}[/dim]\n"
        f"[dim]Target: 10 episodes (3 MEDIUM, 3 HARD, 4 VERY_HARD)[/dim]",
        box=box.DOUBLE_EDGE,
        border_style="blue",
    ))
    
    # Load dataframe
    console.print(f"\n[dim]Loading {csv_path}...[/dim]")
    df = pd.read_csv(csv_path)
    console.print(f"[green]✓[/green] Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Setup temp workdir
    workdir = tempfile.mkdtemp()
    shutil.copy(csv_path, workdir)
    
    episodes = []
    n_turns = 0
    
    with JupyterKernel(workdir=workdir) as kernel:
        llm = APILLM()
        
        try:
            # Bootstrap exploration
            console.print("\n[bold cyan]Running bootstrap exploration...[/bold cyan]")
            bootstrap_result = kernel.execute(BOOTSTRAP_CODE)
            bootstrap_output = bootstrap_result.stdout or "[no output]"
            display_bootstrap(bootstrap_output)
            
            # Build prompt and start conversation
            system_prompt = build_prompt(dataset_description, bootstrap_output)
            conversation = [{"role": "user", "content": system_prompt}]
            
            for turn in range(1, max_turns + 1):
                n_turns = turn
                console.print()
                console.rule(f"[bold blue]Turn {turn}/{max_turns}[/bold blue]")
                
                # Get LLM response with spinner
                with console.status("[bold magenta]Thinking...[/bold magenta]", spinner="dots"):
                    response = llm(conversation)
                
                display_assistant_response(response, turn)
                conversation.append({"role": "assistant", "content": response})
                
                # Check for done signal
                if re.search(r'^DONE\b', response, re.MULTILINE):
                    console.print("\n[bold green]Agent signaled DONE - extracting episodes...[/bold green]")
                    
                    # Extract JSON episodes
                    episodes = extract_json_episodes(response)
                    
                    if episodes:
                        display_episodes_summary(episodes)
                        for i, ep in enumerate(episodes, 1):
                            display_episode(ep, i)
                    else:
                        console.print("[red]Failed to parse episodes from response.[/red]")
                        console.print("[dim]Raw response tail:[/dim]")
                        console.print(response[-2000:] if len(response) > 2000 else response)
                    break
                
                # Execute tool calls
                code_blocks = extract_code_blocks(response)
                
                if not code_blocks:
                    feedback = "No tool call found. Use <code>{\"tool\": \"...\", ...}</code> to explore the data."
                    console.print(f"\n[yellow]{feedback}[/yellow]")
                else:
                    results = []
                    for i, code in enumerate(code_blocks, 1):
                        tool_name, output, success = execute_tool_call(code.strip(), df)
                        display_tool_call(code, tool_name, output, success, i)
                        results.append(f"[Call {i}]\n[{tool_name}]\n{output}")
                    
                    feedback = "\n\n".join(results)
                
                # Truncate if needed
                if len(feedback) > MAX_OUTPUT_CHARS:
                    feedback = feedback[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
                
                feedback += "\n\nContinue exploring. Note candidate questions as you go. When ready, write DONE and output your final 10 episodes as JSON."
                conversation.append({"role": "user", "content": feedback})
            
            else:
                console.print(f"\n[yellow]Reached max turns ({max_turns}) - requesting final output...[/yellow]")
                
                # Ask for final episodes
                conversation.append({
                    "role": "user",
                    "content": "You've reached the turn limit. Please output your final 10 episodes now as a JSON array. Write DONE then the JSON."
                })
                
                with console.status("[bold magenta]Generating final episodes...[/bold magenta]", spinner="dots"):
                    response = llm(conversation)
                
                display_assistant_response(response, max_turns + 1)
                episodes = extract_json_episodes(response)
                
                if episodes:
                    display_episodes_summary(episodes)
                    for i, ep in enumerate(episodes, 1):
                        display_episode(ep, i)
                else:
                    console.print("[red]Failed to parse final episodes.[/red]")
        
        finally:
            del llm
            torch.cuda.empty_cache()
    
    # Save if output path provided
    if output_path and episodes:
        metadata = {
            "dataset_id": Path(csv_path).stem,
            "generation_timestamp": datetime.now().isoformat(),
            "teacher_model": "grok-4.1-fast",
            "n_turns": n_turns,
        }
        save_episodes(episodes, output_path, metadata)
    
    console.print("\n[dim]Cleanup complete.[/dim]")
    return episodes


def main():
    parser = argparse.ArgumentParser(description="CSV Exploration Agent with Rich Terminal UI")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV file")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum conversation turns")
    parser.add_argument("--description", default=None, help="Dataset description (uses default if not provided)")
    parser.add_argument("--output", "-o", default=None, help="Output path for episodes JSONL")
    args = parser.parse_args()
    
    description = args.description or DEFAULT_DATASET_DESCRIPTION
    
    try:
        run_pipeline(
            csv_path=args.csv,
            dataset_description=description,
            max_turns=args.max_turns,
            output_path=args.output,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception:
        console.print_exception()


if __name__ == "__main__":
    main()
