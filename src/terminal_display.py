"""Terminal display functions for the CSV exploration agent."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import box

# Single canonical console instance
console = Console()

# Max chars to show in panels (keeps output readable)
MAX_ASSISTANT_CHARS = 2000
MAX_BOOTSTRAP_CHARS = 1500
MAX_TOOL_OUTPUT_CHARS = 1000


# =============================================================================
# Pipeline lifecycle
# =============================================================================

def header(csv_path: str):
    """Display the pipeline header."""
    console.print()
    console.print(Panel.fit(
        f"[bold blue]CSV Exploration Agent[/bold blue]\n"
        f"[dim]{csv_path} → 10 episodes[/dim]",
        border_style="blue",
    ))


def loading(csv_path: str):
    """Show loading message."""
    console.print(f"[dim]Loading {csv_path}...[/dim]")


def loaded(rows: int, cols: int):
    """Show loaded confirmation."""
    console.print(f"[green]✓[/green] {rows:,} rows × {cols} cols\n")


def bootstrap_start():
    """Show bootstrap starting message."""
    console.print("[cyan]Running bootstrap...[/cyan]")


def bootstrap_output(output: str):
    """Display bootstrap exploration results (compact)."""
    # Show truncated version
    display = output[:MAX_BOOTSTRAP_CHARS]
    if len(output) > MAX_BOOTSTRAP_CHARS:
        display += f"\n[dim]... ({len(output)} chars total)[/dim]"
    
    console.print(Panel(
        Text(display, style="dim"),
        title="[cyan]📊 Bootstrap[/cyan]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def cleanup():
    """Show cleanup message."""
    console.print("\n[dim]Done.[/dim]")


# =============================================================================
# Turn cycle
# =============================================================================

def turn_start(turn: int, max_turns: int):
    """Display turn separator."""
    console.print()
    console.rule(f"[bold]Turn {turn}/{max_turns}[/bold]", style="blue")


def thinking():
    """Return a spinner context manager for LLM thinking."""
    return console.status("[magenta]Thinking...[/magenta]", spinner="dots")


def assistant(response: str, turn: int):
    """Display the LLM's response (truncated for readability)."""
    # Truncate for display
    display = response[:MAX_ASSISTANT_CHARS]
    if len(response) > MAX_ASSISTANT_CHARS:
        display += f"\n\n[dim]... ({len(response)} chars total)[/dim]"
    
    console.print()
    console.print(Panel(
        Markdown(display),
        title=f"[magenta]🤖 Turn {turn}[/magenta]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def no_tool_call():
    """Warn about missing tool call."""
    console.print("[yellow]⚠ No <code> tool call found[/yellow]")


def tool_result(code: str, tool_name: str, output: str, success: bool, call_num: int):
    """Display a tool call and its result (compact)."""
    style = "green" if success else "red"
    icon = "✓" if success else "✗"
    
    # Truncate output
    display_output = output[:MAX_TOOL_OUTPUT_CHARS]
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        display_output += f"\n... ({len(output)} chars)"
    
    # Compact: tool name + result in one panel
    console.print(Panel(
        Text(display_output),
        title=f"[{style}]{icon}[/{style}] [yellow]{tool_name}[/yellow] [dim]#{call_num}[/dim]",
        border_style="dim",
        box=box.SIMPLE,
        padding=(0, 1),
    ))


# =============================================================================
# Done / episodes
# =============================================================================

def done_signal():
    """Show that agent signaled done."""
    console.print("\n[bold green]✓ DONE - extracting episodes...[/bold green]")


def parse_failed(response: str):
    """Show parse failure with response tail."""
    console.print("[red]✗ Failed to parse episodes[/red]")
    console.print(Panel(
        Text(response[-1500:] if len(response) > 1500 else response, style="dim"),
        title="[red]Response tail[/red]",
        border_style="red",
        box=box.SIMPLE,
    ))


def max_turns_reached(max_turns: int):
    """Show max turns warning."""
    console.print(f"\n[yellow]⚠ Max turns ({max_turns}) - forcing final output[/yellow]")


def generating_final():
    """Return spinner for final episode generation."""
    return console.status("[magenta]Generating final episodes...[/magenta]", spinner="dots")


def episodes_summary(episodes: list[dict]):
    """Display compact summary table of all episodes."""
    console.print()
    
    if not episodes:
        console.print("[yellow]No episodes parsed.[/yellow]")
        return
    
    # Count by difficulty
    counts = {"MEDIUM": 0, "HARD": 0, "VERY_HARD": 0}
    
    table = Table(box=box.SIMPLE, padding=(0, 1), collapse_padding=True)
    table.add_column("#", style="dim", width=2)
    table.add_column("Diff", width=6)
    table.add_column("Hooks", width=5)
    table.add_column("Question", max_width=70)
    
    for i, ep in enumerate(episodes, 1):
        if not isinstance(ep, dict):
            continue  # Skip non-dict items
        
        diff = ep.get("difficulty", "?")
        diff_color = {"MEDIUM": "cyan", "HARD": "yellow", "VERY_HARD": "red"}.get(diff, "white")
        counts[diff] = counts.get(diff, 0) + 1
        
        hooks = ep.get("hooks", [])
        question = ep.get("question_text", "?")
        if len(question) > 67:
            question = question[:67] + "..."
        
        table.add_row(
            str(i),
            f"[{diff_color}]{diff[:3]}[/{diff_color}]",  # MED/HAR/VER
            str(len(hooks)),
            question,
        )
    
    console.print(Panel(
        table,
        title=f"[green]✓ {len(episodes)} Episodes[/green]",
        border_style="green",
        box=box.ROUNDED,
    ))
    
    # Distribution
    dist = f"[dim]M={counts.get('MEDIUM', 0)} H={counts.get('HARD', 0)} VH={counts.get('VERY_HARD', 0)}[/dim]"
    console.print(dist)


def episode(ep: dict, idx: int):
    """Display a single episode (compact)."""
    if not isinstance(ep, dict):
        return  # Skip non-dict items
    
    difficulty = ep.get("difficulty", "?")
    diff_color = {"MEDIUM": "cyan", "HARD": "yellow", "VERY_HARD": "red"}.get(difficulty, "white")
    
    question = ep.get("question_text", "No question")
    hooks = ep.get("hooks", [])
    teacher_answers = ep.get("teacher_answers", {})
    
    # Header with question
    console.print()
    console.print(f"[bold {diff_color}]#{idx} [{difficulty}][/bold {diff_color}] {question[:100]}{'...' if len(question) > 100 else ''}")
    
    # Hooks as compact list
    if hooks:
        hook_lines = []
        for h in hooks:
            hid = h.get("id", "?")
            tool = h.get("tool", "?")
            deps = h.get("depends_on", [])
            ans = teacher_answers.get(hid, "?")
            
            # Format answer
            if isinstance(ans, float):
                ans_str = f"{ans:.3g}"
            elif isinstance(ans, dict):
                ans_str = "{...}"
            else:
                ans_str = str(ans)[:20]
            
            dep_str = f"←{','.join(deps)}" if deps else ""
            hook_lines.append(f"  [cyan]{hid}[/cyan] [dim]{tool}[/dim] = [green]{ans_str}[/green] {dep_str}")
        
        console.print("\n".join(hook_lines[:8]))  # Max 8 hooks shown
        if len(hooks) > 8:
            console.print(f"  [dim]... +{len(hooks) - 8} more hooks[/dim]")


def saved(path: str, count: int):
    """Show save confirmation."""
    console.print(f"\n[green]✓[/green] Saved {count} episodes → [bold]{path}[/bold]")


def interrupted():
    """Show interrupt message."""
    console.print("\n[yellow]Interrupted.[/yellow]")


def exception():
    """Print exception traceback."""
    console.print_exception()
