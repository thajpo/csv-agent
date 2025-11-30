"""Minimal rich terminal output for CSV exploration agent."""

from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel

console = Console()

# =============================================================================
# Lifecycle
# =============================================================================

def header(csv_path: str):
    console.print(f"\n[bold blue]CSV Agent[/bold blue] → {csv_path}\n")

def loading(csv_path: str):
    console.print(f"[dim]Loading {csv_path}...[/dim]", end=" ")

def loaded(rows: int, cols: int):
    console.print(f"[green]✓[/green] {rows:,} × {cols}")

def bootstrap_start():
    pass

def bootstrap_output(output: str):
    console.print(Panel(output, title="[cyan]Bootstrap[/cyan]", border_style="dim"))

def cleanup():
    console.print("[dim]Done.[/dim]")

# =============================================================================
# Turn cycle
# =============================================================================

def turn_start(turn: int, max_turns: int):
    console.rule(f"[bold]Turn {turn}/{max_turns}[/bold]", style="blue")

@contextmanager
def thinking():
    with console.status("[magenta]Thinking...[/magenta]", spinner="dots"):
        yield

def assistant(response: str, turn: int):
    console.print(Panel(response, title="[magenta]Assistant[/magenta]", border_style="magenta"))

def no_tool_call():
    console.print("[yellow]⚠ No tool call[/yellow]")

def tool_result(code: str, tool_name: str, output: str, success: bool, call_num: int):
    style = "green" if success else "red"
    console.print(f"[{style}]{'✓' if success else '✗'}[/{style}] [yellow]{tool_name}[/yellow]")
    console.print(f"[dim]{output}[/dim]")

# =============================================================================
# Done / episodes
# =============================================================================

def done_signal():
    console.print("\n[bold green]✓ DONE[/bold green]")

def parse_failed(response: str):
    console.print("[red]✗ Failed to parse episodes[/red]")

def max_turns_reached(max_turns: int):
    console.print(f"[yellow]Max turns ({max_turns})[/yellow]")

@contextmanager
def generating_final():
    with console.status("[magenta]Generating...[/magenta]", spinner="dots"):
        yield

def episodes_summary(episodes: list[dict]):
    console.print(f"\n[green]✓ {len(episodes)} episodes[/green]")

def episode(ep: dict, idx: int):
    if not isinstance(ep, dict):
        return
    diff = ep.get("difficulty", "?")
    q = ep.get("question_text", "?")
    n_hooks = len(ep.get("hooks", []))
    console.print(f"  [dim]{idx}.[/dim] [{diff}] ({n_hooks}h) {q}")

def saved(path: str, count: int):
    console.print(f"[green]✓[/green] Saved {count} → [bold]{path}[/bold]")

def interrupted():
    console.print("[yellow]Interrupted.[/yellow]")

def exception():
    console.print_exception()
