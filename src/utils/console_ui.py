"""
Shared Rich console UI components for data generation pipelines.

This module provides reusable UI utilities for question_gen and episode_gen
to ensure consistent, beautiful terminal output.
"""
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown


def truncate_value(value: Any, max_len: int = 80) -> str:
    """
    Truncate large values for display.

    - DataFrames: show shape only
    - Dicts: show first 2 keys + "..."
    - Lists: show first 2 items + "..."
    - Strings: truncate with "..."
    """
    # Check for pandas DataFrame/Series
    type_name = type(value).__name__
    if type_name == "DataFrame":
        return f"<DataFrame {value.shape[0]}×{value.shape[1]}>"
    if type_name == "Series":
        return f"<Series len={len(value)}>"
    if type_name == "ndarray":
        return f"<ndarray shape={value.shape}>"

    if isinstance(value, dict):
        if len(value) <= 2:
            s = str(value)
        else:
            keys = list(value.keys())[:2]
            preview = ", ".join(f"{k!r}: {truncate_value(value[k], 20)}" for k in keys)
            s = "{" + preview + ", ...}"
        return s[:max_len] + "..." if len(s) > max_len else s

    if isinstance(value, (list, tuple)):
        if len(value) <= 2:
            s = str(value)
        else:
            preview = ", ".join(truncate_value(v, 20) for v in value[:2])
            bracket = "[" if isinstance(value, list) else "("
            end = "]" if isinstance(value, list) else ")"
            s = f"{bracket}{preview}, ...{end}"
        return s[:max_len] + "..." if len(s) > max_len else s

    s = str(value)
    return s[:max_len] + "..." if len(s) > max_len else s


class ConsoleUI:
    """Base class for Rich console output with common formatting utilities."""

    def __init__(self):
        self.console = Console()

    # ============= Headers and Sections =============

    def print_header(self, title: str, **kwargs) -> None:
        """Print a formatted header."""
        self.console.print(f"\n[bold green]{title}[/bold green]", **kwargs)

    def print_section(self, title: str, width: int = 60) -> None:
        """Print a section separator with title."""
        self.console.print(f"\n[bold cyan]{'=' * width}[/bold cyan]")
        self.console.print(f"[bold cyan]{title}[/bold cyan]")
        self.console.print(f"[bold cyan]{'=' * width}[/bold cyan]\n")

    def print_subsection(self, title: str, width: int = 60) -> None:
        """Print a subsection separator with title."""
        self.console.print(f"\n[bold yellow]{'-' * width}[/bold yellow]")
        self.console.print(f"[bold yellow]{title}[/bold yellow]")
        self.console.print(f"[bold yellow]{'-' * width}[/bold yellow]\n")

    # ============= Info and Key-Value Display =============

    def print_info(self, label: str, value: str) -> None:
        """Print a key-value info line."""
        self.console.print(f"[bold blue]{label}:[/bold blue] {value}")

    def print_key_value(self, key: str, value: Any, indent: int = 0) -> None:
        """Print key-value pair with optional indentation."""
        prefix = " " * indent
        self.console.print(f"{prefix}[bold]{key}:[/bold] {value}")

    # ============= Status Indicators =============

    def print_status(self, message: str, style: str = "yellow") -> None:
        """Print a status message."""
        self.console.print(f"[{style}]{message}[/{style}]")

    def print_success(self, message: str) -> None:
        """Print a success message with checkmark."""
        self.console.print(f"[bold green]✓ {message}[/bold green]")

    def print_error(self, message: str) -> None:
        """Print an error message with X mark."""
        self.console.print(f"[bold red]✗ {message}[/bold red]")

    def print_warning(self, message: str) -> None:
        """Print a warning message with warning symbol."""
        self.console.print(f"[bold yellow]⚠ {message}[/bold yellow]")

    # ============= Code Display =============

    def print_code_panel(
        self,
        code: str,
        title: str = "Code",
        language: str = "python",
        border_style: str = "magenta"
    ) -> None:
        """Display code in a panel with syntax highlighting."""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(Panel(
            syntax,
            title=f"[bold {border_style}]{title}[/bold {border_style}]",
            border_style=border_style
        ))

    def print_code_block(self, code: str, language: str = "python") -> None:
        """Display code with syntax highlighting (no panel)."""
        syntax = Syntax(code, language, theme="monokai", line_numbers=False)
        self.console.print(syntax)

    # ============= Execution Results =============

    def print_execution_success(self, stdout: str = "") -> None:
        """Display successful execution result."""
        self.console.print("[bold green]✓ Execution Successful[/bold green]")
        if stdout.strip():
            self.console.print(Panel(
                stdout,
                title="[bold green]OUTPUT[/bold green]",
                border_style="green",
                padding=(1, 2)
            ))
        else:
            self.console.print("[dim]  No output[/dim]")

    def print_execution_failure(self, error_message: str) -> None:
        """Display execution failure."""
        self.console.print("[bold red]✗ Execution Failed[/bold red]")
        self.console.print(Panel(
            error_message,
            title="[bold red]ERROR[/bold red]",
            border_style="red"
        ))

    # ============= Panels and Containers =============

    def print_panel(
        self,
        content: str,
        title: str | None = None,
        border_style: str = "cyan",
        padding: tuple[int, int] = (0, 1)
    ) -> None:
        """Print content in a bordered panel."""
        panel_kwargs = {
            "border_style": border_style,
            "padding": padding
        }
        if title:
            panel_kwargs["title"] = f"[bold {border_style}]{title}[/bold {border_style}]"

        self.console.print(Panel(content, **panel_kwargs))

    def print_markdown_panel(
        self,
        content: str,
        title: str | None = None,
        border_style: str = "yellow"
    ) -> None:
        """Print markdown content in a panel."""
        panel_kwargs = {"border_style": border_style}
        if title:
            panel_kwargs["title"] = f"[bold {border_style}]{title}[/bold {border_style}]"

        self.console.print(Panel(Markdown(content), **panel_kwargs))

    # ============= Utility =============

    def print_empty_line(self) -> None:
        """Print an empty line."""
        self.console.print()

    def print_separator(self, char: str = "─", width: int = 60, style: str = "dim") -> None:
        """Print a horizontal separator line."""
        self.console.print(f"[{style}]{char * width}[/{style}]")

    def print_raw(self, text: str, **kwargs) -> None:
        """Print raw text (delegates to console.print)."""
        self.console.print(text, **kwargs)
