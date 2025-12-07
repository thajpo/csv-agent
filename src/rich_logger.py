"""
Rich logging handler for CSV agent.

This module provides a logging.Handler implementation that formats
log records using Rich console output. This allows clean separation
between RL logic (Environment) and presentation (RichHandler).
"""

import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel


class RichHandler(logging.Handler):
    """
    Logging handler that formats events with Rich console output.

    This handler receives log events from Environment and formats them
    with Rich formatting (colors, panels, rules, etc.).
    """

    def __init__(self, rollout_config):
        """
        Initialize RichHandler.

        Args:
            rollout_config: RolloutConfig with rollout-specific configuration
        """
        super().__init__()
        self.console = Console()
        self.rollout_config = rollout_config
        self._status = None  # For tracking spinner state

    def emit(self, record: logging.LogRecord):
        """
        Handle a log record by formatting it with Rich.

        This method is called by Python's logging system whenever
        logger.info() is called. It dispatches to specific handlers
        based on the event type (record.msg).

        Args:
            record: LogRecord containing msg and extra fields
        """
        try:
            # Dispatch based on event type
            if record.msg == "episode_start":
                self._handle_episode_start(record)
            elif record.msg == "turn_start":
                self._handle_turn_start(record)
            elif record.msg == "model_thinking_start":
                self._handle_model_thinking_start(record)
            elif record.msg == "model_thinking_end":
                self._handle_model_thinking_end(record)
            elif record.msg == "model_response":
                self._handle_model_response(record)
            elif record.msg == "tool_executed":
                self._handle_tool_executed(record)
            elif record.msg == "no_tool_call":
                self._handle_no_tool_call(record)
            elif record.msg == "episode_complete":
                self._handle_episode_complete(record)
            elif record.msg == "max_turns_reached":
                self._handle_max_turns_reached(record)
            else:
                # Fallback for unknown events
                self.console.print(f"[dim]{record.msg}[/dim]")
        except Exception:
            # Don't let logging errors crash the application
            self.handleError(record)

    def _handle_episode_start(self, record: logging.LogRecord):
        """Handle episode start event."""
        csv_path = getattr(record, "csv_path", "unknown")
        self.console.print(f"\n[cyan]Starting episode...[/cyan]")
        self.console.print(f"[dim]Dataset: {csv_path}[/dim]\n")

    def _handle_turn_start(self, record: logging.LogRecord):
        """Handle turn start event."""
        turn = getattr(record, "turn", "?")
        max_turns = getattr(record, "max_turns", "?")
        self.console.rule(f"[bold]Turn {turn}/{max_turns}[/bold]", style="blue")

    def _handle_model_thinking_start(self, record: logging.LogRecord):
        """Handle model thinking start event."""
        # Note: Rich status/spinner is tricky with logging
        # For now, just print a simple message
        self.console.print("[magenta]Thinking...[/magenta]")

    def _handle_model_thinking_end(self, record: logging.LogRecord):
        """Handle model thinking end event."""
        # Spinner would end here if we implemented it
        pass

    def _handle_model_response(self, record: logging.LogRecord):
        """Handle model response event."""
        response = getattr(record, "response", "")
        self.console.print(Panel(
            response,
            title="[magenta]Assistant[/magenta]",
            border_style="magenta"
        ))

    def _handle_tool_executed(self, record: logging.LogRecord):
        """Handle tool execution event."""
        tool_name = getattr(record, "tool", "unknown")
        output = getattr(record, "output", "")
        success = getattr(record, "success", False)

        # Print tool name with success/failure indicator
        color = "green" if success else "red"
        symbol = "✓" if success else "✗"
        self.console.print(f"[{color}]{symbol}[/{color}] [yellow]{tool_name}[/yellow]")

        # Print output
        self.console.print(f"[dim]{output}[/dim]")

    def _handle_no_tool_call(self, record: logging.LogRecord):
        """Handle no tool call event."""
        self.console.print("[yellow]⚠ No tool call[/yellow]")

    def _handle_episode_complete(self, record: logging.LogRecord):
        """Handle episode complete event."""
        results = getattr(record, "results", None)
        self.console.print("\n[bold green]✓ DONE[/bold green]")

        if results:
            self.summarize_results(results)

    def _handle_max_turns_reached(self, record: logging.LogRecord):
        """Handle max turns reached event."""
        max_turns = getattr(record, "max_turns", "?")
        self.console.print(f"[yellow]Max turns ({max_turns})[/yellow]")

    def summarize_results(self, data: list[dict] | None):
        """
        Pretty-print parsed results based on pipeline mode.

        This is the same logic from the old synthetic_pipeline.py
        summarize_results() function.

        Args:
            data: List of parsed results (episodes, question plans, or tool recommendations)
        """
        if not data:
            self.console.print(self.rollout_config.parse_error_msg)
            return

        self.console.print(f"\n[green]✓ {len(data)} {self.rollout_config.success_label}[/green]")

        pipeline_mode = getattr(self.rollout_config, "mode", "").lower()

        if pipeline_mode == "tool-feedback":
            for i, rec in enumerate(data, 1):
                name = rec.get("name", "?")
                priority = rec.get("priority", "?")
                why = rec.get("why", "?")[:60]
                self.console.print(f"  [dim]{i}.[/dim] [{priority}] [bold]{name}[/bold]: {why}")
        elif pipeline_mode == "explore":
            for i, plan in enumerate(data, 1):
                diff = plan.get("difficulty", "?")
                steps = plan.get("expected_steps", "?")
                q = plan.get("question_text", "?")
                self.console.print(f"  [dim]{i}.[/dim] [{diff}|steps={steps}] {q}")
        else:  # episodes mode
            for i, ep in enumerate(data, 1):
                if isinstance(ep, dict):
                    diff = ep.get("difficulty", "?")
                    q = ep.get("question_text", "?")
                    n_hooks = len(ep.get("hooks", []))
                    self.console.print(f"  [dim]{i}.[/dim] [{diff}] ({n_hooks}h) {q}")


class LogContext:
    """
    Context manager for logging start/end events.

    Usage:
        with LogContext(logger, "model_thinking"):
            response = model(...)

    This logs "model_thinking_start" on enter and "model_thinking_end" on exit.
    """

    def __init__(self, logger: logging.Logger | None, event: str, extra: dict | None = None):
        """
        Initialize LogContext.

        Args:
            logger: Logger instance or None
            event: Base event name (e.g., "model_thinking")
            extra: Optional extra data to pass to logger
        """
        self.logger = logger
        self.event = event
        self.extra = extra or {}

    def __enter__(self):
        """Log start event on context entry."""
        if self.logger:
            self.logger.info(self.event + "_start", extra=self.extra)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log end event on context exit."""
        if self.logger:
            self.logger.info(self.event + "_end", extra=self.extra)
        return False  # Don't suppress exceptions


def setup_rich_logger(rollout_config) -> logging.Logger:
    """
    Create a logger configured with RichHandler.

    This is a convenience function for setting up Rich logging
    in the CLI/pipeline.

    Args:
        rollout_config: RolloutConfig with rollout-specific configuration

    Returns:
        Configured logger with RichHandler attached

    Usage:
        logger = setup_rich_logger(rollout_config)
        env = Environment(..., logger=logger)
        env.rollout()  # Outputs to Rich console
    """
    logger = logging.getLogger("csv_agent")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add RichHandler
    logger.addHandler(RichHandler(rollout_config))

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
