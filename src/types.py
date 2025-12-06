"""
Type definitions for CSV agent.

This module contains dataclasses and type definitions used throughout
the CSV agent codebase.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvironmentConfig:
    """Configuration for the Environment."""
    csv_path: str = "data.csv"
    model: str = "grok-4.1-fast"
    mode: str = "explore"  # "explore", "episodes", or "tool-feedback"
    max_turns: int = 10
    target_questions: int = 10


@dataclass
class StateConfig:
    """State configuration for an episode."""
    input: str
    observation: str = ""
    conversation: list[dict[str, str]] = field(default_factory=list)
    completed: bool = False
    results: Any = None
