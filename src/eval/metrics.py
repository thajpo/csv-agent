"""
Metrics and result types for evaluation.

Defines the data structures for evaluation results and aggregate metrics.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result of evaluating a single episode."""

    episode_id: str
    question_text: str
    difficulty: str | None

    # Evaluation outcomes
    final_answer_correct: bool
    execution_success: bool

    # Metadata
    num_turns: int
    elapsed_seconds: float

    # Raw values for debugging
    expected_answer: Any = None
    actual_answer: Any = None

    # Error info if execution failed
    error_message: str | None = None


@dataclass
class EvalMetrics:
    """Aggregate metrics across multiple episodes."""

    # Overall metrics
    accuracy: float  # % episodes with correct final answer
    execution_success_rate: float  # % episodes that submitted an answer

    # Efficiency metrics
    avg_turns: float
    avg_elapsed_seconds: float

    # Breakdown by difficulty
    accuracy_by_difficulty: dict[str, float] = field(default_factory=dict)

    # Raw counts
    total_episodes: int = 0
    total_correct: int = 0
    total_executed: int = 0

    # Per-difficulty counts (for debugging)
    episodes_by_difficulty: dict[str, int] = field(default_factory=dict)
    correct_by_difficulty: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": self.accuracy,
            "execution_success_rate": self.execution_success_rate,
            "avg_turns": self.avg_turns,
            "avg_elapsed_seconds": self.avg_elapsed_seconds,
            "accuracy_by_difficulty": self.accuracy_by_difficulty,
            "total_episodes": self.total_episodes,
            "total_correct": self.total_correct,
            "total_executed": self.total_executed,
            "episodes_by_difficulty": self.episodes_by_difficulty,
            "correct_by_difficulty": self.correct_by_difficulty,
        }
