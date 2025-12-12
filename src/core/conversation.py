"""
Conversation management for CSV agent.

This module contains classes for managing conversation state and turns
with automatic context management (purging) to stay within token limits.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Any


class CodeCellResult(BaseModel):
    """Result of executing one code cell."""
    code: str
    success: bool
    stdout: str
    stderr: str
    submitted_answer: Any | None = None

    class Config:
        arbitrary_types_allowed = True


class Turn(BaseModel):
    """Rich representation of a single conversation turn."""
    turn_number: int
    timestamp: datetime = datetime.now()

    # Model interaction
    model_response: str

    # Code execution (for Jupyter-style execution)
    code_cells: list[str] = []
    execution_results: list[CodeCellResult] = []

    # Metadata (extensible)
    reasoning: str | None = None
    done_signal: bool = False
    feedback_message: str = ""
    estimated_tokens: int | None = None

    class Config:
        arbitrary_types_allowed = True

    def to_messages(self) -> list[dict]:
        """Convert turn to OpenAI message format."""
        messages = [
            {"role": "assistant", "content": self.model_response}
        ]
        if not self.done_signal and self.feedback_message:
            messages.append({"role": "user", "content": self.feedback_message})
        return messages

    def estimate_tokens(self) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        total_chars = len(self.model_response) + len(self.feedback_message)
        return total_chars // 4


class ConversationManager(BaseModel):
    """Manages conversation history with smart context management."""

    # System prompt (always kept)
    system_prompt: str

    # Active turns (kept in context)
    active_turns: list[Turn] = []

    # Archived turns (purged from context but saved)
    archived_turns: list[Turn] = []

    # Configuration (should be provided from config.yaml via EnvironmentConfig)
    max_active_turns: int
    max_context_tokens: int

    # State
    total_turns: int = 0

    def add_turn(self, turn: Turn) -> None:
        """Add a new turn and potentially trigger purging."""
        turn.turn_number = self.total_turns
        turn.estimated_tokens = turn.estimate_tokens()

        self.active_turns.append(turn)
        self.total_turns += 1

        # Check if we need to purge
        if self._should_purge():
            self._purge_oldest_turns()

    def _should_purge(self) -> bool:
        """Decide if we need to purge based on token count or turn count."""
        # Token-based purging
        total_tokens = self._estimate_total_tokens()
        if total_tokens > self.max_context_tokens:
            return True

        # Turn-based purging (keep last N turns)
        if len(self.active_turns) > self.max_active_turns:
            return True

        return False

    def _estimate_total_tokens(self) -> int:
        """Estimate total tokens in active context."""
        system_tokens = len(self.system_prompt) // 4
        turn_tokens = sum(t.estimated_tokens or 0 for t in self.active_turns)
        return system_tokens + turn_tokens

    def _purge_oldest_turns(self) -> None:
        """Move oldest turns to archive, keeping last N."""
        while len(self.active_turns) > self.max_active_turns:
            oldest = self.active_turns.pop(0)
            self.archived_turns.append(oldest)

    def to_openai_messages(self) -> list[dict]:
        """Convert to OpenAI message format for model API."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add all active turns
        for turn in self.active_turns:
            messages.extend(turn.to_messages())

        return messages

    def get_turn_summary(self, turn_number: int) -> Turn | None:
        """Get a specific turn (from active or archived)."""
        # Search active turns
        for turn in self.active_turns:
            if turn.turn_number == turn_number:
                return turn

        # Search archived turns
        for turn in self.archived_turns:
            if turn.turn_number == turn_number:
                return turn

        return None
