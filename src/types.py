"""
Type definitions for CSV agent.

This module contains Pydantic models and type definitions used throughout
the CSV agent codebase.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime


# ============= Hook and Episode Types (for RL/verifier) =============

class HookParams(BaseModel):
    """Parameters for a hook (tool call)."""
    group_col: str
    target_col: str
    agg: str


class Hook(BaseModel):
    """A single hook (verifiable intermediate step) in an episode."""
    id: str
    tool: str
    params: HookParams
    depends_on: List[str]


class Episode(BaseModel):
    """A complete episode with question, hooks, and answers."""
    question_text: str
    difficulty: str
    hooks: List[Hook]
    teacher_answers: Dict[str, str]
    solution_trace: str


# ============= Question Generation Types =============

class QuestionGeneration(BaseModel):
    """A question blueprint/plan."""
    question_text: str
    reasoning_path: str
    difficulty: str
    key_columns: List[str]
    expected_steps: int


class Answer(BaseModel):
    """An answer to a question with hooks and verification."""
    question: QuestionGeneration
    hooks: List[Hook]
    teacher_answers: Dict[str, str]
    solution_trace: str
    computed_answers: Dict[str, str]
    execution_valid: bool | None = None


# ============= Turn and Conversation Types =============

class ToolCall(BaseModel):
    """Structured representation of a tool call."""
    tool_name: str
    params: dict
    raw_code: str
    result: str
    success: bool
    timestamp: datetime = datetime.now()


class Turn(BaseModel):
    """Rich representation of a single conversation turn."""
    turn_number: int
    timestamp: datetime = datetime.now()

    # Model interaction
    model_response: str
    truncated_response: str

    # Structured tool data
    tool_calls: List[ToolCall] = []

    # Metadata (extensible)
    reasoning: Optional[str] = None
    done_signal: bool = False
    feedback_message: str = ""
    estimated_tokens: Optional[int] = None

    def to_messages(self) -> list[dict]:
        """Convert turn to OpenAI message format."""
        messages = [
            {"role": "assistant", "content": self.truncated_response}
        ]
        if not self.done_signal and self.feedback_message:
            messages.append({"role": "user", "content": self.feedback_message})
        return messages

    def estimate_tokens(self) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        total_chars = len(self.truncated_response) + len(self.feedback_message)
        return total_chars // 4


class ConversationManager(BaseModel):
    """Manages conversation history with smart context management."""

    # System prompt (always kept)
    system_prompt: str

    # Active turns (kept in context)
    active_turns: List[Turn] = []

    # Archived turns (purged from context but saved)
    archived_turns: List[Turn] = []

    # Configuration
    max_active_turns: int = 5
    max_context_tokens: int = 80_000

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

    def get_turn_summary(self, turn_number: int) -> Optional[Turn]:
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


# ============= Environment Configuration Types =============

class EnvironmentConfig(BaseModel):
    """Configuration for the Environment."""
    csv_path: str = "data.csv"
    model: str = "grok-4.1-fast"
    pipeline_mode: str = "explore"  # "explore", "episodes", or "tool-feedback"
    max_turns: int = 10
    target_questions: int = 10

    # Context management configuration
    max_active_turns: int = 5
    max_context_tokens: int = 80_000


class StateConfig(BaseModel):
    """State configuration for an episode."""
    input: str
    conversation_manager: "ConversationManager"
    n_turns: int
    is_completed: bool
    current_turn: int