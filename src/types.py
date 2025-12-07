"""
Type definitions for CSV agent.

This module contains Pydantic models and type definitions used throughout
the CSV agent codebase.
"""

from pydantic import BaseModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.conversation import ConversationManager


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
    depends_on: list[str]


class Episode(BaseModel):
    """A complete episode with question, hooks, and answers."""
    question_text: str
    difficulty: str
    hooks: list[Hook]
    teacher_answers: dict[str, str]
    solution_trace: str


# ============= Question Generation Types =============

class QuestionGeneration(BaseModel):
    """A question blueprint/plan."""
    question_text: str
    reasoning_path: str
    difficulty: str
    key_columns: list[str]
    expected_steps: int


class Answer(BaseModel):
    """An answer to a question with hooks and verification."""
    question: QuestionGeneration
    hooks: list[Hook]
    teacher_answers: dict[str, str]
    solution_trace: str
    computed_answers: dict[str, str]
    execution_valid: bool | None = None


# ============= Turn and Conversation Types =============
# (Moved to src/conversation.py for better separation of concerns)


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