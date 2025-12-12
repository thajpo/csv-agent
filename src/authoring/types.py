"""
Type definitions for the authoring pipeline (offline dataset generation).

This module contains types used for teacher trace generation, question
exploration, and triangulation verification.
"""

from pydantic import BaseModel
from typing import Any
from datetime import datetime

# Import shared types from core
from src.core.types import Artifact


# ============= Teacher Trace Types =============

class TeacherTrace(BaseModel):
    """Teacher's solution execution record."""
    question: str
    hint: str | None = None
    code_cells: list[str]                   # Raw Python code per turn
    artifacts: dict[str, Artifact]          # name → Artifact
    final_answer: Any = None                # The submit() value
    final_answer_hash: str | None = None
    execution_success: bool

    class Config:
        arbitrary_types_allowed = True


class Episode(BaseModel):
    """Complete training episode with scavenger hunt rewards."""
    id: str                                 # Unique episode ID
    timestamp: datetime = datetime.now()
    question: str
    hint: str | None = None

    # Teacher data
    teacher_trace: TeacherTrace
    consistency_traces: list[TeacherTrace] = []  # For triangulation verification

    # Student data (populated during training)
    student_trace: Any | None = None  # StudentTrace from training.types

    # Rewards (scavenger hunt)
    intermediate_matches: list[str] = []    # Artifact names student matched
    final_match: bool = False               # Did final answer match?
    dense_reward: int = 0                   # +1 per intermediate match
    sparse_reward: int = 0                  # +5 if final match
    total_reward: float = 0.0               # Combined score

    # Metadata
    difficulty: str | None = None           # Optional difficulty tagging
    verified: bool = False                  # Passed triangulation?

    class Config:
        arbitrary_types_allowed = True


# ============= Question Generation Types =============

class ExplorationTurn(BaseModel):
    """Single turn in dataset exploration conversation."""
    turn_number: int
    reasoning: str                              # Model's written notes (what/why/hypotheses)
    code_cells: list[str]                       # Python code executed
    execution_results: list[Any]                # CodeCellResult objects (avoiding circular import)
    timestamp: datetime

    class Config:
        arbitrary_types_allowed = True


class ExplorationTrace(BaseModel):
    """Complete exploration session for question generation."""
    csv_path: str
    turns: list[ExplorationTurn]
    questions_generated: list[dict]             # Final questions
    total_turns: int
    timestamp: datetime = datetime.now()

    class Config:
        arbitrary_types_allowed = True


class GeneratedQuestion(BaseModel):
    """LLM-generated question with metadata."""
    question: str
    hint: str                                   # Max 2-3 sentences
    n_steps: int                                # Estimated step count
    difficulty: str                             # EASY, MEDIUM, HARD, VERY_HARD

    # Optional metadata for tracking
    exploration_turn: int | None = None         # Which turn inspired this question
