"""
Type definitions for the authoring pipeline (offline dataset generation).

This module contains types used for teacher trace generation, question
exploration, and triangulation verification.
"""

from pydantic import BaseModel
from typing import Any
from datetime import datetime

# Import shared types from core
from src.core.types import Artifact, ExecutionTrace, Question


# ============= Episode (Simplified) =============

class Episode(BaseModel):
    """A verified training episode."""
    id: str
    question: Question                              # First-class Question object
    teacher_trace: ExecutionTrace                   # Teacher execution WITH hint
    consistency_traces: list[ExecutionTrace] = []   # Consistency traces WITHOUT hint (for triangulation)
    verified: bool = False                          # Passed triangulation?

    # Student execution (populated during training)
    student_trace: ExecutionTrace | None = None

    # Reward calculation (computed, ephemeral)
    reward_summary: dict | None = None              # {"intermediate_matches": [...], "final_match": bool, ...}

    # Metadata
    timestamp: datetime = datetime.now()

    class Config:
        arbitrary_types_allowed = True


# ============= Question Exploration Types =============

class ExplorationTurn(BaseModel):
    """Single turn during dataset exploration for question generation."""
    turn_number: int
    reasoning: str
    code_cells: list[str]
    execution_results: list[Any]  # CodeCellResult objects
    timestamp: datetime

    class Config:
        arbitrary_types_allowed = True


class ExplorationTrace(BaseModel):
    """Record of exploration session for question generation."""
    csv_path: str
    turns: list[ExplorationTurn]
    questions_generated: list[dict]
    total_turns: int
    timestamp: datetime = datetime.now()

    class Config:
        arbitrary_types_allowed = True


class GeneratedQuestion(BaseModel):
    """A generated question from exploration."""
    question: str
    hint: str
    n_steps: int
    difficulty: str  # EASY, MEDIUM, HARD, VERY_HARD

    class Config:
        arbitrary_types_allowed = True
