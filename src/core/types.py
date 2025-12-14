"""
Core type definitions shared across authoring and training pipelines.

This module contains base types used by both offline dataset generation
and online training.

Note: hash_artifact is defined in src.utils.hashing to avoid duplication.
"""

from pydantic import BaseModel
from typing import Any


# ============= Core Artifact Type =============

class Artifact(BaseModel):
    """A checkpoint variable from code execution."""
    name: str                # Variable name (e.g., 'df_filtered')
    hash: str               # Hash of its state
    type: str               # 'DataFrame' | 'scalar' | 'other'

    class Config:
        arbitrary_types_allowed = True


# ============= New Unified Types (Phase 1) =============

class Question(BaseModel):
    """A question with metadata."""
    question_text: str
    hint: str | None = None
    difficulty: str | None = None  # EASY, MEDIUM, HARD, VERY_HARD
    n_steps: int | None = None     # Expected step count

    # For tracking/versioning
    id: str | None = None
    created_at: Any | None = None  # datetime, but avoid import for now


class ExecutionTrace(BaseModel):
    """Record of a code execution session (teacher or student)."""
    code_cells: list[str]                   # Raw Python code per turn
    artifacts: dict[str, Artifact]          # name → Artifact
    final_answer: Any | None = None         # The submit() value
    final_answer_hash: str | None = None
    execution_success: bool

    # Optional metadata (for debugging/analysis)
    total_turns: int = 0
    archived_turn_count: int = 0            # Turns purged from context

    class Config:
        arbitrary_types_allowed = True
