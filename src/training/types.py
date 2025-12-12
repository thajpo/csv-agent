"""
Type definitions for the training pipeline (online RL).

This module contains types used for student trace collection and
RL-specific data structures.
"""

from pydantic import BaseModel
from typing import Any

# Import shared types from core
from src.core.types import Artifact


# ============= Student Trace Types =============

class StudentTrace(BaseModel):
    """Student's solution attempt."""
    question: str
    code_cells: list[str]
    artifacts: dict[str, Artifact]
    final_answer: Any | None = None
    final_answer_hash: str | None = None
    execution_success: bool

    class Config:
        arbitrary_types_allowed = True
