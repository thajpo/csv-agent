"""
Core type definitions shared across authoring and training pipelines.

This module contains artifact hashing utilities and base types used by
both offline dataset generation and online training.
"""

from pydantic import BaseModel
from typing import Any
from hashlib import sha256
import pickle
import pandas as pd

# ============= Artifact Hashing =============

def hash_artifact(obj: Any) -> str:
    """
    Hash a DataFrame or scalar value for state matching.

    Args:
        obj: A pandas DataFrame or scalar (int, float, str, bool, None)

    Returns:
        16-character hex hash of the object's state
    """
    if isinstance(obj, pd.DataFrame):
        # Hash structure (columns, dtypes) and values
        try:
            content = pickle.dumps({
                'columns': obj.columns.tolist(),
                'dtypes': obj.dtypes.tolist(),
                'values': obj.values.tobytes(),
                'shape': obj.shape
            })
        except Exception:
            # Fallback for non-byteable dtypes
            content = pickle.dumps({
                'columns': obj.columns.tolist(),
                'dtypes': [str(dt) for dt in obj.dtypes],
                'values': obj.to_dict(),
                'shape': obj.shape
            })
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        # Scalar: just pickle it
        content = pickle.dumps(obj)
    else:
        # Try to pickle anything else
        content = pickle.dumps(obj)

    return sha256(content).hexdigest()[:16]  # 16 chars is enough


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
