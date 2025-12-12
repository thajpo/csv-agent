"""
Core type definitions shared across authoring and training pipelines.

This module contains artifact hashing utilities and base types used by
both offline dataset generation and online training.
"""

from pydantic import BaseModel
from typing import Any
from datetime import datetime
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


# ============= Environment Configuration Types =============

class EnvironmentConfig(BaseModel):
    """Configuration for the Environment."""
    csv_path: str = "data.csv"
    model: str = "grok-4.1-fast"
    pipeline_mode: str = "teacher-tutor"  # "teacher-tutor", "teacher-consistency", "student"
    max_turns: int = 10
    target_questions: int = 10

    # Context management configuration
    max_active_turns: int = 5
    max_context_tokens: int = 80_000


class StateConfig(BaseModel):
    """State configuration for an episode."""
    input: str
    conversation_manager: Any  # ConversationManager - avoiding circular import
    n_turns: int
    is_completed: bool
    current_turn: int

    class Config:
        arbitrary_types_allowed = True
