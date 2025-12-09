"""
Type definitions for CSV agent.

This module contains Pydantic models and type definitions used throughout
the CSV agent codebase.
"""

from pydantic import BaseModel
from typing import TYPE_CHECKING, Any
from datetime import datetime
from hashlib import sha256
import pickle
import pandas as pd

if TYPE_CHECKING:
    from src.conversation import ConversationManager


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


# ============= Artifact and Trace Types (for scavenger hunt) =============

class Artifact(BaseModel):
    """A checkpoint variable from code execution."""
    name: str                # Variable name (e.g., 'df_filtered')
    hash: str               # Hash of its state
    type: str               # 'DataFrame' | 'scalar' | 'other'

    class Config:
        arbitrary_types_allowed = True


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
    student_trace: StudentTrace | None = None

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


# ============= Legacy Types (kept for compatibility, may be removed later) =============

class HookParams(BaseModel):
    """Parameters for a hook (tool call). DEPRECATED - use Artifact-based system."""
    group_col: str
    target_col: str
    agg: str


class Hook(BaseModel):
    """A single hook (verifiable intermediate step). DEPRECATED - use Artifact-based system."""
    id: str
    tool: str
    params: HookParams
    depends_on: list[str]


class QuestionGeneration(BaseModel):
    """A question blueprint/plan."""
    question_text: str
    reasoning_path: str
    difficulty: str
    key_columns: list[str]
    expected_steps: int


class Answer(BaseModel):
    """An answer to a question with hooks and verification. DEPRECATED - use Episode."""
    question: QuestionGeneration
    hooks: list[Hook]
    teacher_answers: dict[str, str]
    solution_trace: str
    computed_answers: dict[str, str]
    execution_valid: bool | None = None


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
    conversation_manager: "ConversationManager"
    n_turns: int
    is_completed: bool
    current_turn: int
