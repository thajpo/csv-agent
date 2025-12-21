"""
Type definitions for CSV Agent.

All shared types in one place:
- Core types (Artifact, Question, ExecutionTrace)
- Episode types (Episode, EpisodeJSONL)
- Exploration types (ExplorationTurn, ExplorationTrace)
"""

from pydantic import BaseModel, ConfigDict
from typing import Any, NamedTuple
from datetime import datetime


# ============= Result Types =============

class ExecutionResult(NamedTuple):
    """Result of parsing code execution output."""
    success: bool
    stdout: str
    stderr: str


# ============= Core Types =============

class Artifact(BaseModel):
    """A checkpoint variable from code execution."""
    name: str                # Variable name (e.g., 'df_filtered')
    hash: str               # Hash of its state
    type: str               # 'DataFrame' | 'scalar' | 'other'

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Question(BaseModel):
    """A question with metadata."""
    question_text: str
    hint: str | None = None
    difficulty: str | None = None  # EASY, MEDIUM, HARD, VERY_HARD
    n_steps: int | None = None     # Expected step count

    # For tracking/versioning
    id: str | None = None
    created_at: Any | None = None  # datetime


class ExecutionTrace(BaseModel):
    """Record of a code execution session (teacher or student)."""
    code_cells: list[str]                   # Raw Python code per turn
    final_answer: Any | None = None         # The submit() value
    final_answer_hash: str | None = None
    execution_success: bool

    # Metadata from submit(..., extra=...)
    submission_metadata: dict[str, Any] = {}

    # Optional metadata (for debugging/analysis)
    total_turns: int = 0
    archived_turn_count: int = 0            # Turns purged from context

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============= Episode Types =============

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

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EpisodeJSONL(BaseModel):
    """Episode formatted for JSONL training data (nested SFT + RL)."""
    episode_id: str
    timestamp: datetime
    verified: bool

    # Question
    question: dict

    # Traces
    teacher_gold_trace: dict
    consistency_traces: list[dict]

    # Training data
    conversation_for_sft: dict  # {"system_prompt": str, "messages": list}
    rl_verification_data: dict

    # Metadata
    triangulation_metadata: dict

    @classmethod
    def from_episode(
        cls,
        episode: Episode,
        gold_conversation: list[dict],
        system_prompt: str,
        consistency_conversations: list[list[dict]],
    ) -> "EpisodeJSONL":
        """Convert Episode to JSONL format."""
        from collections import Counter

        # Build triangulation metadata
        consistency_traces = episode.consistency_traces
        consistency_hashes = [
            t.final_answer_hash for t in consistency_traces
            if t.final_answer_hash is not None
        ]

        if consistency_hashes:
            majority_hash, majority_count = Counter(consistency_hashes).most_common(1)[0]
        else:
            majority_hash, majority_count = None, 0

        # Extract messages (skip system prompt - it's separated)
        messages_without_system = gold_conversation[1:] if gold_conversation else []

        return cls(
            episode_id=episode.id,
            timestamp=episode.timestamp,
            verified=episode.verified,
            question=episode.question.model_dump(),
            teacher_gold_trace=episode.teacher_trace.model_dump(),
            consistency_traces=[t.model_dump() for t in consistency_traces],
            conversation_for_sft={
                "system_prompt": system_prompt,
                "messages": messages_without_system,
            },
            rl_verification_data={
                "expected_final_answer_hash": episode.teacher_trace.final_answer_hash,
                "expected_final_answer": episode.teacher_trace.final_answer,
                "intermediate_artifact_hashes": {},  # Legacy field, not currently populated
            },
            triangulation_metadata={
                "n_consistency_runs": len(consistency_traces),
                "n_consistency_succeeded": len(consistency_hashes),
                "majority_answer_hash": majority_hash,
                "majority_count": majority_count,
                "gold_matches_majority": episode.verified,
            },
        )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============= Exploration Types =============

class ExplorationTurn(BaseModel):
    """Single turn during dataset exploration for question generation."""
    turn_number: int
    reasoning: str
    code_cells: list[str]
    execution_results: list[Any]  # CodeCellResult objects
    timestamp: datetime

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ExplorationTrace(BaseModel):
    """Record of exploration session for question generation."""
    csv_path: str
    turns: list[ExplorationTurn]
    questions_generated: list[dict]
    total_turns: int
    timestamp: datetime = datetime.now()

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GeneratedQuestion(BaseModel):
    """A generated question from exploration."""
    question: str
    hint: str
    n_steps: int
    difficulty: str  # EASY, MEDIUM, HARD, VERY_HARD

    model_config = ConfigDict(arbitrary_types_allowed=True)
