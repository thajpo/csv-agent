"""
Type definitions for CSV Agent.

All shared types in one place:
- Core types (Question, ExecutionTrace, Hook)
- Episode types (Episode, EpisodeJSONL)
- Exploration types (ExplorationTurn, ExplorationTrace)
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, NamedTuple
from datetime import datetime


# ============= Result Types =============

class ExecutionResult(NamedTuple):
    """Result of parsing code execution output."""
    success: bool
    stdout: str
    stderr: str


# ============= Core Types =============

class Question(BaseModel):
    """A question with metadata."""
    question_text: str
    hint: str | None = None
    difficulty: str | None = None  # EASY, MEDIUM, HARD, VERY_HARD
    n_steps: int | None = None     # Expected step count

    # For tracking/versioning
    id: str | None = None
    created_at: Any | None = None  # datetime

    def generate_id(self) -> str:
        """Generate deterministic ID from question_text + hint."""
        import hashlib
        content = f"{self.question_text}|{self.hint or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        """Create Question from dict, auto-generating ID if missing."""
        # Map 'question' key to 'question_text' if needed
        data = dict(d)
        if "question" in data and "question_text" not in data:
            data["question_text"] = data.pop("question")
        
        q = cls(**data)
        if q.id is None:
            q.id = q.generate_id()
        return q


class Hook(BaseModel):
    """A verifiable checkpoint in the solution trace.
    
    Hooks capture intermediate states during code execution for RL reward.
    The value_hash allows verification without storing the actual value.
    The depends_on field tracks which previous hooks must be computed first.
    """
    code_line: str                      # The code that produced this
    variable_name: str | None = None    # e.g., 'df_filtered'
    value_hash: str                     # Hash of the value at this point
    description: str | None = None      # Optional semantic description
    depends_on: list[str] = []          # Names of hooks this depends on (DAG edges)


class ExecutionTrace(BaseModel):
    """Record of a code execution session (teacher or student)."""
    code_cells: list[str]                   # Raw Python code per turn
    final_answer: Any | None = None         # The submit() value
    final_answer_hash: str | None = None
    execution_success: bool

    # Intermediate checkpoints for RL dense reward
    hooks: list[Hook] = []

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
    timestamp: datetime = Field(default_factory=datetime.now)

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
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(arbitrary_types_allowed=True)
