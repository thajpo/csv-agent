"""
Type definitions for CSV Agent.

All shared types in one place:
- Core types (Question, ExecutionTrace, Hook)
- Episode types (Episode, EpisodeJSONL)
- Exploration types (ExplorationTurn, ExplorationTrace)
- TypedDicts for JSONL serialization
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, NamedTuple, TypedDict, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from src.core.conversation import CodeCellResult


# ============= TypedDicts for JSONL Serialization =============

class QuestionDict(TypedDict, total=False):
    """Serialized Question structure."""
    id: str | None
    question_text: str
    hint: str | None
    difficulty: str | None
    n_steps: int | None
    created_at: str | None
    template_name: str | None
    template_params: dict[str, Any] | None
    output_type: str | None
    output_schema: str | None
    ground_truth_hash: str | None


class HookDict(TypedDict, total=False):
    """Serialized Hook structure."""
    code_line: str
    variable_name: str | None
    value_hash: str
    description: str | None
    depends_on: list[str]


class ExecutionTraceDict(TypedDict, total=False):
    """Serialized ExecutionTrace structure."""
    code_cells: list[str]
    final_answer: Any
    final_answer_hash: str | None
    execution_success: bool
    hooks: list[HookDict]
    submission_metadata: dict[str, Any]
    total_turns: int
    archived_turn_count: int



class ConversationForSFTDict(TypedDict):
    """Conversation data for SFT training."""
    system_prompt: str
    messages: list[dict]  # MessageDict from conversation.py


class RLVerificationDataDict(TypedDict):
    """RL verification data."""
    expected_final_answer_hash: str | None
    expected_final_answer: Any


class TriangulationMetadataDict(TypedDict):
    """Triangulation results."""
    n_consistency_runs: int
    n_consistency_succeeded: int
    majority_answer_hash: str | None
    majority_count: int
    gold_matches_majority: bool


class TimingMetadataDict(TypedDict):
    """Execution timing for episode generation."""
    gold_elapsed: float
    consistency_elapsed: list[float]
    total_elapsed: float
    avg_elapsed: float


# ============= Core Types =============

class Question(BaseModel):
    """A question with metadata."""
    question_text: str
    hint: str | None = None
    difficulty: str | None = None  # EASY, MEDIUM, HARD, VERY_HARD
    n_steps: int | None = None     # Expected step count
    template_name: str | None = None
    template_params: dict[str, Any] | None = None
    output_type: str | None = None
    output_schema: str | None = None
    ground_truth_hash: str | None = None

    # Synthetic question evaluation fields
    ground_truth: Any | None = None  # Actual ground truth value (for synthetic)
    template: str | None = None  # Template name that generated this question

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
        data = dict(d)

        # Map 'question' key to 'question_text' if needed
        if "question" in data and "question_text" not in data:
            data["question_text"] = data.pop("question")

        # Map internal fields to public fields (synthetic questions)
        if "_ground_truth" in data:
            data["ground_truth"] = data.pop("_ground_truth")
        if "_template" in data:
            data["template"] = data.pop("_template")

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
    depends_on: list[str] = Field(default_factory=list)  # Names of hooks this depends on (DAG edges)


class ExecutionTrace(BaseModel):
    """Record of a code execution session (teacher or student)."""
    code_cells: list[str]                   # Raw Python code per turn
    final_answer: Any | None = None         # The submit() value
    final_answer_hash: str | None = None
    execution_success: bool

    # Intermediate checkpoints for RL dense reward
    hooks: list[Hook] = Field(default_factory=list)

    # Metadata from submit(..., extra=...)
    submission_metadata: dict[str, Any] = Field(default_factory=dict)

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
    consistency_traces: list[ExecutionTrace] = Field(default_factory=list)  # Consistency traces WITHOUT hint
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

    # Data source
    csv_source: str  # Path to CSV used for this episode

    # Question
    question: QuestionDict

    # Traces
    teacher_gold_trace: ExecutionTraceDict
    consistency_traces: list[ExecutionTraceDict]

    # Training data
    conversation_for_sft: ConversationForSFTDict
    rl_verification_data: RLVerificationDataDict

    # Metadata
    triangulation_metadata: TriangulationMetadataDict
    timing_metadata: TimingMetadataDict

    @classmethod
    def from_episode(
        cls,
        episode: Episode,
        gold_conversation: list[dict],
        system_prompt: str,
        consistency_conversations: list[list[dict]],
        csv_source: str,
        timing_metadata: TimingMetadataDict,
        majority_answer_hash: str | None = None,
        majority_count: int | None = None,
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
            fallback_majority_hash, fallback_majority_count = (
                Counter(consistency_hashes).most_common(1)[0]
            )
        else:
            fallback_majority_hash, fallback_majority_count = None, 0

        majority_hash = (
            majority_answer_hash
            if majority_answer_hash is not None
            else fallback_majority_hash
        )
        majority_votes = (
            majority_count
            if majority_count is not None
            else fallback_majority_count
        )

        # Extract messages (skip system prompt - it's separated)
        messages_without_system = gold_conversation[1:] if gold_conversation else []

        return cls(
            episode_id=episode.id,
            timestamp=episode.timestamp,
            verified=episode.verified,
            csv_source=csv_source,
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
                "majority_count": majority_votes,
                "gold_matches_majority": episode.verified,
            },
            timing_metadata=timing_metadata,
        )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============= Exploration Types =============

class ExplorationTurn(BaseModel):
    """Single turn during dataset exploration for question generation."""
    turn_number: int
    reasoning: str
    code_cells: list[str]
    execution_results: list["CodeCellResult"]  # Forward ref to avoid circular import
    timestamp: datetime

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ExplorationTrace(BaseModel):
    """Record of exploration session for question generation."""
    csv_path: str
    turns: list[ExplorationTurn]
    questions_generated: list[QuestionDict]  # Use typed dict
    total_turns: int
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(arbitrary_types_allowed=True)
