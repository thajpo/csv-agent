"""
Type definitions for CSV Agent.

All shared types in one place:
- Core types (Question, Hook)
- Turn-based types (ExecutionResult, Turn, Trace)
- Episode types (EpisodeJSONL)
- Exploration types (ExplorationTurn, ExplorationTrace)
- TypedDicts for JSONL serialization
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, NamedTuple, TypedDict, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from src.core.conversation import CodeCellResult


# ============= Core TypedDicts =============


class QADict(TypedDict, total=False):
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
    ground_truth: Any | None


class HookDict(TypedDict, total=False):
    """Captured intermediate state during execution."""

    variable_name: str | None
    code_line: str  # The code that produced this value
    value: Any  # Raw normalized value (for PRM training)
    value_hash: str  # Hash for comparison
    depends_on: list[str]  # DAG edges to prior hooks
    description: str | None


# ============= Turn-Based TypedDicts (NEW) =============


class ExecutionResultDict(TypedDict):
    """Result of executing one code cell."""

    success: bool
    stdout: str
    stderr: str
    hooks: list[HookDict]  # Hooks captured in THIS cell
    submitted_answer: Any | None  # If submit() was called in this cell


class TurnDict(TypedDict):
    """Single turn = model output + execution result."""

    turn_index: int
    reasoning: str  # Model's thinking/explanation
    code: str  # Code block (single cell for now)
    execution: ExecutionResultDict  # What happened when code ran


class TraceDict(TypedDict):
    """Complete trace = sequence of turns + final outcome."""

    turns: list[TurnDict]
    final_answer: Any | None
    final_answer_hash: str | None
    success: bool  # Did execution complete with submit()?


# ============= Metadata TypedDicts =============


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


# ============= Result Types (NamedTuples) =============


class TriangulationResult(NamedTuple):
    """Result from triangulate_teacher()."""

    gold_trace: "TraceDict"
    gold_conversation: list[dict]
    system_prompt: str
    consistency_results: list[tuple["TraceDict", list[dict]]]
    verified: bool
    timing_metadata: dict
    majority_answer_hash: str | None
    majority_count: int


class BatchTriangulationResult(NamedTuple):
    """Result from batch_triangulate() - includes the question."""

    question: dict
    gold_trace: "TraceDict"
    gold_conversation: list[dict]
    system_prompt: str
    consistency_results: list[tuple["TraceDict", list[dict]]]
    verified: bool
    timing_metadata: dict
    majority_answer_hash: str | None
    majority_count: int


# ============= Core Pydantic Models =============


class Question(BaseModel):
    """A question with metadata."""

    question_text: str
    hint: str | None = None
    difficulty: str | None = None  # EASY, MEDIUM, HARD, VERY_HARD
    n_steps: int | None = None  # Expected step count
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

    code_line: str  # The code that produced this
    variable_name: str | None = None  # e.g., 'df_filtered'
    value_hash: str  # Hash of the value at this point
    value: Any = None  # Normalized value for PRM training
    description: str | None = None  # Optional semantic description
    depends_on: list[str] = Field(
        default_factory=list
    )  # Names of hooks this depends on (DAG edges)


# ============= Episode JSONL Schema (NEW) =============


class EpisodeJSONL(BaseModel):
    """
    Episode formatted for JSONL training data.

    This is the canonical format for storing training episodes.
    All training formats (SFT, RL, PRM) are derived from this structure
    at training time, not at data generation time.
    """

    episode_id: str
    timestamp: datetime
    csv_source: str

    # Question
    question: QADict

    # Traces (source of truth)
    gold_trace: TraceDict  # Teacher WITH hint
    consistency_traces: list[TraceDict]  # Teacher WITHOUT hint (N runs)

    # Verification
    verified: bool
    triangulation: TriangulationMetadataDict
    timing: TimingMetadataDict

    # Provenance (optional)
    source: str | None = None  # "synthetic" or "llm" - tracks question origin

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
    questions_generated: list[QADict]
    total_turns: int
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(arbitrary_types_allowed=True)
