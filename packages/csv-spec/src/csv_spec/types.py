"""
Type definitions for CSV Agent.

This is the CONTRACT between environment and trainer. All shared types in one place:
- Core types (Question, Hook)
- Turn-based types (ExecutionResult, Turn, Trace)
- Episode types (EpisodeJSONL)
- Action/Step contract types (ActionSpec, StepResult) - NEW
- Exploration types (ExplorationTurn, ExplorationTrace)
- TypedDicts for JSONL serialization

IMPORTANT: Changes here affect both environment and trainer.
If you modify any type, you MUST update:
1. Environment parsing/validation
2. Trainer action formatting/consumption
3. Test fixtures
"""

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal, NamedTuple, TypedDict, Union
from datetime import datetime


# ============= Core TypedDicts =============


class QADict(TypedDict, total=False):
    """Serialized Question structure."""

    id: str | None
    question_text: str
    hint: str | None
    difficulty: str | None
    n_steps: int | None
    created_at: str | None
    category: str | None
    tags: list[str] | None
    template_name: str | None
    template_params: dict[str, Any] | None
    output_type: str | None
    output_schema: str | None
    ground_truth_hash: str | None
    ground_truth_hashes: list[str] | None  # All valid answer hashes (for multi-outcome validation)
    ground_truth: Any | None


class HookDict(TypedDict, total=False):
    """Captured intermediate state during execution.

    Value storage policy (for PRM training):
        - Scalars (int, float, str, bool, None): Stored in full
        - DataFrame/Series: Bounded summary (shape, dtypes, head rows, numeric stats)
        - Other complex types (dict, list): Stored if < 100KB, else type+size metadata
        - value_hash always computed on full normalized value for verification
    """

    variable_name: str | None
    code_line: str  # The code that produced this value
    value: Any  # Scalar, summary dict, or type metadata (always present for PRM)
    value_hash: str  # Hash of full normalized value for verification
    depends_on: list[str]  # DAG edges to prior hooks
    description: str | None


# ============= Turn-Based TypedDicts =============


class ExecutionResultDict(TypedDict):
    """Result of executing one code cell."""

    success: bool
    stdout: str
    stderr: str
    hooks: list[HookDict]  # Hooks captured in THIS cell
    submitted_answer: Any | None  # If submit() was called in this cell


class CodeDiffDict(TypedDict):
    """Simple diff showing what changed between failed and fixed code."""

    removed_lines: list[str]  # Lines in failed code but not in fixed
    added_lines: list[str]  # Lines in fixed code but not in failed


class CorrectionDict(TypedDict, total=False):
    """Metadata about a self-correction (when this turn fixes a previous failure).

    This enables training models on error recovery behavior:
    - Recognize errors from feedback
    - Diagnose the issue
    - Generate appropriate fixes
    """

    corrects_turn: int  # Index of the failed turn this corrects
    error_type: str  # Exception class: "KeyError", "ValueError", "SyntaxError", etc.
    error_message: str  # The specific error message
    attempts_since_error: int  # How many turns since the error (usually 1)
    code_diff: CodeDiffDict  # What changed between failed and fixed code


class TurnDict(TypedDict, total=False):
    """Single turn = model output + execution result.

    Optional `correction` field is present when this turn successfully
    fixes a previous failed turn - useful for self-correction training.
    """

    turn_index: int
    reasoning: str  # Model's thinking/explanation
    code: str  # Code block (single cell for now)
    execution: ExecutionResultDict  # What happened when code ran
    correction: CorrectionDict | None  # Present if this turn fixes a previous failure


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


# ============= Diagnostic Types =============


class FailureCategory(str, Enum):
    """Classification of triangulation outcomes.

    Used to distinguish WHY a question failed:
    - GOOD: Verified successfully (gold matches majority)
    - AMBIGUOUS: Multiple distinct answer clusters (question has multiple interpretations)
    - TOO_HARD: Single wrong cluster (model consistently gets wrong answer)
    - HINT_NECESSARY: Gold only succeeds with hint
    - EXECUTION_FAILURE: Most traces failed to execute
    """

    GOOD = "good"
    AMBIGUOUS = "ambiguous"
    TOO_HARD = "too_hard"
    HINT_NECESSARY = "hint_necessary"
    EXECUTION_FAILURE = "execution_failure"


class AnswerClusterDict(TypedDict):
    """A cluster of equivalent answers from consistency traces."""

    answer_hash: str
    member_count: int
    representative_answer: Any
    member_indices: list[int]  # Which consistency trace indices


class AnswerDistributionDict(TypedDict):
    """Statistical summary of consistency trace answers."""

    total_traces: int
    successful_traces: int
    execution_failures: int
    cluster_count: int
    entropy: float  # Shannon entropy - high = ambiguous
    majority_confidence: float  # majority_count / successful_traces
    clusters: list[AnswerClusterDict]


class DiagnosticMetadataDict(TypedDict, total=False):
    """Rich diagnostic information for failure analysis."""

    failure_category: str  # FailureCategory value
    answer_distribution: AnswerDistributionDict
    gold_answer_hash: str | None
    gold_execution_success: bool
    gold_matches_majority: bool
    gold_cluster_index: int | None  # Which cluster gold belongs to (-1 if none)
    classification_confidence: float
    classification_reasoning: str


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
    diagnostics: "DiagnosticMetadataDict | None" = None  # Optional failure analysis


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
    diagnostics: "DiagnosticMetadataDict | None" = None  # Optional failure analysis


# ============= Core Pydantic Models =============


class Question(BaseModel):
    """A question with metadata."""

    question_text: str
    hint: str | None = None
    difficulty: str | None = None  # EASY, MEDIUM, HARD, VERY_HARD
    n_steps: int | None = None  # Expected step count
    category: str | None = None
    tags: list[str] | None = None
    template_name: str | None = None
    template_params: dict[str, Any] | None = None
    output_type: str | None = None
    output_schema: str | None = None
    ground_truth_hash: str | None = None
    ground_truth_hashes: list[str] | None = None  # All valid answer hashes (for multi-outcome validation)

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

    Value storage policy (for PRM training):
        - Scalars (int, float, str, bool, None): Stored in full
        - DataFrame/Series: Bounded summary (shape, dtypes, head rows, numeric stats)
        - Other complex types (dict, list): Stored if < 100KB, else type+size metadata
        - value_hash always computed on full normalized value for verification
    """

    code_line: str  # The code that produced this
    variable_name: str | None = None  # e.g., 'df_filtered'
    value_hash: str  # Hash of full normalized value for verification
    value: Any = None  # Scalar, summary dict, or type metadata (always present)
    description: str | None = None  # Optional semantic description
    depends_on: list[str] = Field(
        default_factory=list
    )  # Names of hooks this depends on (DAG edges)


# ============= Episode JSONL Schema =============


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
    execution_results: list[Any]  # CodeCellResult from conversation.py
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


# ============= ACTION/STEP CONTRACT (NEW) =============
# These types define the interface between trainer and environment.
# The trainer produces actions, the environment returns step results.


class CodeAction(BaseModel):
    """Model writes Python code to execute.

    This is the primary action type - model submits code to run.
    """

    code: str


class SubmitAction(BaseModel):
    """Model submits final answer (terminal action).

    This action ends the episode. The answer is evaluated against ground truth.
    """

    answer: Any
    hooks: list[HookDict] = Field(default_factory=list)


# Union type for all possible actions
ActionSpec = Union[CodeAction, SubmitAction]


class StepResult(BaseModel):
    """Structured result from environment after executing an action.

    This is what the environment returns after processing an action.
    Both env and trainer must agree on this structure.
    """

    success: bool  # Did execution succeed without errors?
    stdout: str  # Normalized/truncated output
    stderr: str  # Error output if any
    hooks: list[HookDict] = Field(default_factory=list)  # Captured hooks
    submitted_answer: Any | None = None  # If submit() was called
    terminal: bool = False  # Is the episode done?
    terminal_reason: Literal["submit", "max_turns", "error"] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
