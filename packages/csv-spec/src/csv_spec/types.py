"""
Type definitions for CSV Agent.

This is the CONTRACT between environment and trainer. All shared types in one place:
- Core types (Question, Hook)
- Turn-based types (ExecutionResult, Turn, Trace)
- Episode types (EpisodeJSONL)
- Prefix-value research types (TrajectoryPrefix, PrefixValueRecord)
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
from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    ground_truth_hashes: (
        list[str] | None
    )  # All valid answer hashes (for multi-outcome validation)
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
    event_line: int | None  # 1-based source line where hook() executed
    event_provenance_reason: str | None


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


# ============= Prefix-Value Research Types =============


class TrajectoryPrefix(BaseModel):
    """Public agent state at a completed turn boundary.

    The prefix intentionally excludes the expected answer and its hashes. Those
    are private verifier inputs and must never become critic features.
    ``consumed_turns`` counts all actor responses against the horizon, including
    format-invalid responses that produced no execution turn.
    """

    prefix_id: str
    episode_id: str
    csv_source: str
    system_prompt: str
    question_text: str
    turns: list[TurnDict]
    turn_responses: list[str]
    turn_completed: list[bool]
    conversation_messages: list[dict[str, str]]
    consumed_turns: int = Field(ge=0)
    max_turns: int = Field(gt=0)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="after")
    def validate_nonterminal_boundary(self) -> "TrajectoryPrefix":
        if self.consumed_turns >= self.max_turns:
            raise ValueError("prefix must leave at least one continuation turn")
        if self.consumed_turns < len(self.turns):
            raise ValueError("consumed_turns cannot be less than executed turns")
        if len(self.turn_responses) != len(self.turns):
            raise ValueError("turn_responses must align with prefix turns")
        if len(self.turn_completed) != len(self.turns):
            raise ValueError("turn_completed must align with prefix turns")
        if any(self.turn_completed):
            raise ValueError("prefix must not contain a terminal turn")
        for expected_index, turn in enumerate(self.turns):
            if turn.get("turn_index") != expected_index:
                raise ValueError("prefix turns must be contiguous and zero-indexed")
        for message in self.conversation_messages:
            if set(message) != {"role", "content"}:
                raise ValueError("conversation messages must contain role and content")
            if message["role"] not in {"assistant", "user"}:
                raise ValueError("prefix conversation cannot contain system messages")
        message_roles = [message["role"] for message in self.conversation_messages]
        if any(
            left == right
            for left, right in zip(message_roles, message_roles[1:], strict=False)
        ) or (message_roles and message_roles[-1] != "user"):
            raise ValueError(
                "prefix conversation must alternate and end with user feedback"
            )
        represented_turns = message_roles.count("user")
        if represented_turns > self.consumed_turns:
            raise ValueError("conversation cannot contain more turns than consumed")
        if self.consumed_turns == 0 and self.conversation_messages:
            raise ValueError("an initial prefix cannot contain conversation messages")
        if self.consumed_turns > 0 and not self.conversation_messages:
            raise ValueError("a consumed prefix must retain conversation messages")
        if self.turns:
            if len(self.conversation_messages) < 2:
                raise ValueError("prefix conversation is missing the completed turn")
            assistant, feedback = self.conversation_messages[-2:]
            if (
                assistant
                != {
                    "role": "assistant",
                    "content": self.turn_responses[-1],
                }
                or feedback["role"] != "user"
            ):
                raise ValueError(
                    "prefix conversation does not end at its turn boundary"
                )
        return self


class ContinuationPolicy(BaseModel):
    """Frozen actor settings that give a prefix value its meaning."""

    model: str
    sampling_args: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class PrefixContinuation(BaseModel):
    """One attempted continuation and its terminal-verifier judgment."""

    rollout_index: int = Field(ge=0)
    seed: int | None = None
    trace: TraceDict | None = None
    verifier_verdict: bool | None = None
    error: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="after")
    def validate_outcome(self) -> "PrefixContinuation":
        if self.verifier_verdict is not None and self.trace is None:
            raise ValueError("a verifier verdict requires a continuation trace")
        if self.error is not None and self.verifier_verdict is not None:
            raise ValueError("errored continuations cannot carry verifier labels")
        return self


class PrefixValueRecord(BaseModel):
    """Auditable future-success estimate over attempted continuations.

    A numeric value is available only when every attempt has a verifier label.
    Infrastructure or verifier errors make the estimate unavailable instead of
    treating system failure as policy failure.
    """

    prefix: TrajectoryPrefix
    policy: ContinuationPolicy
    continuations: list[PrefixContinuation]
    attempted_continuations: int = Field(ge=0)
    labeled_continuations: int = Field(ge=0)
    successful_continuations: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    code_commit: str
    dataset_revision: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="after")
    def validate_aggregate(self) -> "PrefixValueRecord":
        attempted = len(self.continuations)
        labeled = sum(
            outcome.verifier_verdict is not None for outcome in self.continuations
        )
        successes = sum(
            outcome.verifier_verdict is True for outcome in self.continuations
        )
        expected_value = (
            successes / attempted if attempted and labeled == attempted else None
        )

        if self.attempted_continuations != attempted:
            raise ValueError("attempted_continuations does not match outcomes")
        if self.labeled_continuations != labeled:
            raise ValueError("labeled_continuations does not match verifier verdicts")
        if self.successful_continuations != successes:
            raise ValueError(
                "successful_continuations does not match verifier verdicts"
            )
        if self.value != expected_value:
            raise ValueError(
                "value must equal successful divided by attempted continuations"
            )
        return self


# ============= Metadata TypedDicts =============


class TriangulationMetadataDict(TypedDict):
    """Triangulation results."""

    n_consistency_runs: int
    n_consistency_succeeded: int
    majority_answer_hash: str | None
    majority_count: int
    gold_matches_majority: bool
    float_tolerance: float


class TimingMetadataDict(TypedDict):
    """Execution timing for episode generation."""

    gold_elapsed: float
    consistency_elapsed: list[float]
    total_elapsed: float
    avg_elapsed: float


# ============= Process Report Types =============


class ProcessStepEvidenceDict(TypedDict):
    """Observed evidence about a process step.

    Evidence fields are diagnostic inputs. Except for terminal verification,
    they do not establish that an action made useful computational progress.
    """

    final_verified: bool | None
    trace_success: bool
    code_line_grounded: bool
    dependency_valid: bool
    duplicate: bool
    consensus_matches: int
    consensus_total: int
    reasons: list[str]


class ProcessStepReportDict(TypedDict):
    """One observed process step with verified or heuristic judgment."""

    step_index: int
    turn_index: int
    hook_index: int | None
    step_type: Literal["hook", "submit"]
    code_line: str
    variable_name: str | None
    value: Any
    value_hash: str | None
    description: str | None
    depends_on: list[str]
    semantic_role: str | None
    label: float | None
    label_kind: Literal["verified", "heuristic", "unlabeled"]
    label_source: str
    evidence: ProcessStepEvidenceDict


class ProcessReportSummaryDict(TypedDict):
    """Aggregate process-step judgment counts for one episode."""

    total_steps: int
    labeled_steps: int
    verified_steps: int
    heuristic_steps: int
    unlabeled_steps: int
    positive_steps: int
    negative_steps: int


class ProcessReportDict(TypedDict):
    """Ordered diagnostic process observations stored on each episode."""

    summary: ProcessReportSummaryDict
    steps: list[ProcessStepReportDict]


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
    ground_truth_hashes: list[str] | None = (
        None  # All valid answer hashes (for multi-outcome validation)
    )

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
    event_line: int | None = None  # 1-based source line where hook() executed
    event_provenance_reason: str | None = None


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
    process_report: ProcessReportDict

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
