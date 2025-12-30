"""
csv-spec: Type contracts for csv-agent.

This package defines the contract between environment and trainer:
- Types: What actions/results look like
- Normalization: How to canonicalize values for comparison
- Parsing: How to extract actions from model output and results from execution

If you change anything here, you MUST update both:
1. Environment (csv_env.py) - how it parses/validates
2. Trainer (rl_env.py, prompts) - how it formats/consumes
"""

from csv_spec.types import (
    # Core types
    Question,
    Hook,
    # TypedDicts for serialization
    QADict,
    HookDict,
    ExecutionResultDict,
    TurnDict,
    TraceDict,
    CorrectionDict,
    CodeDiffDict,
    TriangulationMetadataDict,
    TimingMetadataDict,
    # Diagnostic types
    FailureCategory,
    AnswerClusterDict,
    AnswerDistributionDict,
    DiagnosticMetadataDict,
    # Episode schema
    EpisodeJSONL,
    # Exploration types
    ExplorationTurn,
    ExplorationTrace,
    # Result types
    TriangulationResult,
    BatchTriangulationResult,
    # Action/Step contract (NEW)
    CodeAction,
    SubmitAction,
    ActionSpec,
    StepResult,
)

from csv_spec.normalization import normalize_value
from csv_spec.hashing import hash_artifact
from csv_spec.parsing import parse_action, parse_step_result

__version__ = "0.1.0"

__all__ = [
    # Core types
    "Question",
    "Hook",
    # TypedDicts
    "QADict",
    "HookDict",
    "ExecutionResultDict",
    "TurnDict",
    "TraceDict",
    "CorrectionDict",
    "CodeDiffDict",
    "TriangulationMetadataDict",
    "TimingMetadataDict",
    # Diagnostic types
    "FailureCategory",
    "AnswerClusterDict",
    "AnswerDistributionDict",
    "DiagnosticMetadataDict",
    # Episode
    "EpisodeJSONL",
    # Exploration
    "ExplorationTurn",
    "ExplorationTrace",
    # Results
    "TriangulationResult",
    "BatchTriangulationResult",
    # Contract types
    "CodeAction",
    "SubmitAction",
    "ActionSpec",
    "StepResult",
    # Functions
    "normalize_value",
    "hash_artifact",
    "parse_action",
    "parse_step_result",
]
