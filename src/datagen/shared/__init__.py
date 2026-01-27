"""Shared utilities for data generation pipeline.

Provides central contracts for dataset metadata, question I/O, submission parsing,
filters, and verification.
"""

from src.datagen.shared.dataset_meta import (
    load_dataset_meta,
    generate_description_from_overview,
)
from src.datagen.shared.questions_io import (
    load_questions,
    save_questions,
    validate_question,
)
from src.datagen.shared.submission import parse_submission, parse_all_submissions
from src.datagen.shared.filters import apply_filters, log_drop, FORBIDDEN_METHOD_TERMS
from src.datagen.shared.verification import (
    verify_question,
    verify_synthetic,
    verify_llm,
    VerificationResult,
)

__all__ = [
    "load_dataset_meta",
    "generate_description_from_overview",
    "load_questions",
    "save_questions",
    "validate_question",
    "parse_submission",
    "parse_all_submissions",
    "apply_filters",
    "log_drop",
    "FORBIDDEN_METHOD_TERMS",
    "verify_question",
    "verify_synthetic",
    "verify_llm",
    "VerificationResult",
]
