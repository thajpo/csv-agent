"""
Evaluation harness for CSV agent models.

This module provides tools for evaluating trained models against test episodes.
"""

from src.eval.metrics import EvalResult, EvalMetrics
from src.eval.evaluator import Evaluator
from src.eval.report import generate_report
from src.eval.synthetic import SyntheticEvaluator, SyntheticEvalMetrics

__all__ = [
    "EvalResult",
    "EvalMetrics",
    "Evaluator",
    "generate_report",
    "SyntheticEvaluator",
    "SyntheticEvalMetrics",
]
