"""
Pydantic nested configuration for CSV agent.

This module provides focused, single-responsibility config classes:
- SamplingArgs: Model sampling parameters
- Config: Main configuration source of truth for the project.

And legacy sub-configs for Environment integration:
- DataConfig
- ModelConfig
- ExecutionConfig
- TaskConfig
"""

import json
import os
from pathlib import Path
from typing import List, Literal, Union, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from src.core.types import Question


def discover_kaggle_datasets(kaggle_dir: str = "data/kaggle") -> list[str]:
    """
    Discover dataset CSV paths from Kaggle directory structure.

    Expects structure:
        data/kaggle/{slug}/data.csv
        data/kaggle/{slug}/meta.json
        data/kaggle/manifest.json

    Returns list of absolute paths to data.csv files.
    """
    kaggle_path = Path(kaggle_dir)
    if not kaggle_path.exists():
        return []

    manifest_path = kaggle_path / "manifest.json"
    if manifest_path.exists():
        # Use manifest for ordering
        with open(manifest_path) as f:
            manifest = json.load(f)
        return [
            str((kaggle_path / entry["slug"] / "data.csv").resolve())
            for entry in manifest
            if (kaggle_path / entry["slug"] / "data.csv").exists()
        ]

    # Fallback: scan directories
    return [str(p.resolve()) for p in sorted(kaggle_path.glob("*/data.csv"))]


# =============================================================================
# 1. Base Components (Used globally)
# =============================================================================


class SamplingArgs(BaseModel):
    """Configuration for LLM sampling parameters."""

    temperature: float = 0.7
    max_tokens: int = 6000
    top_p: float = 1.0


# =============================================================================
# 2. Main Application Configuration (Source of Truth)
# =============================================================================


class Config(BaseModel):
    """
    Main application configuration.
    This class defines the schema and default values for the entire project.
    Values are managed here in Python.
    """

    model_config = ConfigDict(extra="ignore")

    # Data - auto-discovers from kaggle_dir by default
    kaggle_dir: str = "data/kaggle"
    csv_sources: Union[str, List[str]] = Field(
        default_factory=lambda: discover_kaggle_datasets()
    )

    # Execution / Policy
    max_turns: int = 10
    mode: str = "teacher-tutor"
    question: Optional[str] = (
        "What is the mean TL (total length) for the control group?"
    )

    # Models
    teacher_model: str = Field(default="openai/gpt-oss-120b")
    question_gen_model: str = Field(default="openai/gpt-oss-120b")
    sampling_args: SamplingArgs = Field(default_factory=SamplingArgs)

    # Context / Memory
    max_active_turns: int = 5
    max_context_tokens: int = 80000

    # Question source: "llm" (LLM-generated) or "synthetic" (template-based)
    question_source: Literal["llm", "synthetic"] = "llm"

    # pipelines: question generation
    question_gen_max_turns: int = 20
    num_questions_to_generate: int = 15
    min_exploration_turns: int = 3  # Minimum turns before allowing question generation
    question_difficulty_distribution: Dict[str, float] = Field(
        default={"EASY": 0.30, "MEDIUM": 0.30, "HARD": 0.20, "VERY_HARD": 0.20}
    )

    # pipelines: episode generation / triangulation
    n_consistency: int = 7  # Optimal for 8-worker containers (profiled)
    verified_only: bool = False
    float_tolerance: float = 0.1
    max_concurrent_containers: int = 30  # Profiled optimal for this machine

    # Outputs - separate paths for LLM vs synthetic pipelines
    questions_llm_dir: str = "data/questions_llm"
    questions_synthetic_dir: str = "data/questions_synthetic"
    episodes_llm_jsonl: str = "data/episodes/episodes_llm.jsonl"
    episodes_synthetic_jsonl: str = "data/episodes/episodes_synthetic.jsonl"

    # Train/Test/Val Splitting
    split_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Template-based generation (optional branch)
    template_questions_output: str = "questions.json"
    template_n_simple: int = 5
    template_n_comparison: int = 5
    template_n_multi_step: int = 3
    template_n_filtering: int = 3
    template_seed: int = 42


# =============================================================================
# 3. Environment Sub-configs (Internal use for src.core.environment)
# =============================================================================


class DataConfig(BaseModel):
    csv_path: str = "data.csv"
    dataset_description: str = ""
    data_overview: str = ""


class ModelConfig(BaseModel):
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 6000  # Must match SamplingArgs.max_tokens
    top_p: float = 1.0

    def sampling_args_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


class ExecutionConfig(BaseModel):
    max_turns: int = Field(default=10, gt=0)
    max_active_turns: int = Field(default=5, gt=0)
    max_context_tokens: int = Field(default=80_000, gt=0)


class TaskConfig(BaseModel):
    mode: str = "teacher-tutor"
    question: Optional[Question] = None


# =============================================================================
# 4. Global Singleton Configuration
# =============================================================================

# Global singleton configuration object
# This allows 'from src.core.config import config'
config = Config()
