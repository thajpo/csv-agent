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

import os
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from src.core.types import Question


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
    model_config = ConfigDict(extra='ignore')

    # Data
    csv_sources: Union[str, List[str]] = Field(default="data/csv/data.csv")

    # Execution / Policy
    max_turns: int = 10
    mode: str = "teacher-tutor"
    question: Optional[str] = "What is the mean TL (total length) for the control group?"
    hint: Optional[str] = "Filter the data to the control group first, then calculate the mean."
    target_questions: int = 10

    # Models
    teacher_model: str = Field(default="openai/gpt-oss-120b")
    question_gen_model: str = Field(default="openai/gpt-oss-120b")
    sampling_args: SamplingArgs = Field(default_factory=SamplingArgs)

    # Context / Memory
    max_active_turns: int = 5
    max_context_tokens: int = 80000

    # pipelines: question generation
    question_gen_max_turns: int = 20
    num_questions_to_generate: int = 30

    # pipelines: episode generation / triangulation
    n_consistency: int = 5
    verified_only: bool = False
    float_tolerance: float = 0.1

    # Container pool settings (for parallel processing)
    n_containers: int = 2  # Number of Docker containers to create
    workers_per_container: int = 6  # Worker processes per container (fork-based)

    # Outputs
    questions_json: str = "data/questions/questions.json"
    episodes_jsonl: str = "data/episodes/episodes.jsonl"

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
    max_tokens: int = 1000
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
    target_questions: int = Field(default=10, gt=0)


# =============================================================================
# 4. Global Singleton Configuration
# =============================================================================

# Global singleton configuration object
# This allows 'from src.core.config import config'
config = Config()
