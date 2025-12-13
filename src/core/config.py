"""
Pydantic nested configuration for CSV agent.

This module provides focused, single-responsibility config classes:
- DataConfig: Dataset paths and metadata
- ModelConfig: LLM model and sampling parameters
- ExecutionConfig: Execution limits (turns, tokens, context)
- TaskConfig: Task definition (mode, question)
- Config: Main nested container
"""

from pydantic import BaseModel, Field
from src.core.types import Question


class DataConfig(BaseModel):
    """Configuration for dataset access and metadata."""

    csv_path: str = "data.csv"
    dataset_description: str = ""
    data_overview: str = ""


class ModelConfig(BaseModel):
    """Configuration for LLM model and sampling parameters."""

    model_name: str = "grok-4.1-fast"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)  # Validated range
    max_tokens: int = Field(default=1000, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)

    def sampling_args(self) -> dict:
        """Extract sampling args for LLM API."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


class ExecutionConfig(BaseModel):
    """Configuration for execution limits and context management."""

    max_turns: int = Field(default=10, gt=0)
    max_active_turns: int = Field(default=5, gt=0)  # For conversation pruning
    max_context_tokens: int = Field(default=80_000, gt=0)


class TaskConfig(BaseModel):
    """Configuration for the specific task to execute."""

    mode: str = "teacher-tutor"  # teacher-tutor, teacher-consistency, student, question-gen
    question: Question | None = None
    target_questions: int = Field(default=10, gt=0)  # For question-gen mode


class Config(BaseModel):
    """Main configuration with nested sub-configs."""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    execution: ExecutionConfig = ExecutionConfig()
    task: TaskConfig = TaskConfig()
