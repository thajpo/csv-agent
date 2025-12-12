"""
Shared prompt infrastructure for rollout configuration.

This module contains the RolloutConfig dataclass and builder function
used by both authoring and training pipelines.
"""

from dataclasses import dataclass


@dataclass
class RolloutConfig:
    """Configuration for a rollout (system prompt, intermediate messages)."""
    system_prompt: str
    mode: str
    continue_msg: str
    final_msg: str


def build_rollout_config(
    mode: str,
    dataset_description: str,
    data_overview: str,
    question_text: str = "",
    hint: str = "",
    target_questions: int = 10,
) -> RolloutConfig:
    """
    Build rollout config for the given pipeline mode.

    Args:
        mode: Pipeline mode (teacher-tutor, teacher-consistency, student, etc.)
        dataset_description: Description of the dataset
        data_overview: Pre-computed data exploration
        question_text: The question to solve (for teacher/student modes)
        hint: Optional hint for teacher tutor mode
        target_questions: Number of questions to generate (for question-gen mode)
    """
    if mode == "teacher-tutor":
        from src.authoring.prompts import TEACHER_TUTOR_MODE_PROMPT
        return RolloutConfig(
            system_prompt=TEACHER_TUTOR_MODE_PROMPT.format(
                dataset_description=dataset_description,
                data_overview=data_overview,
                question_text=question_text,
                hint=hint
            ),
            mode=mode,
            continue_msg="\n\nWhat will you do next?",
            final_msg="Turn limit reached. Please call submit() with your final answer.",
        )

    elif mode == "teacher-consistency":
        from src.authoring.prompts import TEACHER_CONSISTENCY_PROMPT
        return RolloutConfig(
            system_prompt=TEACHER_CONSISTENCY_PROMPT.format(
                dataset_description=dataset_description,
                data_overview=data_overview,
                question_text=question_text
            ),
            mode=mode,
            continue_msg="\n\nWhat will you do next?",
            final_msg="Turn limit reached. Please call submit() with your final answer.",
        )

    elif mode == "student":
        from src.training.prompts import STUDENT_PROMPT
        return RolloutConfig(
            system_prompt=STUDENT_PROMPT.format(
                dataset_description=dataset_description,
                data_overview=data_overview,
                question_text=question_text
            ),
            mode=mode,
            continue_msg="\n\nWhat will you do next?",
            final_msg="Turn limit reached. Please call submit() with your final answer.",
        )

    else:
        raise ValueError(f"Unknown mode '{mode}' (expected: teacher-tutor, teacher-consistency, student)")
