import sys
from pathlib import Path
import yaml

from src.core.environment import Environment
from src.utils.logger import create_logger
from src.core.config import DataConfig, ModelConfig, ExecutionConfig, TaskConfig
from src.core.types import Question
from src.core.prompts import (
    DEFAULT_DATASET_DESCRIPTION,
    generate_data_overview,
)

# Constants
DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    with open(config_file) as f:
        return yaml.safe_load(f)


def main():
    # Load config from YAML file
    config = load_config()

    # Extract config values (fail-fast on missing keys)
    csv_path = config["csv"]
    max_turns = config["max_turns"]
    # TODO : pull the dataset description from the question set
    description = config["description"] or DEFAULT_DATASET_DESCRIPTION
    pipeline_mode = config["mode"]
    question = config["question"]
    hint = config["hint"]
    teacher_model = config["teacher_model"]
    target_questions = config["target_questions"]
    max_active_turns = config["max_active_turns"]
    max_context_tokens = config["max_context_tokens"]
    sampling_args = config["sampling_args"]

    data_overview = generate_data_overview(csv_path)

    try:
        # Build question object
        question_obj = Question(question_text=question, hint=hint) if question else None

        data_config = DataConfig(
            csv_path=csv_path,
            dataset_description=description,
            data_overview=data_overview,
        )

        model_config = ModelConfig(
            model_name=teacher_model,
            **sampling_args  # Unpack temperature, max_tokens, top_p
        )

        execution_config = ExecutionConfig(
            max_turns=max_turns,
            max_active_turns=max_active_turns,
            max_context_tokens=max_context_tokens,
        )

        task_config = TaskConfig(
            mode=pipeline_mode, # teacher-tutor, teacher-consistency, student, question-gen
            question=question_obj,
            target_questions=target_questions,
        )

        env = Environment(
            data=data_config,
            model=model_config,
            execution=execution_config,
            task=task_config,
            logger=create_logger(),
        )

        # Run the episode
        state = env.rollout()

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
