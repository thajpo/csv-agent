import argparse
import sys
from pathlib import Path
import yaml

from src.training.environment import Environment
from src.utils.rich_logger import setup_rich_logger
from src.core.types import EnvironmentConfig
from src.authoring.prompts import DEFAULT_DATASET_DESCRIPTION, generate_data_overview
from src.core.prompts import build_rollout_config

# Constants
PARSER_DESCRIPTION = "CSV Exploration Agent with Rich Terminal UI"
DEFAULT_CONFIG_PATH = "config.yaml"

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    return {}

def parse_args(config: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PARSER_DESCRIPTION)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config YAML file")
    parser.add_argument("--csv", default=config.get("csv", "csv/data.csv"), help="Path to CSV file")
    parser.add_argument("--max-turns", type=int, default=config.get("max_turns", 10), help="Maximum conversation turns")
    parser.add_argument("--description", default=config.get("description"), help="Dataset description (uses default if not provided)")
    parser.add_argument("--output", "-o", default=config.get("output"), help="Output path for JSONL")
    parser.add_argument("--mode", choices=["teacher-tutor", "teacher-consistency", "student"], default=config.get("mode", "teacher-tutor"), help="Pipeline mode: teacher-tutor=solve with hints, teacher-consistency=solve without hints, student=RL training")
    parser.add_argument("--target-questions", type=int, default=config.get("target_questions", 10), help="Number of question blueprints to generate in explore mode")
    parser.add_argument("--question", default=config.get("question", "What is the mean TL (total length) for the control group?"), help="Question for the teacher/student to solve")
    parser.add_argument("--hint", default=config.get("hint", "Filter the data to the control group first, then calculate the mean."), help="Hint for teacher-tutor mode")
    parser.add_argument("--teacher-model", default=config.get("teacher_model"), help="Teacher model identifier for metadata")
    # Context management args
    parser.add_argument("--max-active-turns", type=int, default=config.get("max_active_turns", 5), help="Maximum number of active turns to keep in context (older turns are archived)")
    parser.add_argument("--max-context-tokens", type=int, default=config.get("max_context_tokens", 80000), help="Maximum context tokens before purging oldest turns")
    return parser.parse_args()

def main():
    # Parse command-line arguments first to get config file path
    parser = argparse.ArgumentParser(description="CSV Exploration Agent with Rich Terminal UI")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    args_prelim, _ = parser.parse_known_args()

    # Load config from YAML file
    config = load_config(args_prelim.config)

    # Parse all arguments with config file defaults
    args = parse_args(config)

    # Validate required config values
    if not args.teacher_model:
        print("Error: teacher_model must be specified in config.yaml")
        sys.exit(1)
    
    description = args.description or DEFAULT_DATASET_DESCRIPTION
    pipeline_mode = args.mode
    data_overview = generate_data_overview(args.csv)

    # Get sampling args from config
    sampling_args = config.get("sampling_args", {})

    try:
        # Environment parameters
        env_config = EnvironmentConfig(
            csv_path=args.csv,
            model=args.teacher_model,
            max_turns=args.max_turns,
            pipeline_mode=pipeline_mode,
            target_questions=args.target_questions,
            max_active_turns=args.max_active_turns,
            max_context_tokens=args.max_context_tokens,
        )

        # The rollout config is used to define environment interaction with the agent
        rollout_config = build_rollout_config(
            mode=pipeline_mode,
            dataset_description=description,
            data_overview=data_overview,
            question_text=args.question,
            hint=args.hint if pipeline_mode == "teacher-tutor" else "",
            target_questions=args.target_questions,
        )

        env = Environment(
            csv_path=args.csv,
            config=env_config,
            sampling_args=sampling_args,
            rollout_config=rollout_config,
            logger=setup_rich_logger(rollout_config),
        )

        # Run the episode
        final_state = env.rollout()

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
