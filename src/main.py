import argparse
import sys
from pathlib import Path
import yaml

from src.environment import Environment
from src.rich_logger import setup_rich_logger
from src.types import EnvironmentConfig
from src.prompts import DEFAULT_DATASET_DESCRIPTION, build_rollout_config
from src.prompts import generate_data_overview

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
    parser.add_argument("--csv", default=config.get("csv", "data.csv"), help="Path to CSV file")
    parser.add_argument("--max-turns", type=int, default=config.get("max_turns", 10), help="Maximum conversation turns")
    parser.add_argument("--description", default=config.get("description"), help="Dataset description (uses default if not provided)")
    parser.add_argument("--output", "-o", default=config.get("output"), help="Output path for JSONL")
    parser.add_argument("--mode", choices=["question-gen", "answer", "tool-feedback"], default=config.get("mode", "question-gen"), help="Pipeline mode: question-gen=question plans, answer=full hook episodes, tool-feedback=tool gap analysis")
    parser.add_argument("--target-questions", type=int, default=config.get("target_questions", 10), help="Number of question blueprints to generate in explore mode")
    parser.add_argument("--tool-feedback", action="store_true", help="Run in tool feedback mode to identify missing tools (alias for --mode tool-feedback)")
    parser.add_argument("--teacher-model", default=config.get("teacher_model", "grok-4.1-fast"), help="Teacher model identifier for metadata")
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
    
    description = args.description or DEFAULT_DATASET_DESCRIPTION

    pipeline_mode = args.mode

    data_overview = generate_data_overview(args.csv)

    if pipeline_mode == "question-gen" or pipeline_mode == "tool-feedback":
        target_questions = ""
    elif pipeline_mode == "answer":
        pass # TODO: Implement answer mode
    else:
        raise ValueError(f"Invalid pipeline mode: {pipeline_mode}")
    
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
            target_questions=target_questions,
        )

        env = Environment(
            csv_path=args.csv,
            config=env_config,
            sampling_args=sampling_args,
            rollout_config=rollout_config,
            logger=setup_rich_logger(rollout_config),
        )
        env.rollout(input=rollout_config.system_prompt)

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
