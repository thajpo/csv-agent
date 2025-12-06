import argparse
import sys
from pathlib import Path
import yaml

from src.environment import Environment
from src.rich_logger import setup_rich_logger
from src.types import EnvironmentConfig
from src.prompts import DEFAULT_DATASET_DESCRIPTION, get_mode_config


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    # Parse command-line arguments first to get config file path
    parser = argparse.ArgumentParser(description="CSV Exploration Agent with Rich Terminal UI")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    args_prelim, _ = parser.parse_known_args()
    
    # Load config from YAML file
    config = load_config(args_prelim.config)
    
    # Parse all arguments with config file defaults
    parser = argparse.ArgumentParser(description="CSV Exploration Agent with Rich Terminal UI")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--csv", default=config.get("csv", "data.csv"), help="Path to CSV file")
    parser.add_argument("--max-turns", type=int, default=config.get("max_turns", 10), help="Maximum conversation turns")
    parser.add_argument("--description", default=config.get("description"), help="Dataset description (uses default if not provided)")
    parser.add_argument("--output", "-o", default=config.get("output"), help="Output path for JSONL")
    parser.add_argument("--mode", choices=["explore", "episodes", "tool-feedback"], default=config.get("mode", "explore"), help="Pipeline mode: explore=question plans, episodes=full hook episodes, tool-feedback=tool gap analysis")
    parser.add_argument("--target-questions", type=int, default=config.get("target_questions", 10), help="Number of question blueprints to generate in explore mode")
    parser.add_argument("--tool-feedback", action="store_true", help="Run in tool feedback mode to identify missing tools (alias for --mode tool-feedback)")
    parser.add_argument("--teacher-model", default=config.get("teacher_model", "grok-4.1-fast"), help="Teacher model identifier for metadata")
    args = parser.parse_args()
    
    description = args.description or DEFAULT_DATASET_DESCRIPTION
    pipeline_mode = "tool-feedback" if args.tool_feedback else args.mode
    
    # Get sampling args from config
    sampling_args = config.get("sampling_args", {
        "temperature": 0.7,
        "max_tokens": 1000,
    })
    
    try:
        env_config = EnvironmentConfig(
            model=args.teacher_model,
            max_turns=args.max_turns,
            pipeline_mode=pipeline_mode,
            target_questions=args.target_questions,
        )
        mode_config = get_mode_config(pipeline_mode, description, env_config.target_questions)
        env = Environment(
            csv_path=args.csv,
            config=env_config,
            sampling_args=sampling_args,
            logger=setup_rich_logger(mode_config),
        )
        env.rollout(input=mode_config.system_prompt)

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
