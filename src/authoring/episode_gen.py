"""
Episode generation pipeline.

This script:
1. Loads questions from CSV (TODO: implement)
2. Runs teacher triangulation on each question
3. Saves verified episodes to disk

Usage:
    python -m src.authoring.episode_gen
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import uuid
import yaml

from src.authoring.teacher import batch_triangulate
from src.core.prompts import generate_data_overview, DEFAULT_DATASET_DESCRIPTION
from src.authoring.types import Episode
from src.core.types import Question


def save_episode(episode: Episode, output_dir: Path) -> Path:
    """
    Save episode as JSON file.

    Args:
        episode: Episode to save
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename: {id}.json
    filepath = output_dir / f"{episode.id}.json"

    with open(filepath, 'w') as f:
        json.dump(episode.model_dump(), f, indent=2, default=str)

    return filepath


def load_questions(questions_path: str) -> list[dict]:
    """Load questions from JSON file."""
    with open(questions_path) as f:
        return json.load(f)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    with open(config_file) as f:
        return yaml.safe_load(f)


def main():
    # Load config
    config = load_config()

    # Extract config values (fail-fast on missing keys)
    csv_path = config["csv"]
    teacher_model = config["teacher_model"]
    max_turns = config["max_turns"]
    temperature = config["sampling_args"]["temperature"]
    max_tokens = config["sampling_args"]["max_tokens"]
    n_consistency = config["n_consistency"]
    verified_only = config["verified_only"]

    # TODO: Load questions from CSV instead of JSON file
    # questions = load_questions_from_csv(config["csv"])
    questions = []  # Placeholder

    # TODO: Derive output directory from config or add to config.yaml
    output_dir = Path("episodes/")  # Placeholder

    # Generate data overview
    data_overview = generate_data_overview(csv_path)

    # Setup logger
    from src.utils.logger import create_logger
    logger = create_logger()

    # Sampling args
    sampling_args = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("pipeline_start", extra={
        "n_questions": len(questions),
        "n_consistency": n_consistency,
        "model": teacher_model,
        "output_dir": str(output_dir),
    })

    # Run batch triangulation
    results = batch_triangulate(
        csv_path=csv_path,
        questions=questions,
        n_consistency=n_consistency,
        model=teacher_model,
        dataset_description=DEFAULT_DATASET_DESCRIPTION,
        data_overview=data_overview,
        max_turns=max_turns,
        sampling_args=sampling_args,
        logger=logger,
    )

    # Convert to Episodes and save
    episodes_saved = 0
    episodes_verified = 0

    for q_dict, gold_trace, consistency_traces, verified in results:
        # Create Episode object
        question_obj = Question(
            question_text=q_dict["question"],
            hint=q_dict.get("hint"),
            difficulty=q_dict.get("difficulty"),
        )

        episode = Episode(
            id=str(uuid.uuid4()),
            question=question_obj,
            teacher_trace=gold_trace,
            consistency_traces=consistency_traces,
            verified=verified,
            timestamp=datetime.now(),
        )

        # Save if verified OR if verified_only is False
        if verified or not verified_only:
            filepath = save_episode(episode, output_dir)
            episodes_saved += 1

            if verified:
                episodes_verified += 1

            logger.info("episode_saved", extra={
                "filepath": str(filepath),
                "verified": verified,
                "question": q_dict["question"][:50] + "...",
            })

    # Summary
    logger.info("pipeline_complete", extra={
        "total_questions": len(questions),
        "episodes_saved": episodes_saved,
        "episodes_verified": episodes_verified,
        "verification_rate": episodes_verified / len(questions) if questions else 0.0,
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
