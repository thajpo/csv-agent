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

from src.datagen.teacher import batch_triangulate
from src.core.prompts import generate_data_overview, DEFAULT_DATASET_DESCRIPTION
from src.datagen.types import Episode, EpisodeJSONL
from src.core.types import Question
from src.utils.logger import create_logger


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

    # Load questions from question_gen.py output
    questions_file = config.get("questions_json", "question/questions.json")
    questions = load_questions(questions_file)
    logger = create_logger()  # Create logger early for this log
    logger.info(f"Loaded {len(questions)} questions from {questions_file}")

    # Output as single JSONL file
    output_jsonl = Path(config.get("episodes_jsonl", "episodes/episodes.jsonl"))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Generate data overview
    data_overview = generate_data_overview(csv_path)

    # Sampling args
    sampling_args = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("pipeline_start", extra={
        "n_questions": len(questions),
        "n_consistency": n_consistency,
        "model": teacher_model,
        "output_file": str(output_jsonl),
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

    # Convert to JSONL episodes and save
    episodes_jsonl = []
    episodes_verified = 0

    for q_dict, gold_trace, gold_conversation, system_prompt, consistency_results, verified in results:
        # Create Question object
        question_obj = Question(
            question_text=q_dict["question"],
            hint=q_dict.get("hint"),
            difficulty=q_dict.get("difficulty"),
            n_steps=q_dict.get("n_steps"),
        )

        # Extract consistency traces (ignore conversations)
        consistency_traces = [trace for trace, _ in consistency_results]
        consistency_conversations = [conv for _, conv in consistency_results]

        # Create Episode object
        episode = Episode(
            id=str(uuid.uuid4()),
            question=question_obj,
            teacher_trace=gold_trace,
            consistency_traces=consistency_traces,
            verified=verified,
            timestamp=datetime.now(),
        )

        # Convert to JSONL format
        episode_jsonl = EpisodeJSONL.from_episode(
            episode=episode,
            gold_conversation=gold_conversation,
            system_prompt=system_prompt,
            consistency_conversations=consistency_conversations,
        )

        # Save if verified OR verified_only is False
        if verified or not verified_only:
            episodes_jsonl.append(episode_jsonl)
            if verified:
                episodes_verified += 1

            logger.info("episode_created", extra={
                "verified": verified,
                "question": q_dict["question"][:50] + "...",
            })

    # Write JSONL file (one episode per line)
    with open(output_jsonl, 'w') as f:
        for ep in episodes_jsonl:
            f.write(json.dumps(ep.model_dump(), default=str) + '\n')

    # Summary
    logger.info("pipeline_complete", extra={
        "output_file": str(output_jsonl),
        "total_questions": len(questions),
        "episodes_saved": len(episodes_jsonl),
        "episodes_verified": episodes_verified,
        "verification_rate": episodes_verified / len(questions) if questions else 0.0,
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
