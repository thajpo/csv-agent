"""
Episode generation pipeline.

This script:
1. Loads questions from questions.json
2. Runs teacher triangulation on each question
3. Saves verified episodes to disk

Usage:
    python -m scripts.generate_episodes --csv data.csv --questions questions.json --output episodes/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import uuid

from src.teacher import batch_triangulate
from src.prompts import generate_data_overview, DEFAULT_DATASET_DESCRIPTION
from src.types import Episode
from src.rich_logger import setup_rich_logger
from src.prompts import RolloutConfig


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


def main():
    parser = argparse.ArgumentParser(description="Generate verified training episodes")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV file")
    parser.add_argument("--questions", default="questions.json", help="Path to questions JSON")
    parser.add_argument("--output", "-o", default="episodes/", help="Output directory for episodes")
    parser.add_argument("--n-consistency", type=int, default=3, help="Number of consistency traces per question")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Teacher model")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per trace")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max tokens per response")
    parser.add_argument("--verified-only", action="store_true", help="Only save verified episodes")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    questions = load_questions(args.questions)

    # Generate data overview
    data_overview = generate_data_overview(args.csv)

    # Setup logger (uses rich terminal UI)
    # Create a minimal rollout config for logger initialization
    rollout_config = RolloutConfig(
        system_prompt="",
        mode="teacher-consistency",
        continue_msg="",
        final_msg=""
    )
    logger = setup_rich_logger(rollout_config)

    # Sampling args
    sampling_args = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    logger.info("pipeline_start", extra={
        "n_questions": len(questions),
        "n_consistency": args.n_consistency,
        "model": args.model,
        "output_dir": str(output_dir),
    })

    # Run batch triangulation
    results = batch_triangulate(
        csv_path=args.csv,
        questions=questions,
        n_consistency=args.n_consistency,
        model=args.model,
        dataset_description=DEFAULT_DATASET_DESCRIPTION,
        data_overview=data_overview,
        max_turns=args.max_turns,
        sampling_args=sampling_args,
        logger=logger,
    )

    # Convert to Episodes and save
    episodes_saved = 0
    episodes_verified = 0

    for q_dict, gold_trace, consistency_traces, verified in results:
        # Create Episode object
        episode = Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            question=q_dict["question"],
            hint=q_dict.get("hint"),
            teacher_trace=gold_trace,
            consistency_traces=consistency_traces,
            verified=verified,
            difficulty=q_dict.get("difficulty"),
        )

        # Save if verified OR if --verified-only is not set
        if verified or not args.verified_only:
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
