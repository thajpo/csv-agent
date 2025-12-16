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
from typing import Any

from src.datagen.teacher import batch_triangulate
from src.datagen.ui import EpisodeGenUI
from src.core.prompts import generate_data_overview, DEFAULT_DATASET_DESCRIPTION
from src.core.types import Episode, EpisodeJSONL, Question, ExecutionTrace


# Create global UI instance
ui = EpisodeGenUI()


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
    float_tol = config.get("float_tolerance", 0.1)

    # Load questions from question_gen.py output
    questions_file = config.get("questions_json", "question/questions.json")
    questions = load_questions(questions_file)

    # Output as single JSONL file
    output_jsonl = Path(config.get("episodes_jsonl", "episodes/episodes.jsonl"))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)



    # Display pipeline header
    ui.print_pipeline_header(
        n_questions=len(questions),
        n_consistency=n_consistency,
        csv_path=csv_path,
        model=teacher_model,
        float_tol=float_tol,
        output_file=str(output_jsonl)
    )

    # Generate data overview
    data_overview = generate_data_overview(csv_path)

    # Sampling args
    sampling_args = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Run batch triangulation with UI
    results = batch_triangulate(
        csv_path=csv_path,
        questions=questions,
        model=teacher_model,  # Required positional arg (3rd)
        n_consistency=n_consistency,
        dataset_description=DEFAULT_DATASET_DESCRIPTION,
        data_overview=data_overview,
        max_turns=max_turns,
        sampling_args=sampling_args,

        ui=ui,
        float_tol=float_tol,
    )

    # Convert to JSONL episodes and save
    episodes_jsonl = []
    episodes_verified = 0

    for q_dict, gold_trace, gold_conversation, system_prompt, consistency_results, verified in results:
        # Create Question object (question, metadata)
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

    # Write JSONL file (one episode per line)
    with open(output_jsonl, 'w') as f:
        for ep in episodes_jsonl:
            f.write(json.dumps(ep.model_dump(), default=str) + '\n')

    # Display final summary
    ui.base.print_section("PIPELINE COMPLETE")
    ui.base.print_key_value("Output file", str(output_jsonl))
    ui.base.print_key_value("Total questions", len(questions))
    ui.base.print_key_value("Episodes saved", len(episodes_jsonl))
    ui.base.print_key_value("Episodes verified", episodes_verified)
    verification_rate = episodes_verified / len(questions) * 100 if questions else 0.0
    ui.base.print_key_value("Verification rate", f"{verification_rate:.1f}%")

    if verification_rate == 100:
        ui.base.print_success("All episodes verified!")
    elif verification_rate >= 80:
        ui.base.print_success(f"High verification rate: {verification_rate:.1f}%")
    elif verification_rate >= 50:
        ui.base.print_warning(f"Moderate verification rate: {verification_rate:.1f}%")
    else:
        ui.base.print_error(f"Low verification rate: {verification_rate:.1f}%")

    ui.base.print_empty_line()

    return 0


if __name__ == "__main__":
    sys.exit(main())
