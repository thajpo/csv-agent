"""
Episode generation pipeline.

This script:
1. Loads questions from CSV (TODO: implement)
2. Runs teacher triangulation on each question
3. Saves verified episodes to disk

Usage:
    python -m src.authoring.episode_gen
"""

import asyncio
import json
import sys
import signal
from pathlib import Path
from datetime import datetime
import uuid

from typing import Any

from src.datagen.teacher import batch_triangulate
from src.datagen.ui import EpisodeGenUI
from src.core.prompts import generate_data_overview, DEFAULT_DATASET_DESCRIPTION
from src.core.types import Episode, EpisodeJSONL, Question, ExecutionTrace
from src.core.config import load_config


from src.utils.docker import cleanup_csv_sandbox_containers


def cleanup_containers():
    """Emergency cleanup of all CSV sandbox containers."""
    cleanup_csv_sandbox_containers()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🛑 Interrupted! Cleaning up containers...")
    cleanup_containers()
    print("✓ Cleanup complete")
    sys.exit(0)


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





async def main():
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load config
    config = load_config()

    # Extract config values (fail-fast on missing keys)
    teacher_model = config["teacher_model"]  # Required
    n_consistency = config.get("n_consistency", 5)
    max_turns = config.get("max_turns", 10)
    float_tol = config.get("float_tolerance", 0.1)
    verified_only = config.get("verified_only", False)
    
    # Sampling args
    sampling_config = config.get("sampling_args", {})
    temperature = sampling_config.get("temperature", 0.7)
    max_tokens = sampling_config.get("max_tokens", 6000)

    # Handle single csv (legacy) or csv_sources (new)
    csv_sources = config.get("csv_sources", config.get("csv", []))
    if isinstance(csv_sources, str):
        csv_sources = [csv_sources]

    # Output as single JSONL file (append mode supported by logic below)
    output_jsonl = Path(config.get("episodes_jsonl", "episodes/episodes.jsonl"))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output file initially if we want to overwrite (optional, but safer to start fresh if running full pipeline)
    # But if users want to append, we should be careful. 
    # Let's overwrite start, then append.
    if output_jsonl.exists():
        output_jsonl.unlink()
        
    # Base questions dir
    base_questions_dir = Path(config.get("questions_json", "question/questions.json")).parent

    total_episodes_saved = 0
    total_verified = 0

    for i, csv_path in enumerate(csv_sources, 1):
        dataset_name = Path(csv_path).stem
        ui.base.print_section(f"Processing CSV {i}/{len(csv_sources)}: {csv_path}")

        # Locate questions
        questions_file = base_questions_dir / dataset_name / "questions.json"
        
        # Legacy fallback for single CSV config
        if not questions_file.exists() and len(csv_sources) == 1:
            legacy_path = Path(config.get("questions_json", "question/questions.json"))
            if legacy_path.exists():
                questions_file = legacy_path

        if not questions_file.exists():
            ui.base.print_warning(f"Skipping {dataset_name}: No questions found at {questions_file}")
            continue

        questions = load_questions(str(questions_file))
        ui.base.print_status(f"Loaded {len(questions)} questions")

        # Display pipeline header for this CSV
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
        results = await batch_triangulate(
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

        # Convert to JSONL episodes and save (Append to global file)
        episodes_jsonl = []
        batch_verified = 0

        for q_dict, gold_trace, gold_conversation, system_prompt, consistency_results, verified in results:
            # Create Question object with auto-generated ID
            question_obj = Question.from_dict(q_dict)

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
                    batch_verified += 1

        # Write JSONL file (Append)
        mode = 'a' if output_jsonl.exists() else 'w'
        with open(output_jsonl, mode) as f:
            for ep in episodes_jsonl:
                f.write(json.dumps(ep.model_dump(), default=str) + '\n')
                
        total_episodes_saved += len(episodes_jsonl)
        total_verified += batch_verified
        
        ui.base.print_success(f"✓ Saved {len(episodes_jsonl)} episodes for {dataset_name} ({batch_verified} verified)")


    # Display final summary
    ui.base.print_section("PIPELINE COMPLETE")
    ui.base.print_key_value("Output file", str(output_jsonl))
    ui.base.print_key_value("Total sources", len(csv_sources))
    ui.base.print_key_value("Total episodes saved", total_episodes_saved)
    ui.base.print_key_value("Total verified", total_verified)
    
    ui.base.print_empty_line()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
