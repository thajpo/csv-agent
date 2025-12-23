"""
Episode generation pipeline.

This script:
1. Loads questions from CSV (TODO: implement)
2. Runs teacher triangulation on each question
3. Saves verified episodes to disk

Usage:
    python -m src.datagen.episode_gen
"""
import asyncio
import json
import sys
import signal
import argparse
from pathlib import Path
from datetime import datetime
import uuid
from typing import Any

from src.datagen.teacher import batch_triangulate
from src.datagen.ui import EpisodeGenUI
from src.core.prompts import generate_data_overview
from src.core.types import Episode, EpisodeJSONL, Question, ExecutionTrace
from src.core.config import config
from src.utils.docker import cleanup_csv_sandbox_containers

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🛑 Interrupted! Cleaning up containers...")
    cleanup_csv_sandbox_containers()
    print("✓ Cleanup complete")
    sys.exit(0)

def load_questions(questions_path: str) -> list[dict]:
    """Load questions from JSON file."""
    with open(questions_path) as f:
        return json.load(f)

async def main(legacy_mode: bool = False):
    # Create global UI instance
    ui = EpisodeGenUI()
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # config is already imported from src.core.config

    teacher_model = config.teacher_model
    n_consistency = config.n_consistency
    max_turns = config.max_turns
    float_tol = config.float_tolerance
    verified_only = config.verified_only
    temperature = config.sampling_args.temperature
    max_tokens = config.sampling_args.max_tokens

    # Handle single csv or list of csvs
    csv_sources = config.csv_sources
    if isinstance(csv_sources, str):
        csv_sources = [csv_sources]

    # Output as single JSONL file
    output_jsonl = Path(config.episodes_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    if output_jsonl.exists():
        output_jsonl.unlink()
        
    # Get parent directory of questions
    base_questions_dir = Path(config.questions_json).parent

    total_episodes_saved = 0
    total_verified = 0

    for i, csv_path in enumerate(csv_sources, 1):
        dataset_name = Path(csv_path).stem
        ui.base.print_section(f"Processing CSV {i}/{len(csv_sources)}: {csv_path}")

        # Determine dataset description (Sidecar Metadata only)
        dataset_description = None
        
        # Look for sidecar metadata: slug.meta.json or csv_filename.meta.json
        meta_path = Path(csv_path).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta_data = json.load(f)
                    dataset_description = meta_data.get("description") or meta_data.get("subtitle")
                    if dataset_description:
                        ui.base.print_status(f"Loaded description from sidecar metadata: {meta_path.name}")
            except Exception as e:
                ui.base.print_warning(f"Failed to read metadata from {meta_path}: {e}")

        if not dataset_description or not dataset_description.strip():
            ui.base.print_error(f"ERROR: No description found for {dataset_name}")
            ui.base.print_info("Hint", f"Create {dataset_name}.meta.json with a 'description' field.")
            continue

        # Locate questions (Modern structure: question/[dataset_name]/questions.json)
        questions_file = base_questions_dir / dataset_name / "questions.json"
        
        # Legacy fallback only if --legacy flag is provided
        if not questions_file.exists() and legacy_mode and len(csv_sources) == 1:
            legacy_path = Path(config.questions_json)
            if legacy_path.exists():
                questions_file = legacy_path
                ui.base.print_status(f"Using legacy flat file: {questions_file}")

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
            dataset_description=dataset_description,
            data_overview=data_overview,
            max_turns=max_turns,
            sampling_args=sampling_args,

            ui=ui,
            float_tol=float_tol,
        )

        # Convert to JSONL episodes and save incrementally (one at a time)
        # This ensures progress is saved even if pipeline crashes mid-batch
        batch_verified = 0
        batch_saved = 0

        for q_dict, gold_trace, gold_conversation, system_prompt, consistency_results, verified, timing_metadata in results:
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
                csv_source=csv_path,
                timing_metadata=timing_metadata,
            )

            # Save immediately if verified OR verified_only is False (incremental saving)
            if verified or not verified_only:
                mode = 'a' if output_jsonl.exists() else 'w'
                with open(output_jsonl, mode) as f:
                    f.write(json.dumps(episode_jsonl.model_dump(), default=str) + '\n')
                batch_saved += 1
                if verified:
                    batch_verified += 1

        total_episodes_saved += batch_saved
        total_verified += batch_verified

        ui.base.print_success(f"✓ Saved {batch_saved} episodes for {dataset_name} ({batch_verified} verified)")


    # Display final summary
    ui.base.print_section("PIPELINE COMPLETE")
    ui.base.print_key_value("Output file", str(output_jsonl))
    ui.base.print_key_value("Total sources", len(csv_sources))
    ui.base.print_key_value("Total episodes saved", total_episodes_saved)
    ui.base.print_key_value("Total verified", total_verified)
    
    ui.base.print_empty_line()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Episode generation pipeline.")
    parser.add_argument("--legacy", action="store_true", help="Allow fallback to legacy flat questions file")
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(main(legacy_mode=args.legacy)))
    except KeyboardInterrupt:
        # Already handled in signal_handler, but just in case
        sys.exit(0)
