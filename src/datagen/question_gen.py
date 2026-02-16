"""
LLM-based question generator for CSV datasets.

This script uses an LLM to:
1. Explore a dataset using a sandboxed Python environment
2. Document exploration observations
3. Generate questions with varying difficulty levels (EASY, MEDIUM, HARD, VERY_HARD)

Configuration is managed via src.core.config.
"""

import asyncio
import json
import re
import sys
import argparse

from pathlib import Path
from datetime import datetime

from src.core.model import APILLM
from src.core.conversation import ConversationHistory, CodeCellResult
from csv_spec import ExplorationTurn, ExplorationTrace
from src.core.prompts import EXPLORATION_SYSTEM_PROMPT, get_exploration_continue_msg
from src.utils.parsing import parse_execution_result, extract_python_cells

from src.datagen.pipeline_ui import QuestionGenUI
from src.core.config import config
from src.datagen.shared.dataset_meta import (
    load_dataset_meta,
    generate_description_from_overview,
)

from src.envs.csv_env import LocalCSVAnalysisEnv
from src.utils.docker import generate_session_id


def get_datasets_with_episodes() -> set[str]:
    """
    Load csv_source values from existing episodes JSONL.

    Returns set of csv paths that have episodes generated.
    Used to prevent accidental question regeneration.
    """
    episodes_path = Path(config.episodes_llm_jsonl)
    if not episodes_path.exists():
        return set()

    datasets = set()
    with open(episodes_path) as f:
        for line in f:
            try:
                ep = json.loads(line)
                csv_source = ep.get("csv_source", "")
                if csv_source:
                    datasets.add(csv_source)
            except json.JSONDecodeError:
                continue
    return datasets


# Rebuild Pydantic model to resolve forward reference to CodeCellResult
ExplorationTurn.model_rebuild()


def build_execution_feedback(results: list[CodeCellResult]) -> str:
    """Build feedback message from execution results."""
    if not results:
        return "No code blocks found. Write Python code in ```python blocks."

    feedback_parts = []
    for i, result in enumerate(results, 1):
        if result.success:
            feedback_parts.append(f"✓ Cell {i} executed successfully")
            if result.stdout.strip():
                feedback_parts.append(f"Output:\n{result.stdout}")
        else:
            feedback_parts.append(f"✗ Cell {i} failed")
            feedback_parts.append(f"Error:\n{result.stderr}")

    return "\n\n".join(feedback_parts)


# Create global UI instance
ui = QuestionGenUI()


def try_parse_questions(response: str) -> list[dict] | None:
    """
    Parse questions from response. Tries multiple formats for robustness.
    Output is always normalized to the same structure regardless of input format.
    """

    def validate_questions(questions: list) -> bool:
        """Check if questions list has valid structure."""

        def _has_required_shape(q: dict) -> bool:
            question_text = q.get("question_text") or q.get("question")
            return (
                isinstance(question_text, str)
                and question_text.strip() != ""
                and "hint" in q
                and "n_steps" in q
                and "difficulty" in q
            )

        return all(isinstance(q, dict) and _has_required_shape(q) for q in questions)

    def normalize_questions(questions: list[dict]) -> list[dict]:
        """Normalize parser output to unified question_text-based shape."""
        normalized: list[dict] = []
        for q in questions:
            question_text = (q.get("question_text") or q.get("question") or "").strip()
            item = {
                "question_text": question_text,
                "hint": q.get("hint"),
                "n_steps": q.get("n_steps"),
                "difficulty": q.get("difficulty"),
            }
            if "id" in q:
                item["id"] = q.get("id")
            normalized.append(item)
        return normalized

    # Strategy 1: Look for ```json fenced blocks (preferred format)
    json_pattern = r"```json\s*\n(.*?)```"
    matches = re.findall(json_pattern, response, re.DOTALL)
    if matches:
        try:
            data = json.loads(matches[0])
            if "questions" in data and isinstance(data["questions"], list):
                if validate_questions(data["questions"]):
                    return normalize_questions(data["questions"])
        except json.JSONDecodeError:
            pass  # Strategy failed, try next

    # Strategy 2: Look for bare JSON object
    json_obj_pattern = r'\{\s*"questions"\s*:\s*\[.*?\]\s*\}'
    matches = re.findall(json_obj_pattern, response, re.DOTALL)
    if matches:
        try:
            data = json.loads(matches[0])
            if "questions" in data and isinstance(data["questions"], list):
                if validate_questions(data["questions"]):
                    return normalize_questions(data["questions"])
        except json.JSONDecodeError:
            pass  # Strategy failed, try next

    # Strategy 3: Look for Python dict assignment
    python_dict_pattern = r"(?:questions|output)\s*=\s*(\{.*?\})"
    matches = re.findall(python_dict_pattern, response, re.DOTALL)
    if matches:
        try:
            data = json.loads(matches[0])
            if "questions" in data and isinstance(data["questions"], list):
                if validate_questions(data["questions"]):
                    return normalize_questions(data["questions"])
        except json.JSONDecodeError:
            pass  # Strategy failed, return None below

    return None


async def force_question_generation(
    llm: APILLM, conversation: ConversationHistory, num_questions: int = 30
) -> list[dict]:
    """
    If model hasn't generated questions by max_turns, force it with a direct prompt.

    Returns:
        List of question dicts
    """
    conversation.add_user_feedback(
        f"You've explored enough. Now generate the {num_questions} questions in JSON format as specified in the system prompt."
    )

    messages = conversation.to_openai_messages()
    response = await llm(messages)

    questions = try_parse_questions(response)

    if not questions:
        raise RuntimeError(
            "Model failed to generate valid questions even after forcing. Check model output."
        )

    return questions


async def explore_and_generate_questions(
    csv_path: str,
    model: str,
    max_turns: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 6000,
    output_dir: str = ".",
    dataset_description: str = "",
    session_id: str | None = None,
) -> tuple[list[dict], ExplorationTrace]:
    """
    LLM explores dataset and generates questions.

    Args:
        csv_path: Path to CSV file
        model: Model identifier for APILLM
        max_turns: Max exploration turns before forcing question generation
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        output_dir: Where to save outputs

    Returns:
        (questions, exploration_trace)
    """
    ui.print_header(f"Question Generation Starting exploration of {csv_path}")
    ui.print_info("Model", model)
    ui.print_info("Max turns", str(max_turns))
    ui.print_empty_line()

    # 1. Setup
    env = LocalCSVAnalysisEnv(csv_path=csv_path, session_id=session_id)
    state = {}
    state = await env.setup_state(state)

    ui.print_status("CSVAnalysisEnv initialized with sandbox")

    llm = APILLM(
        model=model,
        sampling_args={"temperature": temperature, "max_tokens": max_tokens},
    )

    # Get pipeline parameters from config
    num_questions = config.num_questions_to_generate
    min_exploration_turns = config.min_exploration_turns

    conversation = ConversationHistory(
        system_prompt=EXPLORATION_SYSTEM_PROMPT.format(
            dataset_description=dataset_description,
            num_questions=num_questions,
        ),
        max_messages=100,
        max_context_tokens=config.max_context_tokens,
    )

    # 2. Multi-turn exploration loop
    exploration_turns = []
    questions_generated = None

    try:
        for turn_num in range(max_turns):
            ui.print_turn_header(turn_num, max_turns)

            messages = conversation.to_openai_messages()
            ui.print_status("Generating LLM response...")
            response = await llm(messages)

            ui.print_llm_response(response)

            code_cells = extract_python_cells(response)
            ui.print_code_blocks_found(len(code_cells))

            # Execute code
            execution_results = []
            for i, code in enumerate(code_cells, 1):
                ui.print_code_cell(i, code)

                output = await env.python(
                    code=code,
                    sandbox_id=state["sandbox_id"],
                    python_state=state["python_state"],
                )

                # result is now CodeCellResult object
                result = parse_execution_result(output)
                result.code = code
                # If success, use stdout, else keep output in stderr?
                # Actually parse_execution_result handles this splitting already.
                # Just ensure code is set.

                execution_results.append(result)

                if result.success:
                    ui.print_execution_success(result.stdout)
                else:
                    ui.print_execution_failure(result.stderr or output)

            # Save turn
            turn = ExplorationTurn(
                turn_number=turn_num,
                reasoning=response,
                code_cells=code_cells,
                execution_results=execution_results,
                timestamp=datetime.now(),
            )
            exploration_turns.append(turn)

            # Check for completion
            if "<DONE>" in response or "</DONE>" in response:
                if turn_num < min_exploration_turns:
                    ui.print_warning(
                        f"Model tried to finish too early (turn {turn_num + 1}/{min_exploration_turns} minimum)"
                    )
                    ui.print_status(
                        "Rejecting early completion - continuing exploration"
                    )
                else:
                    ui.print_success("Model signaled completion with <DONE>")
                    questions_generated = try_parse_questions(response)
                    if questions_generated:
                        ui.print_success(
                            f"Successfully extracted {len(questions_generated)} questions!"
                        )
                        break
                    else:
                        ui.print_error(
                            "Found <DONE> but couldn't parse questions from response"
                        )

            # Build feedback
            feedback = build_execution_feedback(execution_results)
            feedback += get_exploration_continue_msg(
                turn_num, min_exploration_turns, num_questions
            )

            conversation.add_assistant_response(response)
            conversation.add_user_feedback(feedback)

    finally:
        try:
            await env.destroy_sandbox(state["sandbox_id"])
        except Exception as e:
            ui.print_warning(f"Cleanup warning: {e}")

    # 3. Validate we got questions
    if not questions_generated:
        ui.print_warning("Model didn't generate questions. Forcing...")
        questions_generated = await force_question_generation(
            llm, conversation, num_questions
        )

    # 4. Create trace
    trace_questions = []
    if questions_generated:
        for q in questions_generated:
            trace_questions.append(
                {
                    "question_text": q.get("question_text"),
                    "hint": q.get("hint"),
                    "difficulty": q.get("difficulty"),
                    "n_steps": q.get("n_steps"),
                }
            )

    trace = ExplorationTrace(
        csv_path=csv_path,
        turns=exploration_turns,
        questions_generated=trace_questions,
        total_turns=len(exploration_turns),
        timestamp=datetime.now(),
    )

    # 5. Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Normalize to unified schema
    dataset_name, _ = load_dataset_meta(csv_path)
    question_records = []
    for idx, q in enumerate(questions_generated, start=1):
        question_text = q.get("question_text")
        record = {
            "id": q.get("id") or f"llm_{dataset_name}_{idx:04d}",
            "source": "llm_gen",
            "dataset": dataset_name,
            "question_text": question_text,
            "question_mechanical": None,
            "hint": q.get("hint"),
            "code": None,
            "code_hash": None,
            "ground_truth": None,
            "ground_truth_hash": None,
            "output_schema": None,
            "difficulty": q.get("difficulty"),
            "n_steps": q.get("n_steps"),
            "dataset_description": dataset_description,
        }
        question_records.append(record)

    questions_file = output_path / "questions.json"
    with open(questions_file, "w") as f:
        json.dump(question_records, f, indent=2)
    ui.print_saved_file(questions_file)

    trace_file = output_path / "exploration_trace.json"
    with open(trace_file, "w") as f:
        json.dump(trace.model_dump(), f, indent=2, default=str)
    ui.print_saved_file(trace_file)

    return question_records, trace


async def process_single_dataset(
    csv_path: Path,
    model: str,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    output_dir: Path,
    dataset_name: str,
    dataset_description: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
    session_id: str | None = None,
) -> tuple[str, bool, int | None]:
    """
    Process a single dataset with semaphore-controlled concurrency.

    Returns (dataset_name, success, num_questions)
    """
    async with semaphore:
        ui.print_section(f"[{index}/{total}] Starting: {dataset_name}")

        try:
            questions, trace = await explore_and_generate_questions(
                csv_path=str(csv_path),
                model=model,
                max_turns=max_turns,
                temperature=temperature,
                max_tokens=max_tokens,
                output_dir=str(output_dir),
                dataset_description=dataset_description,
                session_id=session_id,
            )

            ui.print_success(
                f"✓ [{index}/{total}] {dataset_name}: {len(questions)} questions"
            )
            return (dataset_name, True, len(questions))

        except Exception as e:
            ui.print_error(f"✗ [{index}/{total}] {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            return (dataset_name, False, None)


async def run_parallel_generation(
    csv_sources: list[str],
    max_concurrent: int = 10,
    regenerate: bool = False,
) -> tuple[int, int, int]:
    """
    Run question generation for all CSVs with parallel execution.

    Returns (success_count, failure_count, skipped_count)
    """
    # Generate session ID for container isolation
    session_id = generate_session_id()
    ui.print_status(f"Session ID: {session_id}")

    # Common config
    temperature = config.sampling_args.temperature
    max_tokens = config.sampling_args.max_tokens
    model = config.question_gen_model
    max_turns = config.question_gen_max_turns
    base_output_dir = Path(config.questions_llm_dir)

    # Check for existing episodes (lock mechanism)
    datasets_with_episodes = get_datasets_with_episodes() if not regenerate else set()
    skipped_count = 0

    # Prepare tasks
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []

    for i, csv_path_str in enumerate(csv_sources, 1):
        csv_path = Path(csv_path_str)

        # Load dataset metadata using shared module
        dataset_name, dataset_description = load_dataset_meta(csv_path)

        # Generate description from data_overview if missing
        if not dataset_description or not dataset_description.strip():
            from src.core.prompts import generate_data_overview

            data_overview = generate_data_overview(str(csv_path))
            dataset_description = generate_description_from_overview(data_overview)
            ui.print_warning(
                f"{dataset_name}: No description found, synthesized from data_overview"
            )

        # Skip if episodes already exist (unless --regenerate)
        if csv_path_str in datasets_with_episodes:
            ui.print_warning(
                f"Skipping {dataset_name}: episodes already exist (use --regenerate to override)"
            )
            skipped_count += 1
            continue

        # Output directory (per-dataset subfolder)
        output_dir = base_output_dir / dataset_name

        tasks.append(
            process_single_dataset(
                csv_path=csv_path,
                model=model,
                max_turns=max_turns,
                temperature=temperature,
                max_tokens=max_tokens,
                output_dir=output_dir,
                dataset_name=dataset_name,
                dataset_description=dataset_description,
                semaphore=semaphore,
                index=i,
                total=len(csv_sources),
                session_id=session_id,
            )
        )

    if not tasks:
        return 0, 0, skipped_count

    # Run all tasks concurrently (semaphore limits actual parallelism)
    ui.print_header(
        f"Processing {len(tasks)} datasets with {max_concurrent} concurrent workers"
    )
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for _, success, _ in results if success)
    failure_count = sum(1 for _, success, _ in results if not success)

    return success_count, failure_count, skipped_count


def main(max_datasets: int | None = None, regenerate: bool = False):
    csv_sources = config.csv_sources
    if isinstance(csv_sources, str):
        csv_sources = [csv_sources]

    if not csv_sources:
        ui.print_error("No CSV sources found in config (csv or csv_sources)")
        return 2  # Total failure

    # Limit datasets for testing
    if max_datasets and len(csv_sources) > max_datasets:
        csv_sources = csv_sources[:max_datasets]

    max_concurrent = config.max_concurrent_containers
    ui.print_info("Datasets", str(len(csv_sources)))
    ui.print_info("Max concurrent", str(max_concurrent))
    if regenerate:
        ui.print_warning("Regenerate mode: will overwrite existing questions")
    ui.print_empty_line()

    success_count, failure_count, skipped_count = asyncio.run(
        run_parallel_generation(csv_sources, max_concurrent, regenerate=regenerate)
    )

    # Final summary
    ui.print_summary_header()
    summary_parts = [f"{success_count} success", f"{failure_count} failed"]
    if skipped_count > 0:
        summary_parts.append(f"{skipped_count} skipped (episodes exist)")
    ui.print_status(f"Processed {len(csv_sources)} sources: {', '.join(summary_parts)}")

    # Exit codes: 0=success, 1=partial success (some data), 2=total failure
    if success_count == 0 and skipped_count == 0:
        return 2
    elif failure_count > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-based question generator for CSV datasets."
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Limit number of datasets to process (for testing)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate questions even if episodes already exist for the dataset",
    )
    args = parser.parse_args()

    try:
        sys.exit(main(max_datasets=args.max_datasets, regenerate=args.regenerate))
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        sys.exit(0)
