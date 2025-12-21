"""
LLM-based question generator for CSV datasets.

This script uses an LLM to:
1. Explore a dataset using a sandboxed Python environment
2. Document exploration observations
3. Generate questions with varying difficulty levels (EASY, MEDIUM, HARD, VERY_HARD)

Configuration is loaded from config.yaml.
"""
import asyncio
import json
import re
import sys

from pathlib import Path
from datetime import datetime

from src.datagen.ui import QuestionGenUI

from src.envs.csv_env import LocalCSVAnalysisEnv
from src.core.model import APILLM
from src.core.conversation import ConversationHistory, CodeCellResult
from src.core.types import ExplorationTurn, ExplorationTrace
from src.core.prompts import EXPLORATION_SYSTEM_PROMPT, MIN_EXPLORATION_TURNS, get_exploration_continue_msg
from src.utils.interaction import parse_execution_result, extract_python_cells
from src.core.config import load_config


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
        return all(
            isinstance(q, dict) and
            all(key in q for key in ["question", "hint", "n_steps", "difficulty"])
            for q in questions
        )

    # Strategy 1: Look for ```json fenced blocks (preferred format)
    json_pattern = r'```json\s*\n(.*?)```'
    matches = re.findall(json_pattern, response, re.DOTALL)
    if matches:
        try:
            data = json.loads(matches[0])
            if "questions" in data and isinstance(data["questions"], list):
                if validate_questions(data["questions"]):
                    return data["questions"]
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
                    return data["questions"]
        except json.JSONDecodeError:
            pass  # Strategy failed, try next

    # Strategy 3: Look for Python dict assignment
    python_dict_pattern = r'(?:questions|output)\s*=\s*(\{.*?\})'
    matches = re.findall(python_dict_pattern, response, re.DOTALL)
    if matches:
        try:
            data = json.loads(matches[0])
            if "questions" in data and isinstance(data["questions"], list):
                if validate_questions(data["questions"]):
                    return data["questions"]
        except json.JSONDecodeError:
            pass  # Strategy failed, return None below

    return None


def force_question_generation(llm: APILLM, conversation: ConversationHistory) -> list[dict]:
    """
    If model hasn't generated questions by max_turns, force it with a direct prompt.

    Returns:
        List of question dicts
    """
    conversation.add_user_feedback(
        "You've explored enough. Now generate the 13 questions in JSON format as specified in the system prompt."
    )

    messages = conversation.to_openai_messages()
    response = asyncio.get_event_loop().run_until_complete(llm(messages))

    questions = try_parse_questions(response)

    if not questions:
        raise RuntimeError("Model failed to generate valid questions even after forcing. Check model output.")

    return questions


async def explore_and_generate_questions(
    csv_path: str,
    model: str,
    max_turns: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 6000,
    output_dir: str = "."
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
    env = LocalCSVAnalysisEnv(csv_path=csv_path)
    state = {}
    state = await env.setup_state(state)
    
    ui.print_status(f"CSVAnalysisEnv initialized with sandbox")
    
    llm = APILLM(model=model, sampling_args={"temperature": temperature, "max_tokens": max_tokens})
    conversation = ConversationHistory(
        system_prompt=EXPLORATION_SYSTEM_PROMPT,
        max_messages=100,
        max_context_tokens=100_000
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
                    sandbox_state=state["sandbox_state"],
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
                timestamp=datetime.now()
            )
            exploration_turns.append(turn)

            # Check for completion
            if "<DONE>" in response or "</DONE>" in response:
                if turn_num < MIN_EXPLORATION_TURNS:
                    ui.print_warning(f"Model tried to finish too early (turn {turn_num + 1}/{MIN_EXPLORATION_TURNS} minimum)")
                    ui.print_status("Rejecting early completion - continuing exploration")
                else:
                    ui.print_success("Model signaled completion with <DONE>")
                    questions_generated = try_parse_questions(response)
                    if questions_generated:
                        ui.print_success(f"Successfully extracted {len(questions_generated)} questions!")
                        break
                    else:
                        ui.print_error("Found <DONE> but couldn't parse questions from response")

            # Build feedback
            feedback = build_execution_feedback(execution_results)
            feedback += get_exploration_continue_msg(turn_num, MIN_EXPLORATION_TURNS)

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
        questions_generated = force_question_generation(llm, conversation)

    # 4. Create trace
    trace = ExplorationTrace(
        csv_path=csv_path,
        turns=exploration_turns,
        questions_generated=questions_generated,
        total_turns=len(exploration_turns),
        timestamp=datetime.now()
    )

    # 5. Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    questions_file = output_path / "questions.json"
    with open(questions_file, 'w') as f:
        json.dump(questions_generated, f, indent=2)
    ui.print_saved_file(questions_file)

    trace_file = output_path / "exploration_trace.json"
    with open(trace_file, 'w') as f:
        json.dump(trace.model_dump(), f, indent=2, default=str)
    ui.print_saved_file(trace_file)

    return questions_generated, trace


def main():
    try:
        config = load_config("config.yaml")
    except Exception as e:
        ui.print_error(f"Configuration error: {e}")
        return 1

    # Handle single csv (legacy) or csv_sources (new)
    csv_sources = config.csv_sources
    if isinstance(csv_sources, str):
        csv_sources = [csv_sources]
    
    if not csv_sources:
        ui.print_error("No CSV sources found in config (csv or csv_sources)")
        return 1

    # Common config (typed access)
    temperature = config.sampling_args.temperature
    max_tokens = config.sampling_args.max_tokens
    model = config.question_gen_model
    max_turns = config.question_gen_max_turns
    
    # Base output directory
    base_output_dir = Path(config.questions_json).parent

    success_count = 0
    failure_count = 0

    for i, csv_path in enumerate(csv_sources, 1):
        ui.print_section(f"Processing CSV {i}/{len(csv_sources)}: {csv_path}")
        
        # Derive output directory for this CSV (e.g. "data" from "csv/data.csv")
        dataset_name = Path(csv_path).stem
        output_dir = base_output_dir / dataset_name
        
        try:
            questions, trace = asyncio.run(explore_and_generate_questions(
                csv_path=csv_path,
                model=model,
                max_turns=max_turns,
                temperature=temperature,
                max_tokens=max_tokens,
                output_dir=str(output_dir)
            ))
            
            ui.print_success(f"✓ Generated {len(questions)} questions for {dataset_name}")
            success_count += 1
            
            # Print sample for this batch
            ui.print_sample_questions_header()
            for j, q in enumerate(questions[:3], 1):
                ui.print_question_panel(j, q)

        except Exception as e:
            ui.print_error(f"Failed to generate questions for {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            failure_count += 1
            
    # Final summary
    ui.print_summary_header()
    ui.print_status(f"Processed {len(csv_sources)} sources: {success_count} success, {failure_count} failed")
    
    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
