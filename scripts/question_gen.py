"""
LLM-based question generator for CSV datasets.

This script uses an LLM to:
1. Explore a dataset using Jupyter kernel
2. Document exploration observations
3. Generate questions with varying difficulty levels (EASY, MEDIUM, HARD, VERY_HARD)

Usage:
    python -m scripts.question_gen --csv data.csv --output questions.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

from src.kernel import JupyterKernel
from src.model import APILLM
from src.conversation import ConversationManager, Turn, CodeCellResult
from src.types import ExplorationTurn, ExplorationTrace
from src.prompts import EXPLORATION_SYSTEM_PROMPT, EXPLORATION_CONTINUE_MSG


def extract_python_cells(response: str) -> list[str]:
    """Extract ```python...``` code blocks from response."""
    pattern = r'```python\n(.*?)```'
    return re.findall(pattern, response, re.DOTALL)


def try_parse_questions(response: str) -> list[dict] | None:
    """
    Try to parse questions from model response.

    Looks for JSON block with structure:
    {
        "questions": [
            {"question": ..., "hint": ..., "n_steps": ..., "difficulty": ...},
            ...
        ]
    }

    Returns:
        List of question dicts if found and valid, None otherwise
    """
    # Try to find ```json...``` block
    json_pattern = r'```json\n(.*?)```'
    matches = re.findall(json_pattern, response, re.DOTALL)

    if not matches:
        return None

    try:
        data = json.loads(matches[0])
        if "questions" in data and isinstance(data["questions"], list):
            # Validate structure
            questions = data["questions"]
            for q in questions:
                if not all(key in q for key in ["question", "hint", "n_steps", "difficulty"]):
                    return None
            return questions
    except json.JSONDecodeError:
        return None

    return None


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


def force_question_generation(llm: APILLM, conversation: ConversationManager) -> list[dict]:
    """
    If model hasn't generated questions by max_turns, force it with a direct prompt.

    Returns:
        List of question dicts
    """
    # Add forcing message
    force_turn = Turn(
        turn_number=conversation.get_active_turn_count(),
        timestamp=datetime.now(),
        model_response="",
        truncated_response="",
        done_signal=False,
        feedback_message="You've explored enough. Now generate the 13 questions in JSON format as specified in the system prompt.",
        reasoning=None
    )
    conversation.add_turn(force_turn)

    # Get response
    messages = conversation.to_openai_messages()
    response = llm(messages)

    # Try to parse
    questions = try_parse_questions(response)

    if not questions:
        raise RuntimeError("Model failed to generate valid questions even after forcing. Check model output.")

    return questions


def explore_and_generate_questions(
    csv_path: str,
    model: str = "meta-llama/llama-3.2-3b-instruct:free",
    max_turns: int = 20,
    temperature: float = 0.7,
    max_tokens: int = 2000,
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
    print(f"\n[Question Generation] Starting exploration of {csv_path}")
    print(f"[Model] {model}")
    print(f"[Max turns] {max_turns}\n")

    # 1. Setup
    kernel = JupyterKernel(timeout=120, csv_path=csv_path)
    llm = APILLM(model=model, sampling_args={"temperature": temperature, "max_tokens": max_tokens})
    conversation = ConversationManager(
        system_prompt=EXPLORATION_SYSTEM_PROMPT,
        max_active_turns=50,  # Keep full exploration in context
        max_context_tokens=100_000
    )

    # 2. Multi-turn exploration loop
    exploration_turns = []
    questions_generated = None

    for turn_num in range(max_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn_num + 1}/{max_turns}")
        print(f"{'='*60}\n")

        # Get model response
        messages = conversation.to_openai_messages()
        print("[LLM] Generating response...")
        response = llm(messages)

        print(f"\n[Response Preview]\n{response[:500]}{'...' if len(response) > 500 else ''}\n")

        # Extract code cells
        code_cells = extract_python_cells(response)
        print(f"[Code Cells] Found {len(code_cells)} code block(s)")

        # Execute code
        execution_results = []
        for i, code in enumerate(code_cells, 1):
            print(f"\n[Executing Cell {i}]")
            print(f"```python\n{code}\n```")
            result = kernel.execute(code)
            execution_results.append(CodeCellResult(
                code=code,
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr if not result.success else ""
            ))

            if result.success:
                print(f"✓ Success")
                if result.stdout.strip():
                    print(f"Output:\n{result.stdout[:200]}{'...' if len(result.stdout) > 200 else ''}")
            else:
                print(f"✗ Failed: {result.error_message}")

        # Save turn
        turn = ExplorationTurn(
            turn_number=turn_num,
            reasoning=response,
            code_cells=code_cells,
            execution_results=execution_results,
            timestamp=datetime.now()
        )
        exploration_turns.append(turn)

        # Check if model generated questions
        questions_generated = try_parse_questions(response)
        if questions_generated:
            print(f"\n[Success] Model generated {len(questions_generated)} questions!")
            break

        # Build feedback
        feedback = build_execution_feedback(execution_results)
        feedback += EXPLORATION_CONTINUE_MSG

        # Add turn to conversation
        conversation_turn = Turn(
            turn_number=turn_num,
            timestamp=datetime.now(),
            model_response=response,
            truncated_response=response,
            code_cells=code_cells,
            execution_results=execution_results,
            done_signal=False,
            feedback_message=feedback,
            reasoning=None
        )
        conversation.add_turn(conversation_turn)

    # 3. Validate we got questions
    if not questions_generated:
        print("\n[Warning] Model didn't generate questions. Forcing...")
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

    # Save questions.json
    questions_file = output_path / "questions.json"
    with open(questions_file, 'w') as f:
        json.dump(questions_generated, f, indent=2)
    print(f"\n[Saved] Questions → {questions_file}")

    # Save exploration trace
    trace_file = output_path / "exploration_trace.json"
    with open(trace_file, 'w') as f:
        json.dump(trace.model_dump(), f, indent=2, default=str)
    print(f"[Saved] Exploration trace → {trace_file}")

    # Cleanup
    kernel.shutdown()

    return questions_generated, trace


def main():
    parser = argparse.ArgumentParser(description="Generate questions using LLM exploration")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV file")
    parser.add_argument("--output", "-o", default="questions.json", help="Output JSON file for questions")
    parser.add_argument("--exploration-output", default="exploration_trace.json", help="Output JSON file for exploration trace")
    parser.add_argument("--model", default="meta-llama/llama-3.2-3b-instruct:free", help="Model identifier")
    parser.add_argument("--max-turns", type=int, default=20, help="Max exploration turns")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens per response")

    args = parser.parse_args()

    # Set output directory from output file path
    output_dir = str(Path(args.output).parent)
    if output_dir == ".":
        output_dir = "."

    try:
        questions, trace = explore_and_generate_questions(
            csv_path=args.csv,
            model=args.model,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=output_dir
        )

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total questions: {len(questions)}")

        # Count by difficulty
        difficulty_counts = {}
        for q in questions:
            diff = q.get("difficulty", "UNKNOWN")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        for diff, count in sorted(difficulty_counts.items()):
            print(f"  {diff}: {count}")

        print(f"\nSample questions:")
        for i, q in enumerate(questions[:3], 1):
            print(f"\n{i}. [{q['difficulty']}] {q['question']}")
            print(f"   Steps: {q['n_steps']}")
            print(f"   Hint: {q['hint'][:80]}{'...' if len(q['hint']) > 80 else ''}")

        return 0

    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
