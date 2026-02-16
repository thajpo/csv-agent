"""Test pass rates for procedural questions with and without verbalization.

This script:
1. Generates procedural questions
2. Tests solving with verbalization (question_text + hint)
3. Tests solving without verbalization (mechanical description only)
4. Reports pass rates for both conditions
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Literal

from src.datagen.synthetic.programs.program_generator import run_pipeline
from src.datagen.synthetic.profiler import DataProfiler
from src.datagen.teacher import execute_teacher_trace
from src.datagen.pipeline_ui import EpisodeGenUI
from src.core.config import config


async def solve_question(
    question: dict,
    csv_path: str,
    condition: Literal["with_verbalization", "without_verbalization"],
) -> dict:
    """Attempt to solve a question under given condition.

    Args:
        question: The question dict
        csv_path: Path to CSV
        condition: Which verbalization to use

    Returns:
        Result dict with success, answer, error
    """
    # Prepare question text based on condition
    if condition == "with_verbalization":
        question_text = question.get("question_text", "")
        hint = question.get("hint", "")
    else:
        # Use mechanical description only
        question_text = question.get("question_mechanical", "")
        hint = ""

    try:
        # Create UI instance (required parameter)
        ui = EpisodeGenUI()

        # Execute teacher trace with the question
        trace, _conversation, _system_prompt, _elapsed = await execute_teacher_trace(
            csv_path=csv_path,
            question=question_text,
            model=config.teacher_model,
            hint=hint,
            ui=ui,
        )

        # Check if we got an answer
        success = trace.get("success", False)
        answer = trace.get("final_answer")

        # Compare to ground truth
        ground_truth = question.get("ground_truth")
        matches = False
        if success and answer is not None and ground_truth is not None:
            # Simple comparison (could be enhanced with answers_match)
            if isinstance(answer, dict) and isinstance(ground_truth, dict):
                matches = answer == ground_truth
            elif isinstance(answer, (int, float)) and isinstance(
                ground_truth, (int, float)
            ):
                matches = abs(answer - ground_truth) < 0.1

        return {
            "success": success,
            "answer": answer,
            "ground_truth": ground_truth,
            "matches": matches,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "answer": None,
            "ground_truth": question.get("ground_truth"),
            "matches": False,
            "error": str(e),
        }


async def test_dataset(csv_path: str, dataset_name: str, n_questions: int = 10):
    """Test pass rates on a dataset."""
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"File: {csv_path}")
    print("=" * 80)

    # Generate questions
    with tempfile.TemporaryDirectory() as tmpdir:
        questions = await run_pipeline(
            csv_path=csv_path,
            max_programs=n_questions,
            max_verbalize=n_questions,
            skip_verbalization=False,
            output_dir=tmpdir,
        )

        print(f"Generated {len(questions)} questions")

        if not questions:
            print("No questions generated!")
            return None

        # Test with verbalization
        print("\nTesting WITH verbalization...")
        with_verbalization = []
        for i, q in enumerate(questions):
            print(f"  {i + 1}/{len(questions)}", end="\r")
            result = await solve_question(q, csv_path, "with_verbalization")
            with_verbalization.append(result)
        print()

        # Test without verbalization
        print("\nTesting WITHOUT verbalization...")
        without_verbalization = []
        for i, q in enumerate(questions):
            print(f"  {i + 1}/{len(questions)}", end="\r")
            result = await solve_question(q, csv_path, "without_verbalization")
            without_verbalization.append(result)
        print()

        # Calculate pass rates
        def calc_stats(results):
            total = len(results)
            success = sum(1 for r in results if r["success"])
            matches = sum(1 for r in results if r["matches"])
            return {
                "total": total,
                "execution_success": success,
                "answer_matches": matches,
                "execution_rate": success / total if total > 0 else 0,
                "match_rate": matches / total if total > 0 else 0,
            }

        stats_with = calc_stats(with_verbalization)
        stats_without = calc_stats(without_verbalization)

        print(f"\nResults:")
        print(
            f"  With verbalization:    {stats_with['answer_matches']}/{stats_with['total']} correct ({stats_with['match_rate']:.1%})"
        )
        print(
            f"  Without verbalization: {stats_without['answer_matches']}/{stats_without['total']} correct ({stats_without['match_rate']:.1%})"
        )

        improvement = stats_with["match_rate"] - stats_without["match_rate"]
        print(f"  Improvement: {improvement:+.1%}")

        return {
            "dataset": dataset_name,
            "total_questions": len(questions),
            "with_verbalization": stats_with,
            "without_verbalization": stats_without,
            "improvement": improvement,
            "examples": [
                {
                    "question_text": q.get("question_text", "")[:100],
                    "mechanical": q.get("question_mechanical", "")[:100],
                    "with_verbalization_result": with_verbalization[i]["matches"],
                    "without_verbalization_result": without_verbalization[i]["matches"],
                }
                for i, q in enumerate(questions[:3])
            ],
        }


async def main():
    """Run pass rate tests on multiple datasets."""
    datasets = [
        ("data/csv/data.csv", "Base Dataset"),
        ("data/kaggle/fedesoriano_heart-failure-prediction/data.csv", "Heart Failure"),
        ("data/kaggle/gregorut_videogamesales/data.csv", "Video Game Sales"),
    ]

    all_results = []

    for csv_path, name in datasets:
        try:
            result = await test_dataset(csv_path, name, n_questions=8)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error on {name}: {e}")
            import traceback

            traceback.print_exc()

    # Save report
    report_path = "pass_rate_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'=' * 80}")
    print(f"Report saved to: {report_path}")
    print("=" * 80)

    # Summary
    print("\nOverall Summary:")
    for r in all_results:
        print(f"\n{r['dataset']}:")
        print(f"  With verbalization:    {r['with_verbalization']['match_rate']:.1%}")
        print(
            f"  Without verbalization: {r['without_verbalization']['match_rate']:.1%}"
        )
        print(f"  Improvement: {r['improvement']:+.1%}")


if __name__ == "__main__":
    asyncio.run(main())
