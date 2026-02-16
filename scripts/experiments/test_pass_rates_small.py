"""Quick pass rate test - smaller subset for faster results."""

import asyncio
import json
import tempfile
from src.datagen.synthetic.programs.program_generator import run_pipeline
from src.datagen.shared.verification import verify_question
from src.datagen.pipeline_ui import EpisodeGenUI
from src.core.config import config


async def test_question(q, csv_path, condition):
    """Test a single question with verification API."""
    ui = EpisodeGenUI()

    if condition == "with_verbalization":
        question_text = q.get("question_text", "")
        hint = q.get("hint", "")
    else:
        question_text = q.get("question_mechanical", "")
        hint = ""

    # Create question dict for verification
    question_dict = {
        "id": q.get("id", "test"),
        "question_text": question_text,
        "question_mechanical": q.get("question_mechanical", ""),
        "hint": hint,
        "ground_truth": q.get("ground_truth"),
        "ground_truth_hash": q.get("ground_truth_hash"),
        "difficulty": q.get("difficulty", "EASY"),
    }

    try:
        result = await verify_question(
            question=question_dict,
            csv_path=csv_path,
            strategy="ground_truth",
            model=config.teacher_model,
            ui=ui,
        )

        return {
            "success": result.success,
            "matches": result.match if result.match is not None else False,
            "error": result.error,
        }
    except Exception as e:
        return {
            "success": False,
            "matches": False,
            "error": str(e),
        }


async def main():
    """Quick test on one dataset with 3 questions."""
    csv_path = "data/csv/data.csv"
    print("Testing procedural question pass rates (3 questions)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Generating questions...")
        questions = await run_pipeline(
            csv_path=csv_path,
            max_programs=3,
            max_verbalize=3,
            skip_verbalization=False,
            output_dir=tmpdir,
        )

        print(f"Generated {len(questions)} questions\n")

        # Test with verbalization
        print("Testing WITH verbalization...")
        with_v = []
        for i, q in enumerate(questions):
            print(f"  Question {i + 1}/{len(questions)}...", end=" ")
            result = await test_question(q, csv_path, "with_verbalization")
            with_v.append(result)
            status = "✓" if result["matches"] else "✗"
            print(status)

        # Test without verbalization
        print("\nTesting WITHOUT verbalization...")
        without_v = []
        for i, q in enumerate(questions):
            print(f"  Question {i + 1}/{len(questions)}...", end=" ")
            result = await test_question(q, csv_path, "without_verbalization")
            without_v.append(result)
            status = "✓" if result["matches"] else "✗"
            print(status)

        # Calculate stats
        with_match = sum(1 for r in with_v if r["matches"])
        without_match = sum(1 for r in without_v if r["matches"])

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total questions: {len(questions)}")
        print(
            f"With verbalization:    {with_match}/{len(questions)} correct ({with_match / len(questions):.0%})"
        )
        print(
            f"Without verbalization: {without_match}/{len(questions)} correct ({without_match / len(questions):.0%})"
        )

        if with_match > without_match:
            print(f"\n✓ Verbalization helps! (+{with_match - without_match} questions)")
        elif with_match < without_match:
            print(f"\n✗ Verbalization hurts ({with_match - without_match} questions)")
        else:
            print(f"\n= No difference")

        # Show examples
        print("\nExample questions:")
        for i, q in enumerate(questions):
            print(f"\n{i + 1}. {q['n_steps']} steps - {q['program_name']}")
            print(f"   Q: {q['question_text'][:80]}...")
            print(f"   Mechanical: {q['question_mechanical'][:80]}...")
            print(f"   Ground truth: {q['ground_truth']}")
            print(f"   With verbalization: {'✓' if with_v[i]['matches'] else '✗'}")
            print(f"   Without: {'✓' if without_v[i]['matches'] else '✗'}")


if __name__ == "__main__":
    asyncio.run(main())
