"""Quick pass rate test - 2 questions with timing."""

import asyncio
import json
import tempfile
import time
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

    question_dict = {
        "id": q.get("id", "test"),
        "question_text": question_text,
        "question_mechanical": q.get("question_mechanical", ""),
        "hint": hint,
        "ground_truth": q.get("ground_truth"),
        "ground_truth_hash": q.get("ground_truth_hash"),
        "difficulty": q.get("difficulty", "EASY"),
    }

    start = time.time()
    try:
        result = await verify_question(
            question=question_dict,
            csv_path=csv_path,
            strategy="ground_truth",
            model=config.teacher_model,
            ui=ui,
        )
        elapsed = time.time() - start

        return {
            "success": result.success,
            "matches": result.match if result.match is not None else False,
            "error": result.error,
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "success": False,
            "matches": False,
            "error": str(e),
            "elapsed": elapsed,
        }


async def main():
    """Test with 2 questions."""
    csv_path = "data/csv/data.csv"
    print("Procedural Questions: Verbalization Impact Test")
    print("=" * 70)

    # Generate questions
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n[1/4] Generating 2 procedural questions...")
        gen_start = time.time()
        questions = await run_pipeline(
            csv_path=csv_path,
            max_programs=2,
            max_verbalize=2,
            skip_verbalization=False,
            output_dir=tmpdir,
        )
        gen_elapsed = time.time() - gen_start
        print(f"      Generated {len(questions)} questions in {gen_elapsed:.1f}s\n")

        # Test with verbalization
        print("[2/4] Testing WITH verbalization...")
        with_v = []
        for i, q in enumerate(questions):
            print(f"      Q{i + 1}: {q.get('question_text', '')[:60]}...")
            result = await test_question(q, csv_path, "with_verbalization")
            with_v.append(result)
            status = "✓ PASS" if result["matches"] else "✗ FAIL"
            print(f"      Result: {status} ({result['elapsed']:.1f}s)")
            if result["error"]:
                print(f"      Error: {result['error'][:100]}")
        print()

        # Test without verbalization
        print("[3/4] Testing WITHOUT verbalization...")
        without_v = []
        for i, q in enumerate(questions):
            print(f"      Q{i + 1}: {q.get('question_mechanical', '')[:60]}...")
            result = await test_question(q, csv_path, "without_verbalization")
            without_v.append(result)
            status = "✓ PASS" if result["matches"] else "✗ FAIL"
            print(f"      Result: {status} ({result['elapsed']:.1f}s)")
            if result["error"]:
                print(f"      Error: {result['error'][:100]}")
        print()

        # Results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)

        with_match = sum(1 for r in with_v if r["matches"])
        without_match = sum(1 for r in without_v if r["matches"])
        with_time = sum(r["elapsed"] for r in with_v)
        without_time = sum(r["elapsed"] for r in without_v)

        print(f"\nPass Rates:")
        print(
            f"  With verbalization:    {with_match}/{len(questions)} ({with_match / len(questions) * 100:.0f}%)"
        )
        print(
            f"  Without verbalization: {without_match}/{len(questions)} ({without_match / len(questions) * 100:.0f}%)"
        )

        print(f"\nTiming:")
        print(
            f"  With verbalization:    {with_time:.1f}s total, {with_time / len(questions):.1f}s avg"
        )
        print(
            f"  Without verbalization: {without_time:.1f}s total, {without_time / len(questions):.1f}s avg"
        )

        if with_match > without_match:
            print(f"\n✓ Verbalization helps! (+{with_match - without_match} correct)")
        elif with_match < without_match:
            print(f"\n✗ Verbalization hurts ({with_match - without_match} correct)")
        else:
            print(f"\n= No difference in accuracy")

        # Per-question breakdown
        print("\n" + "=" * 70)
        print("PER-QUESTION BREAKDOWN")
        print("=" * 70)
        for i, q in enumerate(questions):
            print(f"\nQ{i + 1}: {q.get('program_name', 'unknown')}")
            print(f"  Ground truth: {q.get('ground_truth')}")
            print(
                f"  With verbalization:    {'✓' if with_v[i]['matches'] else '✗'} ({with_v[i]['elapsed']:.1f}s)"
            )
            print(
                f"  Without verbalization: {'✓' if without_v[i]['matches'] else '✗'} ({without_v[i]['elapsed']:.1f}s)"
            )


if __name__ == "__main__":
    asyncio.run(main())
