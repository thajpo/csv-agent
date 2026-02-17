"""Quick test - just generate procedural questions and inspect them."""

import asyncio
import tempfile
from src.datagen.synthetic.programs.program_generator import run_pipeline


async def main():
    """Generate and inspect procedural questions."""
    csv_path = "data/csv/data.csv"
    print("Generating procedural questions...")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        questions = await run_pipeline(
            csv_path=csv_path,
            max_programs=3,
            max_verbalize=3,
            skip_verbalization=False,  # Enable verbalization
            output_dir=tmpdir,
        )

        print(f"\nGenerated {len(questions)} questions\n")
        print("=" * 60)

        for i, q in enumerate(questions):
            print(f"\nQuestion {i + 1}: {q.get('program_name', 'unknown')}")
            print(f"  Steps: {q.get('n_steps', 'N/A')}")
            print(f"  Difficulty: {q.get('difficulty', 'N/A')}")
            print(f"  Ground truth: {q.get('ground_truth')}")
            print("\n  Verbalized question:")
            print(f"  {q.get('question_text', '')[:200]}...")
            print("\n  Mechanical question:")
            print(f"  {q.get('question_mechanical', '')}")
            print(
                f"\n  Hint: {q.get('hint', '')[:100] if q.get('hint') else 'None'}..."
            )
            print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
