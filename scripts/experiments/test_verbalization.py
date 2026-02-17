"""Test verbalization performance on multiple datasets.

This script generates procedural questions on 3 datasets and analyzes
verbalization quality.
"""

import asyncio
import json
import tempfile
from src.datagen.synthetic.programs.program_generator import run_pipeline
from src.datagen.synthetic.profiler import DataProfiler


async def test_dataset(csv_path: str, dataset_name: str, max_questions: int = 20):
    """Generate procedural questions and analyze verbalization."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {dataset_name}")
    print(f"File: {csv_path}")
    print("=" * 80)

    # Profile dataset
    profiler = DataProfiler()
    profile = profiler.analyze(csv_path)

    print("\nDataset Profile:")
    print(f"  Rows: {profile['shape']['rows']}")
    print(f"  Columns: {profile['shape']['columns']}")
    numeric_cols = [
        c for c, info in profile["columns"].items() if info["type"] == "numeric"
    ]
    cat_cols = [
        c for c, info in profile["columns"].items() if info["type"] == "categorical"
    ]
    print(f"  Numeric: {len(numeric_cols)} ({numeric_cols[:5]}...)")
    print(f"  Categorical: {len(cat_cols)} ({cat_cols[:5]}...)")

    # Generate questions
    with tempfile.TemporaryDirectory() as tmpdir:
        questions = await run_pipeline(
            csv_path=csv_path,
            max_programs=max_questions,
            max_verbalize=max_questions,
            skip_verbalization=False,  # Use template verbalization
            output_dir=tmpdir,
        )

        print(f"\nGenerated {len(questions)} questions")

        # Analyze by template type
        by_template = {}
        for q in questions:
            template_type = "unknown"
            if "cascading" in q.get("program_name", ""):
                template_type = "cascading"
            elif "derived" in q.get("program_name", ""):
                template_type = "derived"
            elif "evidence" in q.get("program_name", ""):
                template_type = "evidence"

            if template_type not in by_template:
                by_template[template_type] = []
            by_template[template_type].append(q)

        # Analyze verbalization quality
        results = {
            "dataset": dataset_name,
            "csv_path": csv_path,
            "total_questions": len(questions),
            "by_template": {},
            "examples": [],
        }

        for template_type, qs in by_template.items():
            results["by_template"][template_type] = {
                "count": len(qs),
                "avg_steps": sum(q["n_steps"] for q in qs) / len(qs) if qs else 0,
            }

        # Show examples
        print("\nVerbalization Examples:")
        for i, q in enumerate(questions[:5], 1):
            print(f"\n{i}. {q['n_steps']} steps - {q.get('program_name', 'unknown')}")
            print(f"   Q: {q['question_text'][:100]}...")
            print(f"   H: {q.get('hint', '')[:80]}...")

            results["examples"].append(
                {
                    "steps": q["n_steps"],
                    "template": q.get("program_name", "unknown"),
                    "question": q["question_text"],
                    "hint": q.get("hint", ""),
                }
            )

        return results


async def main():
    """Test verbalization on 3 datasets."""
    datasets = [
        ("data/csv/data.csv", "Base Dataset"),
        ("data/kaggle/fedesoriano_heart-failure-prediction/data.csv", "Heart Failure"),
        ("data/kaggle/gregorut_videogamesales/data.csv", "Video Game Sales"),
    ]

    all_results = []

    for csv_path, name in datasets:
        try:
            results = await test_dataset(csv_path, name, max_questions=15)
            all_results.append(results)
        except Exception as e:
            print(f"\nError processing {name}: {e}")
            import traceback

            traceback.print_exc()

    # Save report
    report_path = "verbalization_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'=' * 80}")
    print(f"Report saved to: {report_path}")
    print("=" * 80)

    # Summary
    print("\nSummary:")
    for r in all_results:
        print(f"\n{r['dataset']}:")
        print(f"  Total: {r['total_questions']} questions")
        for template, data in r["by_template"].items():
            print(
                f"  {template}: {data['count']} questions, avg {data['avg_steps']:.1f} steps"
            )


if __name__ == "__main__":
    asyncio.run(main())
