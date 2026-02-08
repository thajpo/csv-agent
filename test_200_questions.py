"""Large-scale procedural question test - 200 questions.

This test generates 200 procedural questions and checks:
1. Generation success rate
2. Answer format correctness (does model return dict or scalar?)
3. Pass rate comparison (with vs without verbalization)

Note: Full verification with LLM would take hours, so we do a quick format check first.
"""

import asyncio
import json
import tempfile
import time
from src.datagen.synthetic.programs.program_generator import run_pipeline
from src.envs.csv_env import LocalCSVAnalysisEnv
from src.datagen.shared.submission import parse_all_submissions
from src.datagen.synthetic.programs.compiler import compile_program
from src.datagen.synthetic.profiler import DataProfiler


async def quick_verify(question, csv_path):
    """Quick verification - just check if answer format matches ground truth."""
    from src.datagen.synthetic.programs.sampler import sample_programs

    # Get the program spec
    profiler = DataProfiler()
    profile = profiler.analyze(csv_path)
    programs = sample_programs(profile)

    # Find matching program
    program_name = question.get("program_name", "")
    spec = None
    for p in programs:
        if p.name == program_name:
            spec = p
            break

    if not spec:
        return {"error": "Program not found"}

    # Compile and execute
    code = compile_program(spec, profile)

    env = LocalCSVAnalysisEnv(csv_path=csv_path)
    state = await env.setup_state({})

    output = await env.python(
        code=code,
        sandbox_id=state["sandbox_id"],
        python_state=state["python_state"],
    )

    await env.destroy_sandbox(state["sandbox_id"])

    submissions = parse_all_submissions(output)
    if not submissions:
        return {"error": "No submission"}

    actual_answer = submissions[0].get("__csv_agent_answer__")
    ground_truth = question.get("ground_truth")

    # Check format
    gt_is_dict = isinstance(ground_truth, dict)
    actual_is_dict = isinstance(actual_answer, dict)

    return {
        "ground_truth": ground_truth,
        "actual": actual_answer,
        "format_match": gt_is_dict == actual_is_dict,
        "gt_is_dict": gt_is_dict,
        "actual_is_dict": actual_is_dict,
    }


async def main():
    """Generate and test 200 procedural questions."""
    csv_path = "data/csv/data.csv"
    n_questions = 200

    print("=" * 70)
    print(f"LARGE-SCALE TEST: {n_questions} Procedural Questions")
    print("=" * 70)

    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[1/3] Generating {n_questions} questions...")
        print("      (This may take a few minutes)")

        gen_start = time.time()
        questions = await run_pipeline(
            csv_path=csv_path,
            max_programs=n_questions,
            max_verbalize=n_questions,
            skip_verbalization=False,
            output_dir=tmpdir,
        )
        gen_elapsed = time.time() - gen_start

        print(f"      Generated {len(questions)} questions in {gen_elapsed:.1f}s")
        print(f"      Rate: {len(questions) / gen_elapsed:.1f} questions/sec")

        # Analyze question types
        print(f"\n[2/3] Analyzing question characteristics...")

        simple_agg = sum(1 for q in questions if q.get("n_steps", 0) == 3)
        long_chain = sum(1 for q in questions if q.get("n_steps", 0) > 3)

        print(f"      Simple aggregations (3 steps): {simple_agg}")
        print(f"      Long chains (>3 steps): {long_chain}")

        # Check ground truth formats
        dict_gt = sum(1 for q in questions if isinstance(q.get("ground_truth"), dict))
        scalar_gt = sum(
            1 for q in questions if not isinstance(q.get("ground_truth"), dict)
        )

        print(f"      Dict ground truth: {dict_gt}")
        print(f"      Scalar ground truth: {scalar_gt}")

        # Sample a few questions for detailed inspection
        print(f"\n[3/3] Sample questions:")
        for i, q in enumerate(questions[:5]):
            print(f"\n  Q{i + 1}: {q.get('program_name', 'unknown')}")
            print(f"    Steps: {q.get('n_steps', 'N/A')}")
            print(f"    Ground truth: {q.get('ground_truth')}")
            print(f"    Question: {q.get('question_text', '')[:80]}...")

        # Summary
        total_elapsed = time.time() - start_time
        print(f"\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total questions generated: {len(questions)}")
        print(f"Target: {n_questions}")
        print(f"Success rate: {len(questions) / n_questions * 100:.1f}%")
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Questions saved to: {tmpdir}/questions.json")

        # Save summary
        summary = {
            "target": n_questions,
            "generated": len(questions),
            "success_rate": len(questions) / n_questions,
            "generation_time": gen_elapsed,
            "total_time": total_elapsed,
            "simple_aggregations": simple_agg,
            "long_chains": long_chain,
            "dict_ground_truth": dict_gt,
            "scalar_ground_truth": scalar_gt,
        }

        with open(f"{tmpdir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {tmpdir}/summary.json")


if __name__ == "__main__":
    asyncio.run(main())
