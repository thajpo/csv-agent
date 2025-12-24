"""
Quick test of teacher triangulation.

This tests a single question to verify the triangulation logic works.
"""

import asyncio
from src.datagen.teacher import triangulate_teacher
from src.core.prompts import generate_data_overview, DEFAULT_DATASET_DESCRIPTION

async def main():
    # Test question
    question = "What is the mean TL (total length) for the control group?"
    hint = "Filter the data to the control group first, then calculate the mean."

    # Generate data overview
    data_overview = generate_data_overview("data/csv/data.csv")

    # Run triangulation (just 2 consistency traces for quick test)
    print("Running teacher triangulation...")
    print(f"Question: {question}")
    print(f"Hint: {hint}\n")

    gold_trace, _, _, consistency_results, verified = await triangulate_teacher(
        csv_path="data/csv/data.csv",
        question=question,
        hint=hint,
        model="openai/gpt-oss-120b",  # Required positional arg (4th)
        n_consistency=2,  # Quick test with 2 traces
        dataset_description=DEFAULT_DATASET_DESCRIPTION,
        data_overview=data_overview,
        max_turns=5,
        sampling_args={"temperature": 0.7, "max_tokens": 1000},
    )

    consistency_traces = [t for t, _ in consistency_results]

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nGold Trace (with hint):")
    print(f"  Success: {gold_trace.execution_success}")
    print(f"  Final Answer: {gold_trace.final_answer}")
    print(f"  Final Hash: {gold_trace.final_answer_hash}")
    print(f"  Code Cells: {len(gold_trace.code_cells)}")
    print(f"  Artifacts: {len(gold_trace.artifacts)}")

    print(f"\nConsistency Traces (without hint):")
    for i, trace in enumerate(consistency_traces, 1):
        print(f"  Trace {i}:")
        print(f"    Success: {trace.execution_success}")
        print(f"    Final Answer: {trace.final_answer}")
        print(f"    Final Hash: {trace.final_answer_hash}")

    print(f"\nVerification: {'✓ PASSED' if verified else '✗ FAILED'}")

    if verified:
        print("\nThe gold trace matches the majority of consistency traces!")
        print("This question is verified and ready for training.")
    else:
        print("\nThe gold trace does NOT match the majority.")
        print("This question should be filtered out.")

if __name__ == "__main__":
    asyncio.run(main())
