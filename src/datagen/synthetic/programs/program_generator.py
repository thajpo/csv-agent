"""Program-based question generation pipeline.

Generates questions by:
1) Sampling compositional programs
2) Compiling and executing to get ground truth
3) Filtering program outputs
4) Verbalizing into hypothesis-style questions
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.core.config import config
from src.core.prompts import generate_data_overview
from src.datagen.synthetic.profiler import DataProfiler
from src.datagen.synthetic.verbalizer import QuestionVerbalizer
from src.datagen.synthetic.programs.template_verbalizer import TemplateVerbalizer
from src.datagen.synthetic.programs.compiler import compile_program
from src.datagen.synthetic.programs.filter import filter_programs
from src.datagen.synthetic.programs.sampler import sample_programs
from src.envs.csv_env import LocalCSVAnalysisEnv
from src.datagen.shared.dataset_meta import (
    load_dataset_meta,
    generate_description_from_overview,
)
from src.datagen.shared.filters import FORBIDDEN_METHOD_TERMS, check_program_output
from src.datagen.shared.submission import parse_all_submissions
from csv_spec import hash_artifact
import uuid


OUTPUT_SCHEMAS = {
    "mean": {"column": "<name>", "mean": 0.0},
    "median": {"column": "<name>", "median": 0.0},
    "std": {"column": "<name>", "std": 0.0},
    "variance": {"column": "<name>", "variance": 0.0},
    "mean_series": {"metric": "mean", "value": 0.0},
    "median_series": {"metric": "median", "value": 0.0},
    "std_series": {"metric": "std", "value": 0.0},
    "max_series": {"metric": "max", "value": 0.0},
    "min_series": {"metric": "min", "value": 0.0},
    "argmax_group": {"group": "<name>", "mean": 0.0},
    "argmin_group": {"group": "<name>", "mean": 0.0},
    "argmax_group_median": {"group": "<name>", "median": 0.0},
    "argmin_group_median": {"group": "<name>", "median": 0.0},
    "argmax_group_std": {"group": "<name>", "std": 0.0},
    "argmin_group_std": {"group": "<name>", "std": 0.0},
    "argmax_group_var": {"group": "<name>", "variance": 0.0},
    "argmin_group_var": {"group": "<name>", "variance": 0.0},
    "argmax_group_count": {"group": "<name>", "count": 0},
    "argmin_group_count": {"group": "<name>", "count": 0},
    "ttest_ind": {"test": "ttest|mwu", "stat": 0.0, "p_value": 0.0},
}


def _output_schema_for_op(op_name: str) -> str:
    schema = OUTPUT_SCHEMAS.get(op_name, {"answer": 0.0})
    return json.dumps(schema)


def _generate_mechanical_description(ops: list[str]) -> str:
    """Generate a mechanical description from program operations."""
    if not ops:
        return "Analyze the dataset and return a result"

    op_descriptions = []
    for op in ops:
        # Map common ops to readable descriptions
        if "select" in op.lower():
            op_descriptions.append("select relevant columns")
        elif "bind" in op.lower():
            op_descriptions.append("bind column parameters")
        elif "group" in op.lower() or "groupby" in op.lower():
            op_descriptions.append("group data")
        elif "mean" in op.lower():
            op_descriptions.append("compute mean")
        elif "median" in op.lower():
            op_descriptions.append("compute median")
        elif "std" in op.lower() or "variance" in op.lower():
            op_descriptions.append("compute variance")
        elif "max" in op.lower() or "min" in op.lower():
            op_descriptions.append("find extreme values")
        elif "count" in op.lower():
            op_descriptions.append("count records")
        elif "filter" in op.lower():
            op_descriptions.append("filter rows")
        elif "correlation" in op.lower():
            op_descriptions.append("compute correlation")
        elif "test" in op.lower():
            op_descriptions.append("run statistical test")
        elif "decision" in op.lower() or "choose" in op.lower():
            op_descriptions.append("make decision based on evidence")

    if op_descriptions:
        return f"Program: {' → '.join(op_descriptions)}"
    return f"Execute program: {' → '.join(ops)}"


async def run_pipeline(
    csv_path: str,
    max_programs: int | None = None,
    max_verbalize: int | None = None,
    skip_verbalization: bool = False,
    output_dir: str | None = None,
) -> list[dict]:
    csv_path_str = str(Path(csv_path).resolve())
    profiler = DataProfiler()
    profile = profiler.analyze(csv_path_str)

    # Use shared dataset meta loader
    dataset_name, dataset_description = load_dataset_meta(csv_path_str)

    # Generate description from data_overview if missing
    data_overview = generate_data_overview(csv_path_str)
    if not dataset_description or not dataset_description.strip():
        dataset_description = generate_description_from_overview(data_overview)

    programs = sample_programs(profile)
    if max_programs:
        programs = programs[:max_programs]

    env = LocalCSVAnalysisEnv(csv_path=csv_path_str)
    state = await env.setup_state({})

    executed = []
    total_programs = len(programs)
    for idx, spec in enumerate(programs, start=1):
        print(f"[execute] {idx}/{total_programs} {spec.name}")
        try:
            code = compile_program(spec, profile)
            output = await env.python(
                code=code,
                sandbox_id=state["sandbox_id"],
                python_state=state["python_state"],
            )
        except Exception:
            continue

        submissions = parse_all_submissions(output)
        if not submissions:
            continue

        submission = submissions[0]
        answer = submission.get("__csv_agent_answer__")
        hooks = submission.get("hooks", [])
        if answer is None:
            continue

        executed.append(
            {
                "program": spec,
                "name": spec.name,
                "ops": [op.op_name for op in spec.ops],
                "code": code,
                "answer": answer,
                "hooks": hooks,
                "stdout": output,
                "row_count": profile.get("shape", {}).get("rows", 0),
            }
        )

    await env.destroy_sandbox(state["sandbox_id"])

    filtered = filter_programs(executed)
    if max_verbalize:
        filtered = filtered[:max_verbalize]

    # Initialize verbalizers
    llm_verbalizer = (
        None
        if skip_verbalization
        else QuestionVerbalizer(
            model=config.question_gen_model,
            sampling_args=config.sampling_args.model_dump(),
        )
    )
    template_verbalizer = TemplateVerbalizer()

    verbalized = []
    total_verbalize = len(filtered)

    for idx, item in enumerate(filtered, start=1):
        print(f"[verbalize] {idx}/{total_verbalize} {item['name']}")

        n_steps = len(item["ops"])

        # For long chains (10+ steps), use template verbalizer (deterministic, accurate)
        # For short chains, optionally use LLM verbalizer
        if n_steps >= 10:
            # Template-based verbalization for long chains
            # This ensures the question accurately describes the computation
            from src.datagen.synthetic.programs.spec import ProgramSpec

            spec = ProgramSpec(
                name=item["name"],
                ops=[],  # Not needed for verbalization
                output_type="dict",
                output_schema="",
                difficulty="HARD",
            )
            question, hint = template_verbalizer.verbalize(spec)

            # Optional: Polish with LLM (preserves meaning, adds natural language)
            if not skip_verbalization and False:  # Set to True to enable LLM polish
                question = await template_verbalizer.polish_with_llm(
                    question,
                    hint,
                    config.question_gen_model,
                    config.sampling_args.model_dump(),
                )
        else:
            # Short chains: use LLM verbalizer or mechanical description
            if skip_verbalization:
                terminal = item["ops"][-1]
                question = f"Analyze the data and return an answer. Return as JSON, e.g.: {_output_schema_for_op(terminal)}"
                hint = ""
            elif llm_verbalizer is not None:
                terminal = item["ops"][-1]
                output_schema = _output_schema_for_op(terminal)
                question, hint, _raw = await llm_verbalizer.verbalize(
                    code=item["code"],
                    profile=profile,
                    ground_truth=item["answer"],
                    output_schema=output_schema,
                    data_overview=data_overview,
                    dataset_description=dataset_description,
                    banned_words=list(FORBIDDEN_METHOD_TERMS),
                )
            else:
                # Fallback if somehow we get here with no verbalizer
                terminal = item["ops"][-1]
                question = f"Analyze the data and return an answer. Return as JSON, e.g.: {_output_schema_for_op(terminal)}"
                hint = ""

        verbalized.append({"program": item, "question": question, "hint": hint})

    if llm_verbalizer:
        await llm_verbalizer.aclose()

    # Transform to unified schema and save
    output_path = (
        Path(output_dir)
        if output_dir
        else Path("data/questions_synthetic") / dataset_name
    )
    output_path.mkdir(parents=True, exist_ok=True)

    question_records = []
    for item in verbalized:
        program = item["program"]
        ops = program["ops"]
        code = program["code"]
        answer = program["answer"]

        # Generate deterministic ID
        id_base = f"{dataset_name}_{program['name']}_{hash_artifact(code)[:8]}"
        question_id = f"prog_{id_base}"

        # Build unified schema record
        record = {
            "id": question_id,
            "source": "procedural",
            "dataset": dataset_name,
            "question_mechanical": _generate_mechanical_description(ops),
            "question_text": item["question"],
            "hint": item["hint"],
            "code": code,
            "code_hash": hash_artifact(code),
            "ground_truth": answer,
            "ground_truth_hash": hash_artifact(answer),
            "output_schema": _output_schema_for_op(ops[-1]),
            "n_steps": len(ops),
            "difficulty": None,
            "dataset_description": dataset_description,
            "program_name": program["name"],
            "program_ops": ops,
        }
        question_records.append(record)

    # Save batch
    questions_file = output_path / "questions.json"
    with open(questions_file, "w") as f:
        json.dump(question_records, f, indent=2)

    print(f"Executed: {len(executed)}")
    print(f"Filtered: {len(filtered)}")
    print(f"Verbalized: {len(verbalized)}")
    print(f"Saved {len(question_records)} questions to {questions_file}")

    return question_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Program-based question generation")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--max-programs", type=int, default=None, help="Max programs to execute"
    )
    parser.add_argument(
        "--max-verbalize", type=int, default=None, help="Max programs to verbalize"
    )
    parser.add_argument(
        "--skip-verbalization", action="store_true", help="Skip LLM verbalization"
    )
    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            csv_path=args.csv,
            max_programs=args.max_programs,
            max_verbalize=args.max_verbalize,
            skip_verbalization=args.skip_verbalization,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
