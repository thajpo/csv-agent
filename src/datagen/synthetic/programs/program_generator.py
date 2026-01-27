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
from src.datagen.synthetic.programs.compiler import compile_program
from src.datagen.synthetic.programs.filter import filter_programs
from src.datagen.synthetic.programs.sampler import sample_programs
from src.envs.csv_env import LocalCSVAnalysisEnv
from src.datagen.shared.dataset_meta import (
    load_dataset_meta,
    generate_description_from_overview,
)
from src.datagen.shared.filters import FORBIDDEN_METHOD_TERMS
from src.datagen.shared.submission import parse_all_submissions


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


async def run_pipeline(
    csv_path: str,
    max_programs: int | None = None,
    max_verbalize: int | None = None,
    skip_verbalization: bool = False,
) -> None:
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

    if skip_verbalization:
        verbalized = [
            {
                "program": item,
                "question": f"Analyze the data and return an answer. Return as JSON, e.g.: {_output_schema_for_op(item['ops'][-1])}",
                "hint": "",
            }
            for item in filtered
        ]
    else:
        verbalizer = QuestionVerbalizer(
            model=config.question_gen_model,
            sampling_args=config.sampling_args.model_dump(),
        )
        verbalized = []
        total_verbalize = len(filtered)
        for idx, item in enumerate(filtered, start=1):
            print(f"[verbalize] {idx}/{total_verbalize} {item['name']}")
            terminal = item["ops"][-1]
            output_schema = _output_schema_for_op(terminal)
            question, hint, _raw = await verbalizer.verbalize(
                code=item["code"],
                profile=profile,
                ground_truth=item["answer"],
                output_schema=output_schema,
                data_overview=data_overview,
                dataset_description=dataset_description,
                banned_words=FORBIDDEN_METHOD_TERMS,
            )
            verbalized.append({"program": item, "question": question, "hint": hint})
        await verbalizer.aclose()

    print(f"Executed: {len(executed)}")
    print(f"Filtered: {len(filtered)}")
    print(f"Verbalized: {len(verbalized)}")


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
