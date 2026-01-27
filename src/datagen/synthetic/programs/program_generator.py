"""Program-based question generation pipeline.

Generates questions by:
1) Sampling compositional programs
2) Compiling and executing to get ground truth
3) Filtering program outputs
4) Verbalizing into hypothesis-style questions
5) Validating with a student trace to compute pass rate
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from src.core.config import config
from src.core.prompts import generate_data_overview
from src.datagen.teacher import answers_match, execute_teacher_trace
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


class _NullConsole:
    def print(self, *args, **kwargs) -> None:
        pass


class _SilentTraceUI:
    def __init__(self):
        self.console = _NullConsole()

    def print_trace_start(self, mode: str) -> None:
        return None

    def print_turn(
        self,
        turn_num: int,
        max_turns: int,
        response: str,
        code_cells: list[str],
        execution_results: list[dict],
    ) -> None:
        return None

    def print_trace_complete(
        self,
        success: bool,
        final_answer: Any,
        turns: int,
        elapsed_seconds: float | None = None,
    ) -> None:
        return None


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


def _parse_submissions(output: str) -> list[dict]:
    marker = "✓ Submitted: "
    submissions = []
    pos = 0
    while True:
        idx = output.find(marker, pos)
        if idx == -1:
            break
        start = idx + len(marker)
        end = output.find("\n", start)
        json_str = output[start:] if end == -1 else output[start:end]
        try:
            submission = json.loads(json_str.strip())
        except json.JSONDecodeError:
            pos = start
            continue
        submissions.append(submission)
        pos = start
    return submissions


async def _validate_question(
    csv_path: str,
    question_text: str,
    expected_value: Any,
    expected_hash: str,
    data_overview: str,
    dataset_description: str,
    n_steps: int | None = None,
    difficulty: str | None = None,
) -> tuple[bool, dict]:
    validation_model = (
        config.synthetic_question_validation_model or config.teacher_model
    )
    max_turns = config.synthetic_question_validation_max_turns or config.max_turns
    ui = _SilentTraceUI()

    try:
        trace, _conversation, _system, elapsed = await execute_teacher_trace(
            csv_path=csv_path,
            question=question_text,
            model=validation_model,
            hint=None,
            n_steps=n_steps,
            difficulty=difficulty,
            mode="student",
            dataset_description=dataset_description,
            data_overview=data_overview,
            max_turns=max_turns,
            sampling_args={
                "temperature": config.sampling_args.temperature,
                "max_tokens": config.sampling_args.max_tokens,
                "top_p": config.sampling_args.top_p,
            },
            ui=ui,
            trace_mode="validation",
        )
    except Exception as exc:
        return False, {
            "model": validation_model,
            "success": False,
            "matched": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    success = trace.get("success", False)
    final_answer = trace.get("final_answer")
    final_hash = trace.get("final_answer_hash")
    matched = False
    if success:
        if answers_match(
            final_hash,
            expected_hash,
            final_answer,
            expected_value,
            float_tol=config.float_tolerance,
        ):
            matched = True

    return matched, {
        "model": validation_model,
        "success": success,
        "matched": matched,
        "final_answer_hash": final_hash,
    }


async def run_pipeline(
    csv_path: str,
    max_programs: int | None = None,
    max_verbalize: int | None = None,
    skip_verbalization: bool = False,
    skip_validation: bool = False,
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

        submissions = _parse_submissions(output)
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

    if skip_validation:
        print(f"Executed: {len(executed)}")
        print(f"Filtered: {len(filtered)}")
        print(f"Verbalized: {len(verbalized)}")
        return

    matched = 0
    for item in verbalized:
        program = item["program"]
        expected_value = program["answer"]
        expected_hash = ""
        try:
            from csv_spec import hash_artifact

            expected_hash = hash_artifact(expected_value)
        except Exception:
            expected_hash = ""

        ok, _info = await _validate_question(
            csv_path=csv_path,
            question_text=item["question"],
            expected_value=expected_value,
            expected_hash=expected_hash,
            data_overview=data_overview,
            dataset_description=dataset_description,
        )
        if ok:
            matched += 1

    total = len(verbalized)
    pass_rate = (matched / total) if total else 0.0

    print(f"Executed: {len(executed)}")
    print(f"Filtered: {len(filtered)}")
    print(f"Verbalized: {len(verbalized)}")
    print(f"Pass rate: {matched}/{total} ({pass_rate:.2%})")


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
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip student validation"
    )
    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            csv_path=args.csv,
            max_programs=args.max_programs,
            max_verbalize=args.max_verbalize,
            skip_verbalization=args.skip_verbalization,
            skip_validation=args.skip_validation,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
