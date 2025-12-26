"""
Compositional Question Generator.

Generates verifiable questions by:
1. Profiling the dataset
2. Selecting applicable composition templates
3. Executing templates to get ground truth answers
4. Verbalizing code into natural language questions

This approach guarantees:
- Questions are answerable (we have the ground truth)
- Questions require exploration (reference properties, not column names)
- Multi-turn behavior is structural (can't be one-shot)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import config
from src.datagen.synthetic.profiler import DataProfiler
from src.datagen.synthetic.templates import CompositionTemplate, get_applicable_templates
from src.datagen.synthetic.verbalizer import QuestionVerbalizer
from src.envs.csv_env import LocalCSVAnalysisEnv
from src.utils.hashing import hash_artifact


class CompositionalQuestionGenerator:
    """
    Generate verified questions via code composition.

    Pipeline:
    1. Profile dataset -> understand structure
    2. Select templates -> which patterns apply
    3. Execute in sandbox -> get ground truth
    4. Verbalize via LLM -> natural language question
    """

    def __init__(
        self,
        csv_path: str,
        model: str | None = None,
        sampling_args: dict | None = None,
    ):
        """
        Initialize the generator.

        Args:
            csv_path: Path to CSV dataset
            model: LLM model for verbalization (defaults to config.question_gen_model)
            sampling_args: LLM sampling args (defaults to config.sampling_args)
        """
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.model = model or config.question_gen_model
        self.sampling_args = sampling_args or config.sampling_args.model_dump()

        self.profiler = DataProfiler()
        self.verbalizer: QuestionVerbalizer | None = None
        self.env: LocalCSVAnalysisEnv | None = None
        self.state: dict | None = None

    async def setup(self) -> None:
        """Initialize sandbox and verbalizer."""
        self.verbalizer = QuestionVerbalizer(
            model=self.model,
            sampling_args=self.sampling_args,
        )
        self.env = LocalCSVAnalysisEnv(csv_path=str(self.csv_path))
        self.state = await self.env.setup_state({})

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.env and self.state:
            await self.env.destroy_sandbox(self.state["sandbox_id"])
        if self.verbalizer:
            await self.verbalizer.aclose()

    async def generate(
        self,
        n_questions: int | None = None,
    ) -> dict:
        """
        Generate compositional questions for the dataset.

        Args:
            n_questions: Max questions to generate (None = all applicable templates)

        Returns:
            Dict with dataset_columns and questions list
        """
        # 1. Profile the dataset
        print(f"Profiling dataset: {self.csv_path.name}")
        profile = self.profiler.analyze(str(self.csv_path))
        print(f"  Shape: {profile['shape']['rows']} rows x {profile['shape']['columns']} cols")

        # 2. Get applicable templates
        templates = get_applicable_templates(profile)
        print(f"  Applicable templates: {len(templates)}")

        expanded_templates: list[tuple[CompositionTemplate, dict[str, Any]]] = []
        for template in templates:
            for params in template.iter_param_sets():
                expanded_templates.append((template, params))

        if n_questions and len(expanded_templates) > n_questions:
            expanded_templates = expanded_templates[:n_questions]

        # 3. Execute all templates first (sequential - needs sandbox)
        print(f"\nExecuting {len(expanded_templates)} templates...")
        execution_results = []
        for i, (template, params) in enumerate(expanded_templates):
            params_label = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "default"
            code = template.instantiate(profile, params=params)

            try:
                result = await self._execute_code(code)
                if result:
                    ground_truth, answer_hash = result
                    # Skip questions where template returns an error
                    # (e.g., "No suitable categorical column found")
                    if isinstance(ground_truth, dict) and "error" in ground_truth:
                        print(f"  [{i+1}/{len(expanded_templates)}] {template.name} ({params_label}) - skipped (template error)")
                        continue
                    execution_results.append({
                        "template": template,
                        "params": params,
                        "code": code,
                        "ground_truth": ground_truth,
                        "answer_hash": answer_hash,
                    })
                    print(f"  [{i+1}/{len(expanded_templates)}] {template.name} ({params_label}) ✓")
                else:
                    print(f"  [{i+1}/{len(expanded_templates)}] {template.name} ({params_label}) - no answer")
            except Exception as e:
                print(f"  [{i+1}/{len(expanded_templates)}] {template.name} ({params_label}) FAILED: {e}")

        # 4. Verbalize all concurrently (parallel LLM calls)
        print(f"\nVerbalizing {len(execution_results)} questions concurrently...")

        async def verbalize_one(item: dict) -> dict | None:
            try:
                question_text, hint = await self.verbalizer.verbalize(
                    code=item["code"],
                    profile=profile,
                    ground_truth=item["ground_truth"],
                    output_schema=item["template"].output_schema,
                )
                if not question_text or question_text.startswith("[VERBALIZATION FAILED"):
                    return None
                return {
                    "question": question_text,
                    "hint": hint,
                    "n_steps": item["template"].n_steps,
                    "difficulty": item["template"].difficulty,
                    "template_name": item["template"].name,
                    "template_params": item["params"] or None,
                    "output_type": item["template"].output_type,
                    "output_schema": item["template"].output_schema,
                    "ground_truth_hash": item["answer_hash"],
                    "_ground_truth": item["ground_truth"],
                    "_template": item["template"].name,
                }
            except Exception as e:
                print(f"  Verbalization error: {e}")
                return None

        verbalized = await asyncio.gather(*[verbalize_one(item) for item in execution_results])
        questions = [q for q in verbalized if q is not None]
        print(f"  Verbalized {len(questions)}/{len(execution_results)} successfully")

        # 5. Format output
        df = pd.read_csv(self.csv_path, nrows=0)
        output = {
            "dataset_columns": df.columns.tolist(),
            "questions": questions,
        }

        print(f"\nGenerated {len(questions)} questions")
        return output

    async def _execute_code(self, code: str) -> tuple[Any, str] | None:
        """
        Execute code in sandbox and extract answer.

        Returns:
            Tuple of (ground_truth_value, hash) or None if failed
        """
        # Reset sandbox state for clean execution
        await self.env.reset(
            self.state["sandbox_id"],
            self.state["python_state"],
        )

        # Execute the code
        output = await self.env.python(
            code=code,
            sandbox_id=self.state["sandbox_id"],
            python_state=self.state["python_state"],
        )

        # Parse the submitted answer
        answer = self._parse_submission(output)
        if answer is None:
            print(f"  No answer submitted. Output: {output[:200]}")
            return None

        # Hash the answer
        answer_hash = hash_artifact(answer)

        return answer, answer_hash

    def _parse_submission(self, output: str) -> Any | None:
        """Extract submitted answer from execution output."""
        # Look for the submission marker
        marker = "✓ Submitted: "
        if marker not in output:
            return None

        # Extract JSON after marker
        start = output.index(marker) + len(marker)
        end = output.find("\n", start)
        if end == -1:
            json_str = output[start:]
        else:
            json_str = output[start:end]

        try:
            submission = json.loads(json_str.strip())
            return submission.get("__csv_agent_answer__")
        except json.JSONDecodeError:
            return None


async def generate_questions(
    csv_path: str,
    output_dir: str | None = None,
    n_questions: int | None = None,
    model: str | None = None,
) -> dict:
    """
    Main entry point for generating compositional questions.

    Args:
        csv_path: Path to CSV dataset
        output_dir: Directory to save questions.json (defaults to data/questions/{dataset_name}/)
        n_questions: Max questions to generate
        model: LLM model for verbalization

    Returns:
        Generated questions dict
    """
    generator = CompositionalQuestionGenerator(
        csv_path=csv_path,
        model=model,
    )

    try:
        await generator.setup()
        result = await generator.generate(n_questions=n_questions)

        # Save output
        if output_dir is None:
            # Use parent folder name (e.g., "mirichoi0218_insurance" from ".../mirichoi0218_insurance/data.csv")
            dataset_name = Path(csv_path).parent.name
            output_dir = f"{config.questions_synthetic_dir}/{dataset_name}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        questions_file = output_path / "questions.json"

        with open(questions_file, "w") as f:
            # Keep ground_truth and template for evaluation, but mark as internal
            clean_questions = []
            for q in result["questions"]:
                clean_q = {
                    k: v for k, v in q.items()
                    if k in ("_ground_truth", "_template") or not k.startswith("_")
                }
                clean_questions.append(clean_q)

            output = {
                "dataset_columns": result["dataset_columns"],
                "questions": clean_questions,
            }
            json.dump(output, f, indent=2)

        print(f"\nSaved to: {questions_file}")
        return result

    finally:
        await generator.cleanup()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate compositional questions for a CSV dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to CSV dataset(s). If omitted, uses all datasets from data/kaggle/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for questions.json (default: data/questions/{dataset_name}/)",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Maximum number of questions to generate (default: all applicable)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"LLM model for verbalization (default: {config.question_gen_model})",
    )

    args = parser.parse_args()

    # Auto-discover datasets if none specified
    csv_paths = args.csv if args.csv else config.csv_sources
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    if not csv_paths:
        print("No CSV files found. Specify --csv or ensure data/kaggle/ has datasets.")
        return 1

    total_questions = 0
    failed_csvs = []

    for csv_path in csv_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_path}")
        print("=" * 60)

        try:
            result = asyncio.run(
                generate_questions(
                    csv_path=csv_path,
                    output_dir=args.output_dir,
                    n_questions=args.n_questions,
                    model=args.model,
                )
            )
            total_questions += len(result["questions"])
            print(f"Generated {len(result['questions'])} questions")

        except KeyboardInterrupt:
            print("\nInterrupted")
            return 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed_csvs.append(csv_path)
            continue

    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_questions} questions from {len(csv_paths) - len(failed_csvs)} datasets")
    if failed_csvs:
        print(f"Failed: {len(failed_csvs)} datasets")
        for f in failed_csvs:
            print(f"  - {f}")
    return 0 if not failed_csvs else 1


if __name__ == "__main__":
    sys.exit(main())
