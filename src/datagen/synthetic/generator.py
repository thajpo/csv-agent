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
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import config
from src.core.prompts import generate_data_overview
from src.datagen.teacher import answers_match, execute_teacher_trace
from src.datagen.synthetic.profiler import DataProfiler
from src.datagen.synthetic.templates import (
    CompositionTemplate,
    get_applicable_templates,
    get_eligible_categorical_columns,
    get_eligible_numeric_columns,
)
from src.datagen.synthetic.verbalizer import QuestionVerbalizer
from src.envs.csv_env import LocalCSVAnalysisEnv
from csv_spec import (
    hash_artifact,
    EpisodeJSONL,
    QADict,
    TriangulationMetadataDict,
    TimingMetadataDict,
)
from src.gui.progress_writer import ProgressWriter, NoOpProgressWriter
from src.datagen.shared.dataset_meta import (
    load_dataset_meta,
    generate_description_from_overview,
)
from src.datagen.shared.filters import FORBIDDEN_METHOD_TERMS

# Dataset viability thresholds
MIN_DATASET_ROWS = 50
MIN_DATASET_COLUMNS = 2
MAX_MISSING_PCT_PER_COLUMN = 95.0


def _dataset_is_viable(profile: dict) -> tuple[bool, str]:
    """Gate synthetic generation to avoid degenerate or unlearnable datasets."""
    shape = profile.get("shape", {})
    rows = shape.get("rows", 0) or 0
    cols = shape.get("columns", 0) or 0

    if rows < MIN_DATASET_ROWS:
        return False, f"too few rows ({rows})"
    if cols < MIN_DATASET_COLUMNS:
        return False, f"too few columns ({cols})"

    eligible_numeric = get_eligible_numeric_columns(profile)
    eligible_categorical = get_eligible_categorical_columns(profile)
    if not eligible_numeric and not eligible_categorical:
        return False, "no eligible columns after filtering ids/degenerate fields"

    # Skip datasets that are essentially empty
    columns = profile.get("columns", {})
    if columns:
        high_missing = [
            col
            for col, info in columns.items()
            if info.get("missing_pct", 0) >= MAX_MISSING_PCT_PER_COLUMN
        ]
        if len(high_missing) == cols:
            return False, f"all columns are >={MAX_MISSING_PCT_PER_COLUMN}% missing"

    return True, "ok"


# Method terms that leak the solution approach - shared with verbalizer
FORBIDDEN_METHOD_TERMS = [
    "regression",
    "t-test",
    "anova",
    "p-value",
    "chi-square",
    "bootstrap",
    "cross-validation",
    "k-fold",
    "random_state",
    "train-test",
    "ols",
    "pearson",
    "spearman",
    "ks test",
    "kolmogorov",
    "levene",
    "mann-whitney",
    "propensity",
    "matching",
    "confidence interval",
]


def _question_is_viable(question: str, profile: dict) -> tuple[bool, str]:
    """Reject questions that leak method details or column names."""
    if not question or not question.strip():
        return False, "empty question"

    # Heuristic sentence count to keep prompts concise and non-procedural.
    # Exclude the ICL example part (after "Return as JSON" or "e.g.:") from sentence counting
    # since JSON examples contain periods in decimals that inflate the count.
    question_for_counting = re.split(r"Return as JSON|e\.g\.:", question, maxsplit=1)[0]
    sentence_count = len(re.findall(r"[.!?]", question_for_counting))
    if sentence_count > 3:
        return False, "too many sentences"

    lowered = question.lower()
    if any(term in lowered for term in FORBIDDEN_METHOD_TERMS):
        return False, "mentions method details"

    # Avoid explicit column names in questions to keep them property-based.
    for col in profile.get("columns", {}):
        col_name = str(col).strip().lower()
        if len(col_name) < 4:
            continue
        if col_name in lowered:
            return False, "mentions column names"

    return True, "ok"


class _NullConsole:
    """No-op console that silently ignores all print calls."""

    def print(self, *args, **kwargs) -> None:
        pass


class _SilentTraceUI:
    """No-op UI for silent student validation traces."""

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
        dataset_description: str = "",
    ):
        """
        Initialize the generator.

        Args:
            csv_path: Path to CSV dataset
            model: LLM model for verbalization (defaults to config.question_gen_model)
            sampling_args: LLM sampling args (defaults to config.sampling_args)
            dataset_description: Human description of the dataset (from meta.json)
        """
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Use shared dataset meta loader
        self.dataset_name, self.dataset_description = load_dataset_meta(str(csv_path))

        # Generate description from data_overview if missing
        if not self.dataset_description or not self.dataset_description.strip():
            data_overview = generate_data_overview(str(csv_path))
            self.dataset_description = generate_description_from_overview(data_overview)

        self.model = model or config.question_gen_model
        self.sampling_args = sampling_args or config.sampling_args.model_dump()

        self.profiler = DataProfiler()
        self.verbalizer: QuestionVerbalizer | None = None
        self.env: LocalCSVAnalysisEnv | None = None
        self.state: dict | None = None
        self.data_overview: str = ""

    async def setup(self) -> None:
        """Initialize sandbox, verbalizer (if needed), and generate data overview."""
        # Only initialize verbalizer if we're using LLM verbalization
        if not config.synthetic_skip_verbalization:
            self.verbalizer = QuestionVerbalizer(
                model=self.model,
                sampling_args=self.sampling_args,
            )
        self.env = LocalCSVAnalysisEnv(csv_path=str(self.csv_path))
        self.state = await self.env.setup_state({})

        # Generate data overview for richer verbalization context
        self.data_overview = generate_data_overview(str(self.csv_path))

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.env and self.state:
            await self.env.destroy_sandbox(self.state["sandbox_id"])
        if self.verbalizer:
            await self.verbalizer.aclose()

    async def _validate_question(
        self,
        question_text: str,
        expected_values: list[Any],
        expected_hashes: list[str],
        n_steps: int | None,
        difficulty: str | None,
    ) -> tuple[bool, dict, dict | None]:
        """Run a student validation trace and compare against expected answers.

        Returns:
            (matched, validation_info, trace) - trace is full TraceDict if matched, else None
        """
        validation_model = (
            config.synthetic_question_validation_model or config.teacher_model
        )
        max_turns = config.synthetic_question_validation_max_turns or config.max_turns
        ui = _SilentTraceUI()

        try:
            trace, _conversation, _system, elapsed = await execute_teacher_trace(
                csv_path=str(self.csv_path),
                question=question_text,
                model=validation_model,
                hint=None,
                n_steps=n_steps,
                difficulty=difficulty,
                mode="student",
                dataset_description=self.dataset_description,
                data_overview=self.data_overview,
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
            return (
                False,
                {
                    "model": validation_model,
                    "success": False,
                    "matched": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                None,
            )

        success = trace.get("success", False)
        final_answer = trace.get("final_answer")
        final_hash = trace.get("final_answer_hash")
        matched = False
        if success:
            for expected_value, expected_hash in zip(expected_values, expected_hashes):
                if answers_match(
                    final_hash,
                    expected_hash,
                    final_answer,
                    expected_value,
                    float_tol=config.float_tolerance,
                ):
                    matched = True
                    break

        validation_info = {
            "model": validation_model,
            "success": success,
            "matched": matched,
            "turns": len(trace.get("turns", [])),
            "elapsed": round(float(elapsed), 3),
            "final_answer_hash": final_hash,
        }
        # Always return full trace (matched=positive for SFT, unmatched=negative for DPO)
        return matched, validation_info, trace

    async def generate(
        self,
        n_questions: int | None = None,
        output_path: Path | None = None,
        retry_failed: bool = False,
    ) -> dict:
        """
        Generate compositional questions for the dataset.

        Args:
            n_questions: Max questions to generate (None = all applicable templates)
            output_path: Directory for incremental JSONL output (enables resume)
            retry_failed: If True, re-process previously failed questions

        Returns:
            Dict with dataset_columns and questions list
        """
        # 1. Profile the dataset
        print(f"Profiling dataset: {self.csv_path.name}")
        profile = self.profiler.analyze(str(self.csv_path))
        print(
            f"  Shape: {profile['shape']['rows']} rows x {profile['shape']['columns']} cols"
        )

        # Dataset gates: skip degenerate inputs before template selection.
        is_viable, reason = _dataset_is_viable(profile)
        if not is_viable:
            print(f"  Skipping synthetic generation: {reason}")
            return {
                "dataset_columns": list(profile.get("columns", {}).keys()),
                "questions": [],
            }

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
        # Templates can call submit() multiple times for tie-aware enumeration
        print(f"\nExecuting {len(expanded_templates)} templates...")
        execution_results = []
        for i, (template, params) in enumerate(expanded_templates):
            params_label = (
                ", ".join(f"{k}={v}" for k, v in params.items())
                if params
                else "default"
            )
            code = template.instantiate(profile, params=params)

            try:
                # Use _execute_code_multi to capture ALL submit() calls
                results = await self._execute_code_multi(code)
                if results:
                    # First result is primary, rest are from tie enumeration
                    ground_truth, answer_hash = results[0]

                    # Skip questions where template returns an error
                    # (e.g., "No suitable categorical column found")
                    if isinstance(ground_truth, dict) and "error" in ground_truth:
                        error_msg = ground_truth.get("error", "unknown error")
                        print(
                            f"  [{i + 1}/{len(expanded_templates)}] {template.name} ({params_label}) - skipped: {error_msg}"
                        )
                        continue

                    # Collect all valid answers from multi-submit
                    ground_truths = []
                    answer_hashes = []
                    for gt, h in results:
                        if not (isinstance(gt, dict) and "error" in gt):
                            if h not in answer_hashes:
                                ground_truths.append(gt)
                                answer_hashes.append(h)

                    # Also execute alternative code templates if present
                    alt_codes = template.instantiate_alternatives(profile, params)
                    for alt_idx, alt_code in enumerate(alt_codes):
                        try:
                            alt_results = await self._execute_code_multi(alt_code)
                            if alt_results:
                                for alt_gt, alt_hash in alt_results:
                                    # Skip error answers and duplicates
                                    if not (
                                        isinstance(alt_gt, dict) and "error" in alt_gt
                                    ):
                                        if alt_hash not in answer_hashes:
                                            ground_truths.append(alt_gt)
                                            answer_hashes.append(alt_hash)
                            else:
                                print(f"    [alt {alt_idx + 1}] no result returned")
                        except Exception as alt_exc:
                            print(f"    [alt {alt_idx + 1}] FAILED: {alt_exc}")

                    n_alts = len(ground_truths) - 1
                    alt_suffix = f" (+{n_alts} alt)" if n_alts > 0 else ""

                    execution_results.append(
                        {
                            "template": template,
                            "params": params,
                            "code": code,
                            "ground_truth": ground_truth,  # Primary for verbalization
                            "ground_truths": ground_truths,  # All valid answers
                            "answer_hash": answer_hash,  # Primary hash
                            "answer_hashes": answer_hashes,  # All valid hashes
                        }
                    )
                    print(
                        f"  [{i + 1}/{len(expanded_templates)}] {template.name} ({params_label}) ✓{alt_suffix}"
                    )
                else:
                    print(
                        f"  [{i + 1}/{len(expanded_templates)}] {template.name} ({params_label}) - no answer"
                    )
            except Exception as e:
                print(
                    f"  [{i + 1}/{len(expanded_templates)}] {template.name} ({params_label}) FAILED: {e}"
                )

        # 4. Verbalize (or use mechanical questions)
        if config.synthetic_skip_verbalization:
            # Mechanical questions: use template description directly
            print(
                f"\nUsing mechanical questions for {len(execution_results)} templates..."
            )
            verbalized = []
            for item in execution_results:
                template = item["template"]
                # Build question from template description + output schema
                question_text = f"{template.description}. Return as JSON matching this schema: {template.output_schema}"
                verbalized.append(
                    {
                        "item": item,
                        "candidates": [
                            {
                                "candidate_index": 1,
                                "question": question_text,
                                "hint": "",  # No hint for mechanical questions
                                "raw_response": "[MECHANICAL]",
                            }
                        ],
                    }
                )
        else:
            # LLM verbalization
            print(f"\nVerbalizing {len(execution_results)} templates...")

            n_candidates = max(1, config.synthetic_verbalization_candidates)

            async def verbalize_candidates(item: dict) -> dict:
                """Generate multiple candidates for one template execution."""
                tasks = [
                    self.verbalizer.verbalize(
                        code=item["code"],
                        profile=profile,
                        ground_truth=item["ground_truth"],
                        output_schema=item["template"].output_schema,
                        data_overview=self.data_overview,
                        dataset_description=self.dataset_description,
                        banned_words=FORBIDDEN_METHOD_TERMS,
                    )
                    for _ in range(n_candidates)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                candidates = []
                for idx, result in enumerate(results, start=1):
                    if isinstance(result, Exception):
                        question_text = "[VERBALIZATION FAILED]"
                        hint = ""
                        raw_response = (
                            f"[VERBALIZATION ERROR: {type(result).__name__}: {result}]"
                        )
                    else:
                        question_text, hint, raw_response = result
                    candidates.append(
                        {
                            "candidate_index": idx,
                            "question": question_text,
                            "hint": hint,
                            "raw_response": raw_response,
                        }
                    )
                return {"item": item, "candidates": candidates}

            verbalized = await asyncio.gather(
                *[verbalize_candidates(item) for item in execution_results]
            )

        questions: list[dict] = []
        rejected: dict[str, int] = {}
        rejection_log: list[dict] = []

        # Incremental JSONL output files (for resume capability)
        accepted_jsonl = (
            output_path / "questions_accepted.jsonl" if output_path else None
        )
        rejected_jsonl = (
            output_path / "questions_rejected.jsonl" if output_path else None
        )
        episodes_jsonl = output_path / "episodes.jsonl" if output_path else None
        episodes_failed_jsonl = (
            output_path / "episodes_failed.jsonl" if output_path else None
        )

        # Load already-processed fingerprints for resume
        processed_fingerprints: set[str] = set()
        if accepted_jsonl and accepted_jsonl.exists():
            with open(accepted_jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    fp = entry.get("_fingerprint")
                    if fp:
                        processed_fingerprints.add(fp)
                    questions.append(entry)
            print(f"  Resuming: loaded {len(questions)} already-processed questions")
        if rejected_jsonl and rejected_jsonl.exists() and not retry_failed:
            with open(rejected_jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    fp = entry.get("_fingerprint")
                    if fp:
                        processed_fingerprints.add(fp)
            print(
                f"  Resuming: skipping {len(processed_fingerprints) - len(questions)} rejected questions"
            )
        elif retry_failed and rejected_jsonl and rejected_jsonl.exists():
            # Count how many will be retried
            with open(rejected_jsonl) as f:
                retry_count = sum(1 for _ in f)
            print(
                f"  Retry mode: will re-process {retry_count} previously rejected questions"
            )

        def record_rejection(
            item: dict,
            candidate: dict,
            reason: str,
            validation: dict | None = None,
            fingerprint: str | None = None,
        ) -> None:
            rejected[reason] = rejected.get(reason, 0) + 1
            entry = {
                "dataset": self.dataset_name,
                "template_name": item["template"].name,
                "template_params": item["params"] or None,
                "category": item["template"].category,
                "tags": item["template"].tags,
                "candidate_index": candidate.get("candidate_index"),
                "question": candidate.get("question"),
                "hint": candidate.get("hint"),
                "reason": reason,
                "validation": validation,
                "raw_response": candidate.get("raw_response"),
                "_fingerprint": fingerprint,
            }
            rejection_log.append(entry)
            # Incremental save
            if rejected_jsonl:
                with open(rejected_jsonl, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        for bundle in verbalized:
            item = bundle["item"]
            candidates = bundle["candidates"]

            # Compute fingerprint for this template+params combination
            # Include template CODE (not just name) so code changes trigger regeneration
            template = item["template"]
            fingerprint = hash_artifact(
                {
                    "template_name": template.name,
                    "template_code": template.code_template,
                    "alternative_codes": template.alternative_code_templates or [],
                    "params": item["params"],
                    "dataset": self.dataset_name,
                }
            )

            # Skip if already processed (resume capability)
            if fingerprint in processed_fingerprints:
                continue

            accepted = None
            attempts = 0
            for candidate in candidates:
                attempts += 1
                question_text = candidate.get("question") or ""
                if not question_text.strip():
                    record_rejection(
                        item, candidate, "empty_question", fingerprint=fingerprint
                    )
                    continue
                if question_text.startswith("[VERBALIZATION FAILED"):
                    record_rejection(
                        item, candidate, "verbalization_failed", fingerprint=fingerprint
                    )
                    continue

                # Skip viability filter for mechanical questions (they intentionally contain method details)
                is_mechanical = candidate.get("raw_response") == "[MECHANICAL]"
                if not is_mechanical:
                    is_ok, reason = _question_is_viable(question_text, profile)
                    if not is_ok:
                        record_rejection(
                            item, candidate, reason, fingerprint=fingerprint
                        )
                        continue

                validation_info = None
                validation_trace = None
                # Note: Hints are optional; verification handles them when present.
                # No pre-verification student validation gate (per design doc).

                accepted = {
                    "question": question_text,
                    "hint": candidate.get("hint"),
                    "n_steps": item["template"].n_steps,
                    "difficulty": item["template"].difficulty,
                    "category": item["template"].category,
                    "tags": item["template"].tags,
                    "template_name": item["template"].name,
                    "template_params": item["params"] or None,
                    "output_type": item["template"].output_type,
                    "output_schema": item["template"].output_schema,
                    "verbalization_attempts": attempts,
                    "candidate_index": candidate.get("candidate_index"),
                    "validation": validation_info,
                    # Primary answer (for backward compatibility)
                    "ground_truth_hash": item["answer_hash"],
                    "_ground_truth": item["ground_truth"],
                    # All valid answers (for multi-outcome validation)
                    "ground_truth_hashes": item["answer_hashes"],
                    "_ground_truths": item["ground_truths"],
                    "_template": item["template"].name,
                    "_fingerprint": fingerprint,
                    # Full trace for episode generation (validation IS the episode)
                    "_trace": validation_trace,
                }
                break

            if accepted:
                questions.append(accepted)
                # Incremental save question
                if accepted_jsonl:
                    # Save question without trace (trace goes to episodes)
                    question_entry = {
                        k: v for k, v in accepted.items() if k != "_trace"
                    }
                    with open(accepted_jsonl, "a") as f:
                        f.write(json.dumps(question_entry) + "\n")

                # Save episode with full trace (validation IS the episode)
                if episodes_jsonl and accepted.get("_trace"):
                    validation = accepted.get("validation", {})
                    elapsed = validation.get("elapsed", 0.0)

                    episode = EpisodeJSONL(
                        episode_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        csv_source=str(self.csv_path),
                        question=QADict(
                            id=accepted.get("_fingerprint", ""),
                            question_text=accepted["question"],
                            hint=accepted.get("hint"),
                            difficulty=accepted.get("difficulty"),
                            n_steps=accepted.get("n_steps"),
                            category=accepted.get("category"),
                            tags=accepted.get("tags"),
                            template_name=accepted.get("template_name"),
                            template_params=accepted.get("template_params"),
                            output_type=accepted.get("output_type"),
                            output_schema=accepted.get("output_schema"),
                            ground_truth=accepted.get("_ground_truth"),
                            ground_truth_hash=accepted.get("ground_truth_hash"),
                            ground_truth_hashes=accepted.get("ground_truth_hashes"),
                        ),
                        # For synthetic: validation trace IS the gold trace
                        # (LLM solved without hint, matched ground truth)
                        gold_trace=accepted["_trace"],
                        consistency_traces=[],  # No triangulation needed - we have deterministic ground truth
                        verified=True,  # Verified by ground truth match
                        triangulation=TriangulationMetadataDict(
                            n_consistency_runs=0,
                            n_consistency_succeeded=0,
                            majority_answer_hash=accepted.get("ground_truth_hash"),
                            majority_count=1,
                            gold_matches_majority=True,
                        ),
                        timing=TimingMetadataDict(
                            gold_elapsed=elapsed,
                            consistency_elapsed=[],
                            total_elapsed=elapsed,
                            avg_elapsed=elapsed,
                        ),
                        source="synthetic",
                    )
                    with open(episodes_jsonl, "a") as f:
                        f.write(episode.model_dump_json() + "\n")

        if rejected:
            print("  Question filter rejections:")
            for reason, count in sorted(rejected.items(), key=lambda x: (-x[1], x[0])):
                print(f"    - {reason}: {count}")

        print(
            f"  Generated {len(questions)} questions from {len(execution_results)} templates"
        )

        # 5. Format output
        df = pd.read_csv(self.csv_path, nrows=0)
        output = {
            "dataset_columns": df.columns.tolist(),
            "questions": questions,
            "rejections": rejection_log,
        }

        print(f"\nGenerated {len(questions)} questions")
        return output

    async def _execute_code(self, code: str) -> tuple[Any, str] | None:
        """
        Execute code in sandbox and extract answer(s).

        Returns:
            Tuple of (ground_truth_value, hash) or None if failed.
            For backward compatibility, returns just the first answer.
            Use _execute_code_multi for multiple answers.
        """
        result = await self._execute_code_multi(code)
        if not result:
            return None
        return result[0]  # Return first (primary) answer

    async def _execute_code_multi(self, code: str) -> list[tuple[Any, str]] | None:
        """
        Execute code in sandbox and extract ALL submitted answers.

        Templates can call submit() multiple times for tie-aware enumeration.
        Each submission is captured and hashed separately.

        Returns:
            List of (ground_truth_value, hash) tuples, or None if no answers
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

        # Parse ALL submitted answers
        answers = self._parse_all_submissions(output)
        if not answers:
            print(f"  No answer submitted. Output: {output[:200]}")
            return None

        # Hash each answer
        results = []
        seen_hashes = set()
        for answer in answers:
            answer_hash = hash_artifact(answer)
            # Deduplicate by hash
            if answer_hash not in seen_hashes:
                seen_hashes.add(answer_hash)
                results.append((answer, answer_hash))

        return results if results else None

    def _parse_all_submissions(self, output: str) -> list[Any]:
        """Extract ALL submitted answers from execution output.

        Templates can call submit() multiple times for tie-aware enumeration.
        Returns list of all valid answers found.
        """
        marker = "✓ Submitted: "
        answers = []

        # Find all occurrences of the marker
        pos = 0
        while True:
            idx = output.find(marker, pos)
            if idx == -1:
                break

            start = idx + len(marker)
            end = output.find("\n", start)
            if end == -1:
                json_str = output[start:]
            else:
                json_str = output[start:end]

            try:
                submission = json.loads(json_str.strip())
                answer = submission.get("__csv_agent_answer__")
                if answer is not None:
                    answers.append(answer)
            except json.JSONDecodeError:
                pass

            pos = start

        return answers

    def _parse_submission(self, output: str) -> Any | None:
        """Extract first submitted answer from execution output (backward compat)."""
        answers = self._parse_all_submissions(output)
        return answers[0] if answers else None


def load_dataset_description(csv_path: str) -> str:
    """Load dataset description from meta.json adjacent to CSV."""
    csv_path_obj = Path(csv_path)

    # Try sibling meta.json first (for data/kaggle/{slug}/data.csv structure)
    meta_path = csv_path_obj.parent / "meta.json"
    if not meta_path.exists():
        # Try sidecar format (data.meta.json)
        meta_path = csv_path_obj.with_suffix(".meta.json")

    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                return (
                    meta.get("description")
                    or meta.get("subtitle")
                    or meta.get("title")
                    or ""
                )
        except Exception:
            pass

    return ""


async def generate_questions(
    csv_path: str,
    output_dir: str | None = None,
    n_questions: int | None = None,
    model: str | None = None,
    retry_failed: bool = False,
) -> dict:
    """
    Main entry point for generating compositional questions.

    Args:
        csv_path: Path to CSV dataset
        output_dir: Directory to save questions.json (defaults to data/questions/{dataset_name}/)
        n_questions: Max questions to generate
        model: LLM model for verbalization
        retry_failed: If True, re-process previously failed questions

    Returns:
        Generated questions dict
    """
    # Load dataset description from meta.json
    dataset_description = load_dataset_description(csv_path)
    if dataset_description:
        print(f"Dataset description: {dataset_description[:100]}...")

    generator = CompositionalQuestionGenerator(
        csv_path=csv_path,
        model=model,
        dataset_description=dataset_description,
    )

    # Prepare output path early for incremental saving
    if output_dir is None:
        dataset_name = Path(csv_path).parent.name
        output_dir = f"{config.questions_synthetic_dir}/{dataset_name}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        await generator.setup()
        result = await generator.generate(
            n_questions=n_questions, output_path=output_path, retry_failed=retry_failed
        )

        # Save final output (combines any resumed data with new)
        questions_file = output_path / "questions.json"

        with open(questions_file, "w") as f:
            # Keep ground_truth and template for evaluation, but mark as internal
            clean_questions = []
            for q in result["questions"]:
                clean_q = {
                    k: v
                    for k, v in q.items()
                    if k in ("_ground_truth", "_ground_truths", "_template")
                    or not k.startswith("_")
                }
                clean_questions.append(clean_q)

            output = {
                "dataset_columns": result["dataset_columns"],
                "questions": clean_questions,
            }
            json.dump(output, f, indent=2)

        rejections = result.get("rejections", [])
        if rejections:
            rejection_file = output_path / "questions_rejected.jsonl"
            with open(rejection_file, "w") as f:
                for entry in rejections:
                    f.write(json.dumps(entry) + "\n")
            print(f"Saved rejected questions to: {rejection_file}")

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
    parser.add_argument(
        "--gui-progress",
        type=str,
        default=None,
        help="Path to write progress JSON for GUI polling",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Limit number of datasets to process (for testing)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-process previously failed questions (ignores rejected fingerprints)",
    )

    args = parser.parse_args()

    # Setup progress writer for GUI
    if args.gui_progress:
        progress = ProgressWriter(
            output_path=args.gui_progress, stage="synthetic_generator"
        )
    else:
        progress = NoOpProgressWriter()

    # Auto-discover datasets if none specified
    csv_paths = args.csv if args.csv else config.csv_sources
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    # Limit datasets for testing
    if args.max_datasets and len(csv_paths) > args.max_datasets:
        csv_paths = csv_paths[: args.max_datasets]

    if not csv_paths:
        print("No CSV files found. Specify --csv or ensure data/kaggle/ has datasets.")
        return 1

    total_questions = 0
    failed_csvs = []

    # Initialize progress tracking for each dataset
    for csv_path in csv_paths:
        dataset_name = (
            Path(csv_path).parent.name
            if Path(csv_path).name == "data.csv"
            else Path(csv_path).stem
        )
        progress.set_dataset(dataset_name, 1)  # 1 unit of work per dataset

    for csv_path in csv_paths:
        dataset_name = (
            Path(csv_path).parent.name
            if Path(csv_path).name == "data.csv"
            else Path(csv_path).stem
        )
        progress.set_current(dataset_name)
        progress.log(f"Processing: {dataset_name}")

        print(f"\n{'=' * 60}")
        print(f"Processing: {csv_path}")
        print("=" * 60)

        try:
            result = asyncio.run(
                generate_questions(
                    csv_path=csv_path,
                    output_dir=args.output_dir,
                    n_questions=args.n_questions,
                    model=args.model,
                    retry_failed=args.retry_failed,
                )
            )
            total_questions += len(result["questions"])
            print(f"Generated {len(result['questions'])} questions")
            progress.update_dataset(
                dataset_name, done=1, verified=len(result["questions"])
            )
            progress.log(f"✓ {dataset_name}: {len(result['questions'])} questions")

        except KeyboardInterrupt:
            print("\nInterrupted")
            progress.fail("Interrupted by user")
            return 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed_csvs.append(csv_path)
            progress.update_dataset(dataset_name, done=1, failed=1)
            progress.log(f"✗ {dataset_name}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print(
        f"COMPLETE: {total_questions} questions from {len(csv_paths) - len(failed_csvs)} datasets"
    )
    if failed_csvs:
        print(f"Failed: {len(failed_csvs)} datasets")
        for f in failed_csvs:
            print(f"  - {f}")

    progress.log(
        f"Complete: {total_questions} questions from {len(csv_paths) - len(failed_csvs)} datasets"
    )
    progress.complete()
    # Success if we generated any questions (some datasets may fail due to encoding, etc.)
    return 0 if total_questions > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
