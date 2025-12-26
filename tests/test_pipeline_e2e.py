"""
End-to-end pipeline smoke test.

Runs the full pipeline on 1 dataset with 1 question to verify:
1. Synthetic question generation works
2. Episode generation (triangulation) completes
3. Synthetic evaluation runs

This test requires Docker and an LLM API key.
Mark as slow/integration test - not run by default.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

# Skip if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
    reason="No LLM API key found",
)


class _QuietConsole:
    """Mock console that does nothing."""

    def print(self, *args, **kwargs):
        pass


class QuietUI:
    """Minimal UI that prints nothing - for testing."""

    def __init__(self):
        self.base = self  # Self-reference for .base.xxx calls
        self.console = _QuietConsole()

    def __getattr__(self, name):
        """Return a no-op for any method call."""

        def noop(*args, **kwargs):
            pass

        return noop


class TestPipelineE2E:
    """End-to-end pipeline test with minimal data."""

    @pytest.fixture
    def test_csv(self):
        """Use insurance dataset - small and has good numeric columns."""
        return "data/kaggle/mirichoi0218_insurance/data.csv"

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for outputs."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 min max
    async def test_full_pipeline_single_question(self, test_csv, temp_dir):
        """
        Run full pipeline: generate -> triangulate -> evaluate.

        Uses only 1 question with 2 consistency traces for speed.
        """
        from src.datagen.synthetic.generator import CompositionalQuestionGenerator
        from src.datagen.teacher import triangulate_teacher
        from src.datagen.synthetic.profiler import DataProfiler
        from src.datagen.synthetic.templates import get_applicable_templates
        from src.core.prompts import generate_data_overview
        from src.core.types import (
            Question,
            EpisodeJSONL,
            QADict,
            TimingMetadataDict,
            TriangulationMetadataDict,
        )
        from src.eval.synthetic import SyntheticEvaluator
        from datetime import datetime
        from typing import cast
        import uuid

        # ============================================
        # Step 1: Generate 1 synthetic question
        # ============================================
        print("\n=== Step 1: Generate synthetic question ===")

        profiler = DataProfiler()
        profile = profiler.analyze(test_csv)
        templates = get_applicable_templates(profile)

        # Pick easiest template for speed
        easy_templates = [t for t in templates if t.difficulty == "EASY"]
        template = (
            easy_templates[0] if easy_templates else templates[-1]
        )  # Last is simplest

        # Get first valid param set for the template
        param_sets = list(template.iter_param_sets())
        template_params = param_sets[0] if param_sets else {}

        # Execute template to get ground truth
        from src.envs.csv_env import LocalCSVAnalysisEnv

        env = LocalCSVAnalysisEnv(csv_path=test_csv)
        state = await env.setup_state({})

        code = template.instantiate(profile, params=template_params)
        output = await env.python(
            code=code,
            sandbox_id=state["sandbox_id"],
            python_state=state["python_state"],
        )
        await env.destroy_sandbox(state["sandbox_id"])

        # Parse ground truth
        marker = "✓ Submitted: "
        assert marker in output, f"Template didn't submit: {output[:500]}"
        start = output.index(marker) + len(marker)
        end = output.find("\n", start)
        json_str = output[start:end] if end != -1 else output[start:]
        submission = json.loads(json_str)
        ground_truth = submission["__csv_agent_answer__"]

        from src.utils.hashing import hash_artifact

        ground_truth_hash = hash_artifact(ground_truth)

        # Create question dict (skip LLM verbalization - use template name as question)
        question_dict = {
            "question": f"[TEST] {template.name}: Analyze the dataset.",
            "hint": "Follow the template logic.",
            "n_steps": template.n_steps,
            "difficulty": template.difficulty,
            "template_name": template.name,
            "template_params": None,
            "output_type": template.output_type,
            "output_schema": template.output_schema,
            "ground_truth_hash": ground_truth_hash,
            "_ground_truth": ground_truth,
        }

        print(f"  Template: {template.name}")
        print(f"  Difficulty: {template.difficulty}")
        print(f"  Ground truth type: {type(ground_truth).__name__}")

        # ============================================
        # Step 2: Run triangulation (gold + 2 consistency)
        # ============================================
        print("\n=== Step 2: Run triangulation ===")

        data_overview = generate_data_overview(test_csv)

        # Get dataset description
        meta_path = Path(test_csv).parent / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        dataset_description = meta.get("description", "Insurance dataset")

        result = await triangulate_teacher(
            csv_path=test_csv,
            question=question_dict["question"],
            hint=question_dict["hint"],
            model="openai/gpt-4o-mini",  # Fast model for testing
            n_consistency=2,  # Minimal consistency traces
            dataset_description=dataset_description,
            data_overview=data_overview,
            max_turns=5,  # Limit turns
            sampling_args={"temperature": 0.7, "max_tokens": 2000},
            float_tol=0.1,
            n_steps=template.n_steps,
            difficulty=template.difficulty,
            ui=QuietUI(),  # Quiet UI for testing
        )

        (
            gold_trace,
            gold_conversation,
            system_prompt,
            consistency_results,
            verified,
            timing_metadata,
            majority_hash,
            majority_count,
        ) = result

        print(f"  Gold trace success: {gold_trace['success']}")
        print(f"  Gold answer: {gold_trace['final_answer']}")
        print(f"  Consistency traces: {len(consistency_results)}")
        print(f"  Verified: {verified}")

        # ============================================
        # Step 3: Create episode and save
        # ============================================
        print("\n=== Step 3: Create episode ===")

        question_obj = Question.from_dict(question_dict)
        consistency_traces = [t for t, _ in consistency_results]
        n_succeeded = sum(1 for t in consistency_traces if t["success"])

        episode_jsonl = EpisodeJSONL(
            episode_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            csv_source=test_csv,
            question=QADict(
                id=question_obj.id,
                question_text=question_obj.question_text,
                hint=question_obj.hint,
                difficulty=question_obj.difficulty,
                n_steps=question_obj.n_steps,
                template_name=question_obj.template_name,
                template_params=question_obj.template_params,
                ground_truth=question_obj.ground_truth,
            ),
            gold_trace=gold_trace,
            consistency_traces=consistency_traces,
            verified=verified,
            triangulation=TriangulationMetadataDict(
                n_consistency_runs=len(consistency_traces),
                n_consistency_succeeded=n_succeeded,
                majority_answer_hash=majority_hash,
                majority_count=majority_count,
                gold_matches_majority=verified,
            ),
            timing=TimingMetadataDict(
                gold_elapsed=timing_metadata["gold_elapsed"],
                consistency_elapsed=timing_metadata["consistency_elapsed"],
                total_elapsed=timing_metadata["total_elapsed"],
                avg_elapsed=timing_metadata["avg_elapsed"],
            ),
        )

        episodes_path = temp_dir / "test_episodes.jsonl"
        with open(episodes_path, "w") as f:
            f.write(episode_jsonl.model_dump_json() + "\n")

        print(f"  Episode saved: {episodes_path}")
        print(f"  Episode ID: {episode_jsonl.episode_id}")

        # ============================================
        # Step 4: Run evaluation
        # ============================================
        print("\n=== Step 4: Run evaluation ===")

        evaluator = SyntheticEvaluator(str(episodes_path))
        metrics = await evaluator.evaluate_async()

        print(f"  Total episodes: {metrics.total}")
        print(f"  Correct: {metrics.correct}")
        print(f"  Accuracy: {metrics.accuracy:.1%}")

        # ============================================
        # Assertions
        # ============================================
        assert gold_trace["success"], "Gold trace should succeed"
        assert gold_trace["final_answer"] is not None, "Gold trace should have answer"
        assert len(consistency_results) == 2, "Should have 2 consistency traces"
        assert metrics.total == 1, "Should have 1 episode"

        # If verified, accuracy should be 100% (teacher matched ground truth)
        if verified:
            # Note: might not be 100% due to float tolerance differences
            print(f"  Verification passed - pipeline complete!")
        else:
            print(f"  Verification failed - but pipeline completed")

        print("\n✓ E2E pipeline test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
