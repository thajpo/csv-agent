"""
End-to-end pipeline tests.

Two test modes:
1. Deterministic fixture test - validates types and pipeline contract (always runs)
2. Live integration test - hits real API (requires API key, marked slow)
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio


FIXTURES_DIR = Path(__file__).parent / "fixtures"


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


class TestPipelineContract:
    """
    Deterministic tests using static fixtures.

    These tests validate:
    1. Type contracts - EpisodeJSONL schema is correct
    2. Serialization - episodes can be round-tripped to/from JSON
    3. Evaluation - synthetic evaluator works with fixture data

    If you change the schema, you MUST update the fixture.
    """

    @pytest.fixture
    def expected_episode_path(self):
        """Path to the expected episode fixture."""
        return FIXTURES_DIR / "expected_episode.json"

    def test_fixture_validates_against_schema(self, expected_episode_path):
        """
        Validate that fixture matches EpisodeJSONL schema.

        This is a CONTRACT TEST: if the schema changes, this test fails
        and you must update the fixture to match the new schema.
        """
        from src.core.types import EpisodeJSONL

        with open(expected_episode_path) as f:
            data = json.load(f)

        # This will raise ValidationError if schema mismatches
        episode = EpisodeJSONL(**data)

        # Verify key fields
        assert episode.episode_id == "test-episode-001"
        assert episode.verified is True
        assert episode.gold_trace["success"] is True
        assert len(episode.consistency_traces) == 2
        assert episode.triangulation["n_consistency_runs"] == 2

    def test_episode_roundtrip_serialization(self, expected_episode_path):
        """Test that episodes can be serialized and deserialized."""
        from src.core.types import EpisodeJSONL

        with open(expected_episode_path) as f:
            data = json.load(f)

        episode = EpisodeJSONL(**data)

        # Round-trip through JSON
        json_str = episode.model_dump_json()
        reloaded = EpisodeJSONL.model_validate_json(json_str)

        assert reloaded.episode_id == episode.episode_id
        assert reloaded.verified == episode.verified
        assert reloaded.gold_trace == episode.gold_trace
        assert len(reloaded.consistency_traces) == len(episode.consistency_traces)

    def test_episode_jsonl_format(self, expected_episode_path):
        """Test that episodes work in JSONL format (one per line)."""
        from src.core.types import EpisodeJSONL

        with open(expected_episode_path) as f:
            data = json.load(f)

        episode = EpisodeJSONL(**data)

        # Write to JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(episode.model_dump_json() + "\n")
            temp_path = f.name

        try:
            # Read back
            with open(temp_path) as f:
                line = f.readline()
                reloaded = EpisodeJSONL.model_validate_json(line)

            assert reloaded.episode_id == episode.episode_id
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_synthetic_evaluator_with_fixture(self, expected_episode_path):
        """Test that SyntheticEvaluator works with fixture data."""
        from src.core.types import EpisodeJSONL
        from src.eval.synthetic import SyntheticEvaluator

        with open(expected_episode_path) as f:
            data = json.load(f)

        episode = EpisodeJSONL(**data)

        # Write to temp JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(episode.model_dump_json() + "\n")
            temp_path = f.name

        try:
            evaluator = SyntheticEvaluator(temp_path)
            metrics = await evaluator.evaluate_async()

            assert metrics.total == 1
            # Note: accuracy depends on ground_truth matching final_answer
        finally:
            os.unlink(temp_path)


class TestSyntheticQuestionGen:
    """Tests for synthetic question generation (no API calls)."""

    @pytest.fixture
    def test_csv(self):
        """Use insurance dataset - small and has good numeric columns."""
        return "data/kaggle/mirichoi0218_insurance/data.csv"

    def test_profiler_analyzes_dataset(self, test_csv):
        """Test that DataProfiler works."""
        from src.datagen.synthetic.profiler import DataProfiler

        profiler = DataProfiler()
        profile = profiler.analyze(test_csv)

        assert profile is not None
        assert "columns" in profile
        assert profile["shape"]["rows"] > 0
        assert profile["shape"]["columns"] > 0

    def test_templates_are_applicable(self, test_csv):
        """Test that templates can be selected for a dataset."""
        from src.datagen.synthetic.profiler import DataProfiler
        from src.datagen.synthetic.templates import get_applicable_templates

        profiler = DataProfiler()
        profile = profiler.analyze(test_csv)
        templates = get_applicable_templates(profile)

        assert len(templates) > 0
        # Should have easy templates
        easy = [t for t in templates if t.difficulty == "EASY"]
        assert len(easy) > 0

    @pytest.mark.asyncio
    async def test_template_execution_produces_answer(self, test_csv):
        """Test that templates execute and produce answers."""
        from src.datagen.synthetic.profiler import DataProfiler
        from src.datagen.synthetic.templates import get_applicable_templates
        from src.envs.csv_env import LocalCSVAnalysisEnv
        import json

        profiler = DataProfiler()
        profile = profiler.analyze(test_csv)
        templates = get_applicable_templates(profile)

        # Pick first easy template
        easy_templates = [t for t in templates if t.difficulty == "EASY"]
        template = easy_templates[0]

        # Get params and instantiate
        param_sets = list(template.iter_param_sets())
        params = param_sets[0] if param_sets else {}
        code = template.instantiate(profile, params=params)

        # Execute
        env = LocalCSVAnalysisEnv(csv_path=test_csv)
        state = await env.setup_state({})
        try:
            output = await env.python(
                code=code,
                sandbox_id=state["sandbox_id"],
                python_state=state["python_state"],
            )
        finally:
            await env.destroy_sandbox(state["sandbox_id"])

        # Verify submission
        marker = "✓ Submitted: "
        assert marker in output, f"Template didn't submit: {output[:500]}"

        start = output.index(marker) + len(marker)
        end = output.find("\n", start)
        json_str = output[start:end] if end != -1 else output[start:]
        submission = json.loads(json_str)

        assert "__csv_agent_answer__" in submission
        assert submission["__csv_agent_answer__"] is not None


# Skip live tests if no API key
_has_api_key = bool(
    os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
)


@pytest.mark.skipif(not _has_api_key, reason="No LLM API key found")
@pytest.mark.slow
class TestPipelineLive:
    """
    Live integration tests that hit real APIs.

    These tests are:
    - Slow (minutes)
    - Non-deterministic (LLM output varies)
    - Expensive (API costs)

    Run with: pytest -m slow
    """

    @pytest.fixture
    def test_csv(self):
        return "data/kaggle/mirichoi0218_insurance/data.csv"

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 min max
    async def test_triangulation_live(self, test_csv, temp_dir):
        """
        Live triangulation test - calls actual LLM API.

        This test may fail due to LLM non-determinism - that's expected.
        It's here for manual verification, not CI.
        """
        from src.datagen.synthetic.profiler import DataProfiler
        from src.datagen.synthetic.templates import get_applicable_templates
        from src.datagen.teacher import triangulate_teacher
        from src.core.prompts import generate_data_overview

        # Generate a question
        profiler = DataProfiler()
        profile = profiler.analyze(test_csv)
        templates = get_applicable_templates(profile)
        easy = [t for t in templates if t.difficulty == "EASY"]
        template = easy[0]

        question = f"[TEST] {template.name}: Analyze the dataset."
        hint = "Follow the template logic."

        data_overview = generate_data_overview(test_csv)

        result = await triangulate_teacher(
            csv_path=test_csv,
            question=question,
            hint=hint,
            model="openai/gpt-4o-mini",
            n_consistency=2,
            dataset_description="Insurance dataset",
            data_overview=data_overview,
            max_turns=5,
            sampling_args={"temperature": 0.7, "max_tokens": 2000},
            float_tol=0.1,
            n_steps=template.n_steps,
            difficulty=template.difficulty,
            ui=QuietUI(),
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

        # These assertions may fail due to LLM non-determinism
        # That's okay - this is a smoke test
        print(f"Gold success: {gold_trace['success']}")
        print(f"Verified: {verified}")
        print(f"Consistency results: {len(consistency_results)}")

        assert len(consistency_results) == 2, "Should have 2 consistency traces"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
