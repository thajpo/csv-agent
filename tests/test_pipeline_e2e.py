"""
End-to-end pipeline tests.

Two test modes:
1. Deterministic fixture test - validates types and pipeline contract (always runs)
2. Live integration test - hits real API (requires API key, marked slow)
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.core.config import config


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
        from csv_spec import EpisodeJSONL

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
        from csv_spec import EpisodeJSONL

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
        from csv_spec import EpisodeJSONL

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
        from csv_spec import EpisodeJSONL
        from src.eval.synthetic_eval import SyntheticEvaluator

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


class TestPromptContract:
    """Verify prompt structure doesn't accidentally change."""

    def test_teacher_tutor_prompt_structure(self):
        from src.core.prompts import build_system_prompt
        from csv_spec import Question

        question = Question(
            question_text="What is the mean?",
            hint="Calculate mean",
            n_steps=3,
            difficulty="EASY",
        )
        prompt = build_system_prompt(
            mode="teacher-tutor",
            dataset_description="Test dataset",
            data_overview="shape: (100, 5)",
            question=question,
        )

        assert "HOOKS REQUIRED" in prompt
        assert "hook(" in prompt
        assert "submit(" in prompt
        assert "What is the mean?" in prompt
        assert "Calculate mean" in prompt

    def test_teacher_consistency_prompt_has_no_hint(self):
        from src.core.prompts import build_system_prompt
        from csv_spec import Question

        question = Question(
            question_text="What is the mean?",
            hint="Calculate mean",
            n_steps=3,
        )
        prompt = build_system_prompt(
            mode="teacher-consistency",
            dataset_description="Test dataset",
            data_overview="shape: (100, 5)",
            question=question,
        )

        assert "What is the mean?" in prompt
        assert "Calculate mean" not in prompt


class TestTriangulationLogic:
    """
    Test triangulation orchestration with FakeLLM.

    Uses real Docker but mocked LLM to test:
    - Majority voting logic
    - Trace building
    - Answer matching with tolerance
    """

    @pytest.fixture
    def test_csv(self):
        return "data/csv/data.csv"

    @pytest.mark.skip(
        reason="FakeLLM integration with full triangulation is complex - use live test"
    )
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_triangulation_with_fake_llm(self, test_csv, fake_llm_factory):
        pass

    @pytest.mark.asyncio
    async def test_majority_voting_logic(self):
        from src.datagen.teacher import get_majority_answer, answers_match

        answers = [42.0, 42.05, 42.08, 100.0]
        majority, count = get_majority_answer(answers, float_tol=0.1)

        assert count == 3
        assert answers_match(None, None, majority, 42.0, float_tol=0.1)

    @pytest.mark.asyncio
    async def test_answers_match_tolerance(self):
        from src.datagen.teacher import answers_match

        assert answers_match(None, None, 42.0, 42.05, float_tol=0.1)
        assert not answers_match(None, None, 42.0, 43.0, float_tol=0.1)
        assert answers_match(None, None, {"a": 1.0}, {"a": 1.05}, float_tol=0.1)


@pytest.mark.live
class TestAPISmoke:
    """Minimal API integration test. Run with: pytest --run-live"""

    @pytest.mark.asyncio
    async def test_api_client_works(self):
        """Verify LLM API client works - auth, request format, response parsing."""
        from src.core.model import APILLM

        llm = APILLM(
            model=config.teacher_model,
            sampling_args={"temperature": 0, "max_tokens": 10},
        )
        try:
            response = await llm("Reply with exactly one word: PONG")
            assert "PONG" in response.upper()
        finally:
            await llm.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
