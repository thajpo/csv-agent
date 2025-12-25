"""
Tests for synthetic question generation pipeline.

Tests template instantiation, execution, and evaluation logic
without requiring LLM calls.
"""

import asyncio
import json
import pytest
import pytest_asyncio

from src.datagen.synthetic.profiler import DataProfiler
from src.datagen.synthetic.templates import (
    ALL_TEMPLATES,
    get_applicable_templates,
    MAX_VARIANCE_MEAN,
    STRONGEST_CORRELATION,
    COUNT_HIGH_MISSING_COLUMNS,
)
from src.envs.csv_env import LocalCSVAnalysisEnv


class TestDataProfiler:
    """Test dataset profiling."""

    def test_profile_basic_csv(self):
        """Profiler should analyze a CSV and return structured profile."""
        profiler = DataProfiler()
        profile = profiler.analyze("data/csv/data.csv")

        assert "shape" in profile
        assert profile["shape"]["rows"] > 0
        assert profile["shape"]["columns"] > 0
        assert "columns" in profile
        assert len(profile["columns"]) == profile["shape"]["columns"]

    def test_profile_identifies_numeric_columns(self):
        """Profiler should identify numeric columns."""
        profiler = DataProfiler()
        profile = profiler.analyze("data/csv/data.csv")

        numeric_cols = [
            name
            for name, info in profile["columns"].items()
            if info.get("type") == "numeric"
        ]
        assert len(numeric_cols) > 0

    def test_profile_kaggle_dataset(self):
        """Profiler should work on Kaggle datasets."""
        profiler = DataProfiler()
        profile = profiler.analyze("data/kaggle/mirichoi0218_insurance/data.csv")

        assert profile["shape"]["rows"] > 0
        # Insurance dataset should have numeric columns like age, bmi, charges
        numeric_cols = [
            name
            for name, info in profile["columns"].items()
            if info.get("type") == "numeric"
        ]
        assert len(numeric_cols) >= 3


class TestTemplateApplicability:
    """Test template applicability checks."""

    @pytest.fixture
    def insurance_profile(self):
        """Profile for insurance dataset (has numeric + categorical)."""
        profiler = DataProfiler()
        return profiler.analyze("data/kaggle/mirichoi0218_insurance/data.csv")

    @pytest.fixture
    def netflix_profile(self):
        """Profile for Netflix dataset (mostly categorical)."""
        profiler = DataProfiler()
        return profiler.analyze("data/kaggle/shivamb_netflix-shows/data.csv")

    def test_max_variance_mean_applicable(self, insurance_profile):
        """MAX_VARIANCE_MEAN needs 2+ numeric columns."""
        assert MAX_VARIANCE_MEAN.is_applicable(insurance_profile)

    def test_strongest_correlation_applicable(self, insurance_profile):
        """STRONGEST_CORRELATION needs 3+ numeric columns."""
        assert STRONGEST_CORRELATION.is_applicable(insurance_profile)

    def test_count_high_missing_always_applicable(
        self, insurance_profile, netflix_profile
    ):
        """COUNT_HIGH_MISSING_COLUMNS should always be applicable."""
        assert COUNT_HIGH_MISSING_COLUMNS.is_applicable(insurance_profile)
        assert COUNT_HIGH_MISSING_COLUMNS.is_applicable(netflix_profile)

    def test_get_applicable_templates_returns_sorted(self, insurance_profile):
        """get_applicable_templates should return templates sorted by n_steps desc."""
        templates = get_applicable_templates(insurance_profile)
        assert len(templates) > 0

        # Should be sorted by n_steps descending (hardest first)
        n_steps = [t.n_steps for t in templates]
        assert n_steps == sorted(n_steps, reverse=True)

    def test_fewer_templates_for_limited_dataset(self, netflix_profile):
        """Netflix has few numeric columns, so fewer templates apply."""
        insurance_profile = DataProfiler().analyze(
            "data/kaggle/mirichoi0218_insurance/data.csv"
        )

        netflix_templates = get_applicable_templates(netflix_profile)
        insurance_templates = get_applicable_templates(insurance_profile)

        # Netflix should have fewer applicable templates
        assert len(netflix_templates) < len(insurance_templates)


class TestTemplateInstantiation:
    """Test template code generation."""

    @pytest.fixture
    def profile(self):
        profiler = DataProfiler()
        return profiler.analyze("data/kaggle/mirichoi0218_insurance/data.csv")

    def test_instantiate_produces_valid_code(self, profile):
        """Instantiated template should be valid Python."""
        code = MAX_VARIANCE_MEAN.instantiate(profile)

        # Should contain key elements
        assert "hook(" in code
        assert "submit(" in code
        assert "df" in code  # References the dataframe

    def test_instantiate_with_params(self, profile):
        """Templates with params should substitute them."""
        code = COUNT_HIGH_MISSING_COLUMNS.instantiate(
            profile, params={"missing_threshold": 10.0}
        )

        assert "10.0" in code

    def test_all_templates_instantiate(self, profile):
        """All applicable templates should instantiate without error."""
        templates = get_applicable_templates(profile)

        for template in templates:
            for params in template.iter_param_sets():
                code = template.instantiate(profile, params=params)
                assert len(code) > 0
                assert "submit(" in code


@pytest_asyncio.fixture
async def sandbox():
    """Create a sandbox environment for template execution."""
    env = LocalCSVAnalysisEnv(csv_path="data/kaggle/mirichoi0218_insurance/data.csv")
    state = await env.setup_state({})
    yield env, state
    await env.destroy_sandbox(state["sandbox_id"])


class TestTemplateExecution:
    """Test template execution in sandbox."""

    @pytest.mark.asyncio
    async def test_execute_max_variance_mean(self, sandbox):
        """MAX_VARIANCE_MEAN should execute and produce a result."""
        env, state = sandbox
        profiler = DataProfiler()
        profile = profiler.analyze("data/kaggle/mirichoi0218_insurance/data.csv")

        code = MAX_VARIANCE_MEAN.instantiate(profile)
        output = await env.python(
            code=code,
            sandbox_id=state["sandbox_id"],
            python_state=state["python_state"],
        )

        # Should have submitted an answer
        assert "✓ Submitted:" in output

        # Parse the submission
        marker = "✓ Submitted: "
        start = output.index(marker) + len(marker)
        end = output.find("\n", start)
        json_str = output[start:end] if end != -1 else output[start:]
        submission = json.loads(json_str)

        assert "__csv_agent_answer__" in submission
        answer = submission["__csv_agent_answer__"]
        assert isinstance(answer, (int, float))

    @pytest.mark.asyncio
    async def test_execute_strongest_correlation(self, sandbox):
        """STRONGEST_CORRELATION should return dict with columns and correlation."""
        env, state = sandbox
        profiler = DataProfiler()
        profile = profiler.analyze("data/kaggle/mirichoi0218_insurance/data.csv")

        code = STRONGEST_CORRELATION.instantiate(profile)

        # Reset state for clean execution
        await env.reset(state["sandbox_id"], state["python_state"])

        output = await env.python(
            code=code,
            sandbox_id=state["sandbox_id"],
            python_state=state["python_state"],
        )

        assert "✓ Submitted:" in output

        # Parse and validate structure
        marker = "✓ Submitted: "
        start = output.index(marker) + len(marker)
        end = output.find("\n", start)
        json_str = output[start:end] if end != -1 else output[start:]
        submission = json.loads(json_str)

        answer = submission["__csv_agent_answer__"]
        assert "columns" in answer
        assert "correlation" in answer
        assert len(answer["columns"]) == 2
        assert 0 <= answer["correlation"] <= 1

    @pytest.mark.asyncio
    async def test_execute_count_high_missing(self, sandbox):
        """COUNT_HIGH_MISSING_COLUMNS should return dict with count and columns."""
        env, state = sandbox
        profiler = DataProfiler()
        profile = profiler.analyze("data/kaggle/mirichoi0218_insurance/data.csv")

        code = COUNT_HIGH_MISSING_COLUMNS.instantiate(
            profile, params={"missing_threshold": 5.0}
        )

        await env.reset(state["sandbox_id"], state["python_state"])

        output = await env.python(
            code=code,
            sandbox_id=state["sandbox_id"],
            python_state=state["python_state"],
        )

        assert "✓ Submitted:" in output

        marker = "✓ Submitted: "
        start = output.index(marker) + len(marker)
        end = output.find("\n", start)
        json_str = output[start:end] if end != -1 else output[start:]
        submission = json.loads(json_str)

        answer = submission["__csv_agent_answer__"]
        assert "count" in answer
        assert "columns" in answer
        assert isinstance(answer["count"], int)
        assert isinstance(answer["columns"], list)

    @pytest.mark.asyncio
    async def test_all_applicable_templates_execute(self, sandbox):
        """All applicable templates should execute without error."""
        env, state = sandbox
        profiler = DataProfiler()
        profile = profiler.analyze("data/kaggle/mirichoi0218_insurance/data.csv")

        templates = get_applicable_templates(profile)
        failed = []

        for template in templates:
            for params in template.iter_param_sets():
                await env.reset(state["sandbox_id"], state["python_state"])

                code = template.instantiate(profile, params=params)
                output = await env.python(
                    code=code,
                    sandbox_id=state["sandbox_id"],
                    python_state=state["python_state"],
                )

                if "✓ Submitted:" not in output:
                    params_str = json.dumps(params) if params else "default"
                    failed.append(f"{template.name} ({params_str})")

        assert not failed, f"Templates failed to produce submission: {failed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
