"""Smoke tests for compositional programs (Option B, grammar search)."""

import pytest
import pytest_asyncio

from src.datagen.synthetic.profiler import DataProfiler
from src.datagen.synthetic.programs.compiler import compile_program
from src.datagen.synthetic.programs.sampler import sample_programs
from src.datagen.synthetic.programs.operators import get_operator
from src.envs.csv_env import LocalCSVAnalysisEnv
from pathlib import Path

DATA_CSV = Path("data/csv/data.csv")


@pytest_asyncio.fixture(scope="module")
async def sandbox():
    env = LocalCSVAnalysisEnv(csv_path=str(DATA_CSV))
    state = await env.setup_state({})
    yield env, state
    await env.destroy_sandbox(state["sandbox_id"])


@pytest.fixture
def profile():
    profiler = DataProfiler()
    return profiler.analyze(str(DATA_CSV))


@pytest.fixture
def programs(profile):
    return sample_programs(profile)


class TestProgramCompilation:
    def test_at_least_two_programs_compile(self, profile, programs):
        compiled = 0
        for spec in programs:
            try:
                compile_program(spec, profile)
                compiled += 1
            except Exception:
                pass
        assert compiled >= 2, f"Only {compiled} programs compiled; need >= 2"

    def test_all_programs_compile(self, profile, programs):
        for spec in programs:
            code = compile_program(spec, profile)
            assert code

    @pytest.mark.asyncio
    async def test_at_least_one_program_compiles(self, profile, programs, sandbox):
        env, state = sandbox
        executed = 0
        for spec in programs:
            try:
                code = compile_program(spec, profile)
                output = await env.python(
                    code=code,
                    sandbox_id=state["sandbox_id"],
                    python_state=state["python_state"],
                )
                if "✓ Submitted:" in output:
                    executed += 1
                    break
            except Exception:
                pass
        assert executed >= 1, f"Only {executed} programs executed; need >= 1"

    def test_program_count_floor(self, profile, programs):
        # We are not enforcing program count targets yet.
        assert len(programs) >= 1, "Expected at least one program"

    def test_no_arbitrary_selection(self, profile, programs):
        binary_cats = [
            c
            for c, info in profile.get("columns", {}).items()
            if info.get("type") == "categorical" and info.get("unique_count", 0) == 2
        ]
        if len(binary_cats) >= 2:
            cat_cols_in_progs = set()
            for prog in programs:
                for op in prog.ops:
                    if op.params.get("cat_col"):
                        cat_cols_in_progs.add(op.params["cat_col"])
            assert len(cat_cols_in_progs) >= 2, (
                "Should enumerate all binary cat cols, not pick first"
            )

    def test_system2_coverage(self, profile, programs):
        binary_cats = [
            c
            for c, info in profile.get("columns", {}).items()
            if info.get("type") == "categorical" and info.get("unique_count", 0) == 2
        ]
        if not binary_cats:
            pytest.skip("No binary categorical columns available")

        def _has_decision(op_name: str) -> bool:
            op = get_operator(op_name)
            return op is not None and "decision" in op.attributes

        has_decision = any(
            any(_has_decision(op.op_name) for op in prog.ops) for prog in programs
        )
        assert has_decision, "At least one program should have decision operator"
