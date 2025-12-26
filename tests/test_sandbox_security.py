"""
Test sandbox security restrictions.

Verifies that the worker script properly restricts:
- Dangerous imports (os, subprocess, etc.)
- File operations (open)
- Code execution (exec, eval, compile)
"""

import pytest
import pytest_asyncio
from src.envs.csv_env import LocalCSVAnalysisEnv


@pytest_asyncio.fixture(scope="module")
async def sandbox():
    """Create a sandbox environment for testing (module-scoped for speed)."""
    env = LocalCSVAnalysisEnv(csv_path="data/csv/data.csv")
    state = await env.setup_state(state={})
    sandbox_id = state["sandbox_id"]
    python_state = state["python_state"]
    yield env, sandbox_id, python_state
    await env.destroy_sandbox(sandbox_id)


class TestSandboxSecurity:
    """Test that dangerous operations are blocked."""

    @pytest.mark.asyncio
    async def test_safe_pandas_operations(self, sandbox):
        """Normal pandas operations should work."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "df.shape",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        assert "error" not in result.lower() or "Error" not in result
        assert "(" in result  # Should contain tuple like (100, 5)

    @pytest.mark.asyncio
    async def test_safe_numpy_operations(self, sandbox):
        """Numpy operations should work."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "import numpy as np; np.array([1,2,3]).mean()",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        assert "2.0" in result

    @pytest.mark.asyncio
    async def test_block_os_import(self, sandbox):
        """Import os should be blocked."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "import os",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        assert "not allowed" in result.lower() or "ImportError" in result

    @pytest.mark.asyncio
    async def test_block_subprocess_import(self, sandbox):
        """Import subprocess should be blocked."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "import subprocess",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        assert "not allowed" in result.lower() or "ImportError" in result

    @pytest.mark.asyncio
    async def test_block_open(self, sandbox):
        """open() should not be available."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "open('/etc/passwd', 'r')",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        # Should fail because open is not in builtins
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_block_exec(self, sandbox):
        """exec() should not be available."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "exec('print(1)')",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_block_eval(self, sandbox):
        """eval() should not be available."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "eval('1+1')",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_block_compile(self, sandbox):
        """compile() should not be available."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "compile('1+1', '<string>', 'eval')",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_submit_still_works(self, sandbox):
        """submit() function should still work."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "submit(42)",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        assert "Submitted" in result
        assert "42" in result

    @pytest.mark.asyncio
    async def test_hook_still_works(self, sandbox):
        """hook() function should still work."""
        env, sandbox_id, python_state = sandbox

        result = await env.python(
            "hook(df.shape[0], 'df.shape[0]', name='row_count')",
            sandbox_id=sandbox_id,
            python_state=python_state,
        )
        assert "Hook" in result

    @pytest.mark.asyncio
    async def test_allowed_stdlib_imports(self, sandbox):
        """Safe stdlib modules should be importable."""
        env, sandbox_id, python_state = sandbox

        # Test several allowed modules
        for module in ["math", "json", "re", "datetime", "hashlib", "collections"]:
            result = await env.python(
                f"import {module}; '{module} ok'",
                sandbox_id=sandbox_id,
                python_state=python_state,
            )
            assert f"{module} ok" in result, f"Failed to import {module}: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
