"""
Test the multi-tenant container.

Tests:
- Container creation and startup
- Code execution on workers
- Parallel execution
- Worker isolation and reset
- Security restrictions
"""

import asyncio
import pytest
import pytest_asyncio
from src.envs.container_pool import MultiTenantContainer, Slot, WorkerAdapter


@pytest_asyncio.fixture(scope="module")
async def container():
    """Create a multi-tenant container for testing (module-scoped for speed)."""
    c = MultiTenantContainer(
        csv_path="data/csv/data.csv",
        n_workers=3,
    )
    await c.start()
    yield c
    await c.stop()


class TestMultiTenantContainer:
    """Test multi-tenant container functionality."""

    @pytest.mark.asyncio
    async def test_container_starts(self, container):
        """Container should start and report correct stats."""
        stats = container.get_stats()
        assert stats["n_workers"] == 3
        assert stats["started"] is True
        assert container.container_id is not None

    @pytest.mark.asyncio
    async def test_run_on_worker(self, container):
        """Should execute code on a worker."""
        result = await container.run_on_worker(0, "df.shape")
        assert "(" in result  # Tuple output

    @pytest.mark.asyncio
    async def test_parallel_execution(self, container):
        """Should execute code in parallel on multiple workers."""
        # Run different code on each worker in parallel
        results = await asyncio.gather(
            *[container.run_on_worker(i, f"df.shape[{i % 2}]") for i in range(3)]
        )

        # Should get results from all workers
        assert len(results) == 3
        assert all(r for r in results)  # All should have output

    @pytest.mark.asyncio
    async def test_worker_isolation(self, container):
        """Variables in one worker should not affect another."""
        # Set variable in worker 0
        await container.run_on_worker(0, "my_var = 42")
        result0 = await container.run_on_worker(0, "my_var")
        assert "42" in result0

        # Should not exist in worker 1
        result1 = await container.run_on_worker(1, "my_var")
        assert "NameError" in result1 or "not defined" in result1.lower()

    @pytest.mark.asyncio
    async def test_reset_worker(self, container):
        """Reset should clear worker namespace."""
        # Set a variable
        await container.run_on_worker(0, "test_var = 123")
        result1 = await container.run_on_worker(0, "test_var")
        assert "123" in result1

        # Reset
        await container.reset_worker(0)

        # Variable should be gone
        result2 = await container.run_on_worker(0, "test_var")
        assert "NameError" in result2 or "not defined" in result2.lower()

    @pytest.mark.asyncio
    async def test_reset_all_workers(self, container):
        """Reset all should clear all worker namespaces."""
        # Set variables in all workers
        for i in range(3):
            await container.run_on_worker(i, f"worker_var_{i} = {i}")

        # Reset all
        await container.reset_all_workers()

        # All variables should be gone
        for i in range(3):
            result = await container.run_on_worker(i, f"worker_var_{i}")
            assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_security_blocks_os(self, container):
        """Security: os module should be blocked."""
        result = await container.run_on_worker(0, "import os")
        assert "not allowed" in result.lower() or "ImportError" in result

    @pytest.mark.asyncio
    async def test_security_blocks_open(self, container):
        """Security: open() should not be available."""
        result = await container.run_on_worker(0, "open('/etc/passwd')")
        assert "NameError" in result or "not defined" in result.lower()

    @pytest.mark.asyncio
    async def test_submit_works(self, container):
        """submit() function should work in workers."""
        result = await container.run_on_worker(0, "submit(df.shape[0])")
        assert "Submitted" in result

    @pytest.mark.asyncio
    async def test_hook_works(self, container):
        """hook() function should work in workers."""
        result = await container.run_on_worker(
            0, "hook(df.shape[0], 'df.shape[0]', name='row_count')"
        )
        assert "Hook" in result


class TestWorkerAdapter:
    """Test WorkerAdapter interface compatibility."""

    @pytest.mark.asyncio
    async def test_adapter_python_method(self, container):
        """WorkerAdapter.python() should work like LocalCSVAnalysisEnv."""
        adapter = WorkerAdapter(container, worker_id=0)
        state = WorkerAdapter.create_state(0)

        result = await adapter.python(
            "df.shape",
            sandbox_id=state["sandbox_id"],
            python_state=state["python_state"],
        )
        assert "(" in result  # Tuple output

    @pytest.mark.asyncio
    async def test_adapter_state_tracking(self, container):
        """WorkerAdapter should track execution count."""
        adapter = WorkerAdapter(container, worker_id=0)
        state = WorkerAdapter.create_state(0)

        assert state["python_state"]["execution_count"] == 0

        await adapter.python("1+1", python_state=state["python_state"])
        assert state["python_state"]["execution_count"] == 1

        await adapter.python("2+2", python_state=state["python_state"])
        assert state["python_state"]["execution_count"] == 2

    @pytest.mark.asyncio
    async def test_adapter_reset_state(self, container):
        """WorkerAdapter.reset_state() should reset worker namespace."""
        adapter = WorkerAdapter(container, worker_id=1)
        state = WorkerAdapter.create_state(1)

        # Set a variable
        await adapter.python("adapter_var = 999", python_state=state["python_state"])
        result1 = await adapter.python(
            "adapter_var", python_state=state["python_state"]
        )
        assert "999" in result1

        # Reset
        await adapter.reset_state(state)

        # Variable should be gone
        result2 = await adapter.python(
            "adapter_var", python_state=state["python_state"]
        )
        assert "NameError" in result2 or "not defined" in result2.lower()

    @pytest.mark.asyncio
    async def test_adapter_destroy_is_noop(self, container):
        """WorkerAdapter.destroy_sandbox() should be a no-op."""
        adapter = WorkerAdapter(container, worker_id=0)

        # Should not raise or affect container
        await adapter.destroy_sandbox("worker-0")

        # Container should still work
        result = await container.run_on_worker(0, "1+1")
        assert "2" in result

    @pytest.mark.asyncio
    async def test_adapter_as_container_pool(self, container):
        """WorkerAdapters should work as container_pool for triangulation."""
        # Create adapters like we would for triangulation
        container_pool = [
            (WorkerAdapter(container, i), WorkerAdapter.create_state(i))
            for i in range(container.n_workers)
        ]

        assert len(container_pool) == 3

        # Each adapter should execute independently
        results = await asyncio.gather(
            *[
                adapter.python(f"'worker_{i}'", python_state=state["python_state"])
                for i, (adapter, state) in enumerate(container_pool)
            ]
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"worker_{i}" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
