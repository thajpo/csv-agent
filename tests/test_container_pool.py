"""
Test the multi-tenant container pool.

Tests:
- Pool creation and startup
- Slot acquisition and release
- Code execution on slots
- Parallel execution
- Security restrictions in pool workers
"""

import asyncio
import pytest
import pytest_asyncio
from src.envs.container_pool import ContainerPool, Slot


@pytest_asyncio.fixture
async def pool():
    """Create a small container pool for testing."""
    p = ContainerPool(
        csv_path="data/csv/data.csv",
        n_containers=1,
        workers_per_container=3,
    )
    await p.start()
    yield p
    await p.stop()


class TestContainerPool:
    """Test container pool functionality."""

    @pytest.mark.asyncio
    async def test_pool_starts(self, pool):
        """Pool should start and report correct stats."""
        stats = pool.get_stats()
        assert stats["containers"] == 1
        assert stats["workers_per_container"] == 3
        assert stats["total_slots"] == 3
        assert stats["available"] == 3
        assert stats["in_use"] == 0

    @pytest.mark.asyncio
    async def test_acquire_slots(self, pool):
        """Should be able to acquire slots."""
        slots = await pool.acquire_slots(2)
        assert len(slots) == 2
        assert all(isinstance(s, Slot) for s in slots)

        stats = pool.get_stats()
        assert stats["in_use"] == 2
        assert stats["available"] == 1

    @pytest.mark.asyncio
    async def test_release_slots(self, pool):
        """Should be able to release slots."""
        slots = await pool.acquire_slots(2)
        await pool.release_slots(slots)

        stats = pool.get_stats()
        assert stats["in_use"] == 0
        assert stats["available"] == 3

    @pytest.mark.asyncio
    async def test_run_code(self, pool):
        """Should execute code on a slot."""
        slots = await pool.acquire_slots(1)
        slot = slots[0]

        result = await pool.run_code(slot, "df.shape")
        assert "(" in result  # Tuple output

        await pool.release_slots(slots)

    @pytest.mark.asyncio
    async def test_parallel_execution(self, pool):
        """Should execute code in parallel on multiple slots."""
        slots = await pool.acquire_slots(3)

        # Run different code on each slot in parallel
        results = await asyncio.gather(*[
            pool.run_code(slot, f"df.shape[{i}]")
            for i, slot in enumerate(slots)
        ])

        # Should get results from all slots
        assert len(results) == 3
        assert all(r for r in results)  # All should have output

        await pool.release_slots(slots)

    @pytest.mark.asyncio
    async def test_slot_isolation(self, pool):
        """Variables in one slot should not affect another."""
        slots = await pool.acquire_slots(2)

        # Set variable in slot 0
        await pool.run_code(slots[0], "my_var = 42")
        result0 = await pool.run_code(slots[0], "my_var")
        assert "42" in result0

        # Should not exist in slot 1
        result1 = await pool.run_code(slots[1], "my_var")
        assert "NameError" in result1 or "not defined" in result1.lower()

        await pool.release_slots(slots)

    @pytest.mark.asyncio
    async def test_reset_slot(self, pool):
        """Reset should clear slot namespace."""
        slots = await pool.acquire_slots(1)
        slot = slots[0]

        # Set a variable
        await pool.run_code(slot, "test_var = 123")
        result1 = await pool.run_code(slot, "test_var")
        assert "123" in result1

        # Reset
        await pool.reset_slot(slot)

        # Variable should be gone
        result2 = await pool.run_code(slot, "test_var")
        assert "NameError" in result2 or "not defined" in result2.lower()

        await pool.release_slots(slots)

    @pytest.mark.asyncio
    async def test_security_blocks_os(self, pool):
        """Security: os module should be blocked."""
        slots = await pool.acquire_slots(1)
        slot = slots[0]

        result = await pool.run_code(slot, "import os")
        assert "not allowed" in result.lower() or "ImportError" in result

        await pool.release_slots(slots)

    @pytest.mark.asyncio
    async def test_security_blocks_open(self, pool):
        """Security: open() should not be available."""
        slots = await pool.acquire_slots(1)
        slot = slots[0]

        result = await pool.run_code(slot, "open('/etc/passwd')")
        assert "NameError" in result or "not defined" in result.lower()

        await pool.release_slots(slots)

    @pytest.mark.asyncio
    async def test_submit_works(self, pool):
        """submit() function should work in pool workers."""
        slots = await pool.acquire_slots(1)
        slot = slots[0]

        result = await pool.run_code(slot, "submit(df.shape[0])")
        assert "Submitted" in result

        await pool.release_slots(slots)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
