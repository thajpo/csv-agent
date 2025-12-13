
import pytest
from src.core.kernel import JupyterKernel

def test_kernel_seeding():
    """Verify that the kernel is seeded deterministically."""
    # Run 1
    k1 = JupyterKernel(csv_path="csv/data.csv")
    try:
        r1 = k1.execute("x = np.random.rand(); print(x)")
        assert r1.success, f"Execution failed: {r1.stderr}"
        print(f"DEBUG: stdout='{r1.stdout}', stderr='{r1.stderr}'")
        val1 = float(r1.stdout.strip())
    finally:
        k1.shutdown()

    # Run 2
    k2 = JupyterKernel(csv_path="csv/data.csv")
    try:
        r2 = k2.execute("x = np.random.rand(); print(x)")
        assert r2.success, f"Execution failed: {r2.stderr}"
        val2 = float(r2.stdout.strip())
    finally:
        k2.shutdown()

    assert val1 == val2, f"Random values should be identical: {val1} != {val2}"
    print(f"✓ Seeding verified: {val1} == {val2}")


def test_state_validation_and_restoration():
    """Verify that we can detect state corruption and restore it."""
    kernel = JupyterKernel(csv_path="csv/data.csv")
    try:
        # 1. Check initial valid state
        assert kernel.validate_state(), "Initial state should be valid"

        # 2. Corrupt state (delete df)
        kernel.execute("del df")
        assert not kernel.validate_state(), "State should be invalid after deleting df"

        # 3. Restore state
        kernel.restore_state()
        assert kernel.validate_state(), "State should be valid after restoration"
        
        # Verify df is back
        r = kernel.execute("df.shape")
        assert r.success, "df should be accessible after restoration"
        
    finally:
        kernel.shutdown()
