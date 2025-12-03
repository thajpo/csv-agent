import pytest
from src.llm import LLM, has_gpu


@pytest.mark.skipif(not has_gpu(), reason="No CUDA/ROCm GPU available")
def test_llm_local():
    """Test LLM on available GPU (CUDA or ROCm)."""
    llm = LLM()  # Auto-detects device
    result = llm("Say 'test' and nothing else.")
    assert isinstance(result, str)
    assert len(result) > 0