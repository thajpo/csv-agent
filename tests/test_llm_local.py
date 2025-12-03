import pytest
import torch
from src.llm import LLM


def has_gpu():
    """Check if CUDA or ROCm is available."""
    has_cuda = torch.cuda.is_available()
    has_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    return has_cuda or has_rocm


@pytest.mark.skipif(not has_gpu(), reason="No CUDA/ROCm GPU available")
def test_llm_local():
    """Test LLM on available GPU (CUDA or ROCm)."""
    llm = LLM()  # Auto-detects device
    result = llm("Say 'test' and nothing else.")
    assert isinstance(result, str)
    assert len(result) > 0