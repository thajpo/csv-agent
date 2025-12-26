import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests that hit real LLM APIs",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


class FakeLLM:
    """
    Fake LLM for testing orchestration logic without API calls.

    Yields canned responses in order. Each response should be a valid
    assistant message with ```python ... ``` code blocks.
    """

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.captured_prompts: list = []

    async def __call__(self, prompt: str | list[dict]) -> str:
        self.captured_prompts.append(prompt)
        if self.call_count >= len(self.responses):
            raise RuntimeError(
                f"FakeLLM exhausted: {self.call_count} calls but only {len(self.responses)} responses"
            )
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

    async def aclose(self):
        pass


@pytest.fixture
def fake_llm_simple():
    """FakeLLM that submits 42 on first turn."""
    return FakeLLM(
        ["I'll analyze the data and submit the answer.\n\n```python\nsubmit(42)\n```"]
    )


@pytest.fixture
def fake_llm_factory():
    """Factory fixture to create FakeLLM with custom responses."""
    return FakeLLM
