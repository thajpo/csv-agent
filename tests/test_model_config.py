import pytest

from src.core.config import Config, ModelConfig
from src.core.model import APILLM
from src.core.prompts import format_difficulty_distribution


def test_generation_and_teacher_model_defaults():
    config = Config(csv_sources=[])

    assert config.question_gen_model == "qwen/qwen3.7-max"
    assert config.teacher_model == "openai/gpt-5.5"
    assert config.question_difficulty_distribution == {
        "EASY": 0.25,
        "MEDIUM": 0.25,
        "HARD": 0.25,
        "VERY_HARD": 0.25,
    }
    assert config.teacher_sampling_args_dict()["max_tokens"] == 16000
    assert config.teacher_sampling_args_dict()["reasoning"] == {
        "effort": "xhigh",
        "exclude": True,
    }


def test_model_config_preserves_reasoning_sampling_arg():
    model = ModelConfig(
        model_name="openai/gpt-5.5",
        temperature=0.7,
        max_tokens=6000,
        reasoning={"effort": "xhigh", "exclude": True},
    )

    assert model.sampling_args_dict()["reasoning"] == {
        "effort": "xhigh",
        "exclude": True,
    }


@pytest.mark.asyncio
async def test_api_llm_forwards_reasoning_payload():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeClient:
        def __init__(self):
            self.payload = None

        async def post(self, _url, *, headers, json):
            self.payload = json
            return FakeResponse()

    fake_client = FakeClient()
    llm = APILLM(
        model="openai/gpt-5.5",
        api_key="test-key",
        sampling_args={
            "temperature": 0.7,
            "max_tokens": 6000,
            "reasoning": {"effort": "xhigh", "exclude": True},
        },
    )
    llm.client = fake_client

    assert await llm("hello") == "ok"
    assert fake_client.payload["model"] == "openai/gpt-5.5"
    assert fake_client.payload["reasoning"] == {
        "effort": "xhigh",
        "exclude": True,
    }


@pytest.mark.asyncio
async def test_api_llm_raises_helpful_error_for_missing_content():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "reasoning": "hidden"},
                    }
                ]
            }

    class FakeClient:
        async def post(self, _url, *, headers, json):
            return FakeResponse()

    llm = APILLM(
        model="openai/gpt-5.5",
        api_key="test-key",
        sampling_args={
            "temperature": 0.7,
            "max_tokens": 100,
            "reasoning": {"effort": "xhigh", "exclude": True},
        },
    )
    llm.client = FakeClient()

    with pytest.raises(ValueError, match="increase max_tokens"):
        await llm("hello")


def test_even_difficulty_distribution_prompt_counts():
    rendered = format_difficulty_distribution(
        40, {"EASY": 0.25, "MEDIUM": 0.25, "HARD": 0.25, "VERY_HARD": 0.25}
    )

    assert "- 10 EASY" in rendered
    assert "- 10 MEDIUM" in rendered
    assert "- 10 HARD" in rendered
    assert "- 10 VERY_HARD" in rendered
