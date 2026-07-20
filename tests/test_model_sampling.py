"""Tests for OpenAI-compatible sampling request construction."""

import json

import httpx
import pytest

from src.core.model import APILLM


@pytest.mark.asyncio
async def test_api_llm_passes_optional_seed_to_provider() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "response"}}]},
        )

    llm = APILLM(
        model="test-model",
        api_key="test-key",
        sampling_args={"temperature": 0.7, "seed": 42},
    )
    await llm.client.aclose()
    llm.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await llm("hello") == "response"
    finally:
        await llm.aclose()

    assert captured["seed"] == 42


@pytest.mark.asyncio
async def test_api_llm_treats_null_content_as_an_empty_model_turn() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": None}}]},
        )

    llm = APILLM(
        model="test-model",
        api_key="test-key",
        sampling_args={"temperature": 0.7},
    )
    await llm.client.aclose()
    llm.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await llm("hello") == ""
    finally:
        await llm.aclose()
