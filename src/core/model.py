"""
LLM wrappers for API-based inference.

Usage:
    # OpenRouter (set OPENROUTER_API_KEY env var)
    from src.core.model import APILLM
    llm = APILLM(model="anthropic/claude-sonnet-4")
    
    # Local server (vLLM, Ollama)
    llm = APILLM(base_url="http://localhost:8000/v1", model="Qwen/Qwen3-32B", api_key="none")
"""

import asyncio
import os
import httpx


class APILLM:
    """
    LLM client for OpenAI-compatible APIs (OpenRouter, vLLM, Ollama, etc.).

    IMPORTANT: model parameter has no default. It MUST be provided explicitly
    from config.yaml. Do not add a default model here.
    """

    def __init__(
        self,
        model: str,  # No default! Must come from config.yaml
        *,  # Force remaining args to be keyword-only
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,  # Falls back to OPENROUTER_API_KEY env var
        timeout: float = 120.0,
        sampling_args: dict | None = None,
    ):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=timeout)
        self.sampling_args = sampling_args or {}
    
    async def __call__(
        self,
        prompt: str | list[dict],
    ) -> str:
        # Accept either a string or a list of messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        # Retry logic for API errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": self.sampling_args.get("max_tokens", 8192),
                        "temperature": self.sampling_args.get("temperature", 0.7),
                        "top_p": self.sampling_args.get("top_p", 1.0),
                        "n": self.sampling_args.get("n", 1),
                        "stream": self.sampling_args.get("stream", False),
                        "logprobs": self.sampling_args.get("logprobs", None),
                        "echo": self.sampling_args.get("echo", False),
                        "stop": self.sampling_args.get("stop", None),
                        "presence_penalty": self.sampling_args.get("presence_penalty", 0.0),
                        "frequency_penalty": self.sampling_args.get("frequency_penalty", 0.0),
                    },
                )
                response.raise_for_status()
                json_response = response.json()

                # Handle error responses
                if "error" in json_response:
                    error_code = json_response["error"].get("code")
                    if error_code == 500 and attempt < max_retries - 1:
                        # Retry on 500 errors
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                    raise ValueError(f"API error: {json_response['error']}")

                if "choices" not in json_response:
                    raise ValueError(f"Unexpected API response format. Got: {json_response}")

                return json_response["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < max_retries - 1:
                    # Retry on 5xx errors
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise

        raise RuntimeError(f"Failed after {max_retries} attempts")
    
    async def aclose(self):
        """Async cleanup method."""
        if hasattr(self, "client"):
            await self.client.aclose()
