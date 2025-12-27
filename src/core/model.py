"""
LLM wrappers for local inference and API-based inference.

Usage:
    # Local model (loads into GPU memory)
    from src.llm import LLM
    llm = LLM()
    response = llm("What is 2+2?")

    # OpenRouter (set OPENROUTER_API_KEY env var)
    from src.core.config import config
    from src.llm import APILLM
    llm = APILLM(model=config.teacher_model, sampling_args={...})

    # Local server (vLLM, Ollama)
    llm = APILLM(base_url="http://localhost:8000/v1", model="<your-model>", api_key="none", sampling_args={...})
"""

from typing import Any
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import httpx


def has_gpu() -> bool:
    """Check if CUDA or ROCm GPU is available."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        has_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        return has_cuda or has_rocm
    except ImportError:
        return False


class APILLM:
    """
    LLM client for OpenAI-compatible APIs (OpenRouter, vLLM, Ollama, etc.).

    IMPORTANT: model parameter has no default. It MUST be provided explicitly
    from src.core.config. Do not add a default model here.
    """

    def __init__(
        self,
        model: str,  # Must come from src.core.config
        *,  # Force remaining args to be keyword-only
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,  # Falls back to OPENROUTER_API_KEY env var
        timeout: float = 120.0,
        sampling_args: dict,
    ):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=timeout)
        self.sampling_args = sampling_args
    
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
                        # Only include optional params if explicitly set and non-default
                        **({"stop": self.sampling_args["stop"]} if self.sampling_args.get("stop") else {}),
                        **({"presence_penalty": self.sampling_args["presence_penalty"]} if self.sampling_args.get("presence_penalty", 0) != 0 else {}),
                        **({"frequency_penalty": self.sampling_args["frequency_penalty"]} if self.sampling_args.get("frequency_penalty", 0) != 0 else {}),
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
                # Log the error response body for debugging 4xx errors
                if e.response.status_code >= 400 and e.response.status_code < 500:
                    self._log_openrouter_error(e, messages)
                    try:
                        error_body = e.response.json()
                        print(f"⚠️  API Error {e.response.status_code}: {error_body}")
                    except Exception:
                        print(f"⚠️  API Error {e.response.status_code}: {e.response.text[:500]}")
                raise

            except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
                # Retry on network errors (connection issues, timeouts, read errors)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Network error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise

        raise RuntimeError(f"Failed after {max_retries} attempts")

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        """Estimate tokens from message content (4 chars ~ 1 token)."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4

    def _log_openrouter_error(self, error: httpx.HTTPStatusError, messages: list[dict]) -> None:
        """Write a compact error report to logs/openrouter_400.log."""
        try:
            log_path = Path("logs/openrouter_400.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)

            total_chars = sum(len(m.get("content", "")) for m in messages)
            estimated_tokens = self._estimate_tokens(messages)
            max_message_chars = max((len(m.get("content", "")) for m in messages), default=0)
            error_text = ""
            try:
                error_text = error.response.text
            except Exception:
                error_text = "<no response text>"

            error_summary = {
                "status_code": error.response.status_code,
                "url": str(error.request.url),
                "model": self.model,
                "max_tokens": self.sampling_args.get("max_tokens", 8192),
                "temperature": self.sampling_args.get("temperature", 0.7),
                "top_p": self.sampling_args.get("top_p", 1.0),
                "message_count": len(messages),
                "total_chars": total_chars,
                "estimated_tokens": estimated_tokens,
                "max_message_chars": max_message_chars,
            }

            timestamp = datetime.utcnow().isoformat() + "Z"
            with open(log_path, "a") as f:
                f.write(f"[{timestamp}] OpenRouter 4xx error\n")
                f.write(json.dumps(error_summary, indent=2))
                f.write("\n")
                f.write("response_text:\n")
                f.write(error_text[:2000])
                f.write("\n---\n")
        except Exception:
            pass
    
    async def aclose(self):
        """Async cleanup method."""
        if hasattr(self, "client"):
            await self.client.aclose()


class LLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",  # 3B not released yet, 4B is closest
        device: str | None = None,
        torch_dtype: Any = None,
        attn_implementation: str = "sdpa",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        if not has_gpu():
            raise RuntimeError("No CUDA/ROCm device available")
        
        # Auto-detect device if not specified (ROCm also uses "cuda" device name)
        if device is None:
            device = "cuda"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).to(device)
        self.model.eval()
    
    def __call__(
        self,
        prompt: str | list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        # Accept either a string or a list of messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # Disable Qwen3 <think> mode
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        import torch
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            except RuntimeError as err:
                if "hipblas" in str(err).lower():
                    raise RuntimeError(
                        "ROCm HIPBLAS alloc failed during generation. "
                        "Free GPU memory, lower max_new_tokens, or adjust PYTORCH_HIP_ALLOC_CONF."
                    ) from err
                raise
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response


if __name__ == "__main__":
    print("Loading model...")
    llm = LLM()
    print("Model loaded.\n")
    
    response = llm("What is 2+2? Be brief.")
    print(response)
