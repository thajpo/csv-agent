"""
LLM wrappers for local inference and API-based inference.

Usage:
    # Local model (loads into GPU memory)
    from src.llm import LLM
    llm = LLM()
    response = llm("What is 2+2?")
    
    # OpenRouter (set OPENROUTER_API_KEY env var)
    from src.llm import APILLM
    llm = APILLM()  # defaults to qwen/qwen3-32b
    llm = APILLM(model="anthropic/claude-sonnet-4")
    
    # Local server (vLLM, Ollama)
    llm = APILLM(base_url="http://localhost:8000/v1", model="Qwen/Qwen3-32B", api_key="none")
"""

import os
import httpx

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class APILLM:
    """LLM client for OpenAI-compatible APIs (OpenRouter, vLLM, Ollama, etc.)."""
    
    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "x-ai/grok-4.1-fast:free",
        api_key: str | None = None,  # Falls back to OPENROUTER_API_KEY env var
        timeout: float = 120.0,
    ):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(timeout=timeout)
    
    def __call__(
        self,
        prompt: str | list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        # Accept either a string or a list of messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def __del__(self):
        if hasattr(self, "client"):
            self.client.close()


class LLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",  # 3B not released yet, 4B is closest
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA/ROCm device available")
        
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
