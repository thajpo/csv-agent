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


def has_gpu() -> bool:
    """Check if CUDA or ROCm GPU is available."""
    has_cuda = torch.cuda.is_available()
    has_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    return has_cuda or has_rocm


class APILLM:
    """LLM client for OpenAI-compatible APIs (OpenRouter, vLLM, Ollama, etc.)."""
    
    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        # model: str = "deepseek/deepseek-v3.2-exp",  # or try "google/gemini-2.0-flash-exp:free"
        model: str = "meta-llama/llama-3.2-3b-instruct:free",
        api_key: str | None = None,  # Falls back to OPENROUTER_API_KEY env var
        timeout: float = 120.0,
        sampling_args: dict = {},
    ):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(timeout=timeout)
        self.sampling_args = sampling_args
    
    def __call__(
        self,
        prompt: str | list[dict],
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
        return response.json()["choices"][0]["message"]["content"]
    
    def __del__(self):
        if hasattr(self, "client"):
            self.client.close()


class LLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",  # 3B not released yet, 4B is closest
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ):
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
