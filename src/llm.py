"""
Qwen3-3B wrapper for fast local inference on ROCm.

Usage:
    from src.llm import LLM
    
    llm = LLM()
    response = llm("What is 2+2?")
"""

import os

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",  # was PYTORCH_HIP_ALLOC_CONF (deprecated)
    "garbage_collection_threshold:0.8,max_split_size_mb:128",
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
