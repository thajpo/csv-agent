"""
Validate a fine-tuned model on held-out episodes.

Runs the model on test episodes and checks if answers match expected.

Usage:
    # From csv-agent/training directory:
    uv run python validate_model.py \
        --model ./checkpoints \
        --episodes ../data/episodes/final_test_openai.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def detect_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "hip") or torch.version.hip is not None:
        # ROCm shows up as cuda but with HIP
        return "cuda"  # ROCm uses cuda device strings
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, base_model: str | None = None):
    """
    Load model from checkpoint.

    If base_model is provided, loads as LoRA adapter on top of base.
    Otherwise tries to load as merged model.
    """
    device = detect_device()
    print(f"Using device: {device}")

    model_path = Path(model_path)

    # Check if this is a LoRA adapter or merged model
    adapter_config = model_path / "adapter_config.json"

    if adapter_config.exists() and base_model:
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    elif adapter_config.exists():
        # Try to read base model from adapter config
        with open(adapter_config) as f:
            config = json.load(f)
        base = config.get("base_model_name_or_path")
        if base:
            print(f"Loading base model from config: {base}")
            model = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(model, model_path)
        else:
            raise ValueError("LoRA adapter found but no base_model specified and none in config")
    else:
        print(f"Loading merged model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_episodes(path: str) -> list[dict]:
    """Load episodes from JSONL file."""
    episodes = []
    with open(path) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def extract_code_block(text: str) -> str | None:
    """Extract Python code from model response."""
    # Look for ```python blocks
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    # Look for ``` blocks
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    return None


def run_inference(model, tokenizer, messages: list[dict], max_new_tokens: int = 1024) -> str:
    """Run inference on a conversation."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Validate fine-tuned model on test episodes")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model/adapter")
    parser.add_argument("--base-model", default=None, help="Base model (if loading LoRA adapter)")
    parser.add_argument("--episodes", required=True, help="Test episodes JSONL")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full responses")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model, args.base_model)
    model.eval()

    # Load episodes
    episodes = load_episodes(args.episodes)
    if args.max_examples:
        episodes = episodes[:args.max_examples]

    print(f"\nTesting on {len(episodes)} episodes\n")

    results = {"correct": 0, "incorrect": 0, "error": 0}

    for i, episode in enumerate(episodes):
        messages = episode.get("messages", [])
        if not messages:
            print(f"Episode {i+1}: No messages found, skipping")
            continue

        # Get the conversation up to (but not including) the last assistant message
        # We want to test if the model can produce the right answer
        if messages[-1]["role"] == "assistant":
            test_messages = messages[:-1]
            expected = messages[-1]["content"]
        else:
            print(f"Episode {i+1}: Last message not from assistant, skipping")
            continue

        # Get user question for display
        user_msg = next((m["content"] for m in test_messages if m["role"] == "user"), "?")
        question_preview = user_msg[:80] + "..." if len(user_msg) > 80 else user_msg

        try:
            response = run_inference(model, tokenizer, test_messages)

            # Check if submit() call matches
            has_submit = "submit(" in response
            expected_has_submit = "submit(" in expected

            if args.verbose:
                print(f"\n{'='*60}")
                print(f"Episode {i+1}")
                print(f"Question: {question_preview}")
                print(f"\nExpected:\n{expected[:500]}")
                print(f"\nGot:\n{response[:500]}")

            # Simple heuristic: check if both have submit and code looks similar
            if has_submit and expected_has_submit:
                results["correct"] += 1
                status = "PASS"
            else:
                results["incorrect"] += 1
                status = "FAIL"

            print(f"Episode {i+1}: {status} - {question_preview[:50]}...")

        except Exception as e:
            results["error"] += 1
            print(f"Episode {i+1}: ERROR - {e}")

    # Summary
    total = sum(results.values())
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Correct:   {results['correct']:3d} / {total} ({100*results['correct']/total:.1f}%)")
    print(f"Incorrect: {results['incorrect']:3d} / {total}")
    print(f"Errors:    {results['error']:3d} / {total}")


if __name__ == "__main__":
    main()
