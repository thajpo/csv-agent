# Local SFT Training Plan

**Status**: Deferred - implement when ready to train.

## Goal
Build local SFT training capability into the project:
1. Generate episodes using OpenRouter (already works)
2. Train small model (3-7B) on rented GPU (RunPod/etc)
3. All training code lives in the repo - no external training APIs

---

## Phase 1: Episode Generation (READY)

OpenRouter is already configured. Just set API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
uv run python -m src.datagen.episode_gen
```

Model is configured in `src/core/config.py` (see `teacher_model` and `question_gen_model`)

---

## Phase 2: SFT Training Script

### New file: `src/training/train_sft.py`

```python
"""
Local SFT training script using HuggingFace TRL.

Usage:
    # On GPU machine (RunPod, etc):
    pip install trl peft transformers accelerate bitsandbytes
    python -m src.training.train_sft \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data training_data/train_openai.jsonl \
        --output ./checkpoints
"""

import argparse
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

def format_chat(example):
    """Convert messages to chat format string."""
    messages = example["messages"]
    # Format for model's chat template
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="training_data/train_openai.jsonl")
    parser.add_argument("--output", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # Load and format dataset
    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

    # Training config
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        max_seq_length=4096,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
```

### Dependencies to add
```bash
uv add --group training trl peft transformers accelerate bitsandbytes
```

---

## Phase 3: Cloud GPU Usage (RunPod)

**Workflow:**

1. Rent GPU on RunPod (~$0.50/hr for A10G 24GB)
2. SSH into instance
3. Clone repo + install deps
4. Run training script
5. Download checkpoints

**RunPod setup script:**
```bash
# On RunPod instance:
git clone https://github.com/youruser/csv-agent
cd csv-agent
pip install uv
uv sync --group training

# Run training
python -m src.training.train_sft \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data training_data/train_openai.jsonl \
    --output ./checkpoints \
    --epochs 3
```

---

## Phase 4: Evaluation

Use existing evaluation harness:
```bash
uv run python -m scripts.evaluate_model \
    --model ./checkpoints \
    --episodes test_fixtures/mock_episodes.jsonl
```

---

## Implementation Status

Training script implemented in standalone `training/` package (see `training/train_sft.py`).

---

## Recommended Models

| Model | Size | VRAM (4-bit) | Notes |
|-------|------|--------------|-------|
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ~6GB | Fast iteration |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~10GB | Better quality |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~6GB | Solid baseline |

A10G (24GB) handles 7B with room to spare.

---

## Cost Estimate

- **RunPod A10G**: ~$0.50/hr
- **Training 3B model**: ~1-2 hours for small dataset
- **Total**: ~$1-2 per training run
