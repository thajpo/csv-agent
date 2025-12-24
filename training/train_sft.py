"""
Local SFT training script using HuggingFace TRL.

Usage:
    # From csv-agent/training directory:

    # AMD/ROCm:
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
    uv sync
    uv run python train_sft.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data ../training_data/train_openai.jsonl \
        --output ./checkpoints

    # NVIDIA/CUDA:
    pip install torch
    uv sync --extra cuda
    uv run python train_sft.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data ../training_data/train_openai.jsonl \
        --output ./checkpoints
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def is_rocm() -> bool:
    """Detect if running on AMD ROCm."""
    return torch.version.hip is not None


def get_quantization_config():
    """Get 4-bit quantization config if supported."""
    if is_rocm():
        print("ROCm detected - skipping bitsandbytes (not supported)")
        return None
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except ImportError:
        print("bitsandbytes not available - running without quantization")
        return None


def create_chat_formatter(tokenizer):
    """Create a formatting function with tokenizer in closure."""
    def format_chat(example):
        """Convert messages to chat format string."""
        messages = example["messages"]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    return format_chat


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA on CSV agent episodes")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model to fine-tune")
    parser.add_argument("--data", default="../training_data/train_openai.jsonl", help="Training data JSONL")
    parser.add_argument("--output", default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (default: 2 * lora_r)")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    args = parser.parse_args()

    # Validate input file exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading config
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }

    if not args.no_quantize:
        quant_config = get_quantization_config()
        if quant_config:
            print("Using 4-bit quantization")
            model_kwargs["quantization_config"] = quant_config
        else:
            print("Running without quantization (24GB VRAM should be fine for 7B)")
    else:
        print("Quantization disabled via --no-quantize")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # LoRA configuration
    lora_alpha = args.lora_alpha if args.lora_alpha else args.lora_r * 2
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    print(f"LoRA config: r={args.lora_r}, alpha={lora_alpha}")

    # Load and format dataset
    print(f"Loading dataset: {args.data}")
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"Dataset size: {len(dataset)} examples")

    format_chat = create_chat_formatter(tokenizer)
    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",  # Set to "tensorboard" or "wandb" if needed
    )

    print(f"Training for {args.epochs} epochs, batch_size={args.batch_size}, grad_accum={args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print("Training complete!")


if __name__ == "__main__":
    main()
