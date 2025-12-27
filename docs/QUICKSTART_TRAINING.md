# Quick Start: Fine-Tuning Data Preparation

## MVP End-to-End Pipeline

Validate the full pipeline with test fixtures (no API key needed for data prep):

```bash
# 1. Verify SFT data formatting works
uv run python -m src.training.prepare_finetune_data \
  --input data/fixtures/mock_episodes.jsonl \
  --provider openai \
  --output /tmp/test_openai.jsonl

# 2. Validate output format
head -1 /tmp/test_openai.jsonl | python -m json.tool

# 3. Count examples
wc -l /tmp/test_openai.jsonl
```

**With API key**, run the full pipeline:
```bash
# Generate real episodes (requires LLM API)
export OPENAI_API_KEY=sk-...
uv run python -m src.datagen.episode_gen

# Format for fine-tuning
uv run python -m src.training.prepare_finetune_data \
  --input episodes/episodes.jsonl \
  --provider openai

# Evaluate fine-tuned model (use your fine-tuned model ID)
uv run python -m scripts.evaluate_model \
  --model <your-fine-tuned-model> \
  --episodes data/fixtures/mock_episodes.jsonl
```

---

## TL;DR

Convert episode data to API fine-tuning format in one command:

```bash
# OpenAI
uv run python -m src.training.prepare_finetune_data \
  --input episodes/train.jsonl \
  --provider openai

# Anthropic
uv run python -m src.training.prepare_finetune_data \
  --input episodes/train.jsonl \
  --provider anthropic
```

Output: `training_data/train_{provider}.jsonl`

## Format Comparison

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| System prompt location | In messages array | Separate `system` field |
| Message roles | system, user, assistant | user, assistant |
| File structure | `{"messages": [...]}` | `{"system": "...", "messages": [...]}` |

## Common Commands

```bash
# Verified episodes only (default)
uv run python -m src.training.prepare_finetune_data \
  --input episodes/train.jsonl \
  --provider openai

# Include unverified episodes
uv run python -m src.training.prepare_finetune_data \
  --input episodes/train.jsonl \
  --provider openai \
  --include-unverified

# Custom output path
uv run python -m src.training.prepare_finetune_data \
  --input episodes/train.jsonl \
  --provider anthropic \
  --output my_custom_path.jsonl

# View help
uv run python -m src.training.prepare_finetune_data --help
```

## Full Pipeline Example

```bash
# 1. Generate episodes from questions
uv run python -m src.datagen.episode_gen

# 2. (Optional) Split into train/val/test
# uv run python -m src.training.split_episodes

# 3. Prepare training data
uv run python -m src.training.prepare_finetune_data \
  --input episodes/train.jsonl \
  --provider openai

# 4. Upload to provider
# openai api fine_tuning.jobs.create \
#   -t training_data/train_openai.jsonl \
#   -m <base-model-for-finetuning>
```

## Validation

Check output format:
```bash
# View first training example
head -1 training_data/train_openai.jsonl | python -m json.tool

# Count examples
wc -l training_data/train_openai.jsonl
```

## Troubleshooting

**No output generated?**
- Check that input file contains verified episodes
- Use `--include-unverified` to include all episodes

**Format errors?**
- Ensure episodes have `conversation_for_sft` field
- Validate episode structure with `python -m json.tool < episodes/train.jsonl`

**Missing files?**
- Run `uv run python -m src.datagen.episode_gen` first to generate episodes

## Documentation

- Full docs: `src/training/README.md`
- Source code: `src/training/prepare_finetune_data.py`
