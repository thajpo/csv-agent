# Training Data Preparation

This module prepares episode data for API fine-tuning.

## Usage

### Basic Usage

Convert episodes to OpenAI format:
```bash
uv run python -m src.training.prepare_finetune_data \
  --input episodes/train.jsonl \
  --provider openai \
  --output training_data/train_openai.jsonl
```

Convert episodes to Anthropic format:
```bash
uv run python -m src.training.prepare_finetune_data \
  --input episodes/train.jsonl \
  --provider anthropic \
  --output training_data/train_anthropic.jsonl
```

### Options

- `--input` (required): Path to input episodes JSONL file
- `--provider`: API provider format (`openai` or `anthropic`, default: `openai`)
- `--output`: Output path (default: `training_data/train_{provider}.jsonl`)
- `--include-unverified`: Include unverified episodes (default: verified only)

## Output Formats

### OpenAI Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a data analysis assistant."},
    {"role": "user", "content": "What is the average value in column A?\nHint: Use df['A'].mean()"},
    {"role": "assistant", "content": "result = df['A'].mean()\nsubmit(result)"}
  ]
}
```

### Anthropic Format

```json
{
  "system": "You are a data analysis assistant.",
  "messages": [
    {"role": "user", "content": "What is the average value in column A?\nHint: Use df['A'].mean()"},
    {"role": "assistant", "content": "result = df['A'].mean()\nsubmit(result)"}
  ]
}
```

## Data Flow

1. **Input**: Episodes JSONL file (from `src.datagen.episode_gen`)
2. **Extract**: `conversation_for_sft` field from each verified episode
3. **Format**: Convert to provider-specific format (OpenAI or Anthropic)
4. **Output**: Training-ready JSONL file for API upload

## Implementation Details

- **Default behavior**: Only includes verified episodes (`verified=True`)
- **Warning handling**: Skips episodes with missing/malformed conversation data
- **System prompts**:
  - OpenAI: Included as first message with `role: "system"`
  - Anthropic: Separated into `system` field
- **Message validation**: Ensures all required fields are present

## Example Workflow

```bash
# 1. Generate episodes
uv run python -m src.datagen.episode_gen

# 2. Split into train/val/test (if using split_episodes.py)
# uv run python -m src.training.split_episodes --input episodes/episodes.jsonl

# 3. Prepare for OpenAI fine-tuning
uv run python -m src.training.prepare_finetune_data \
  --input episodes/train.jsonl \
  --provider openai

# 4. Upload to OpenAI
# openai api fine_tuning.jobs.create \
#   -t training_data/train_openai.jsonl \
#   -m gpt-4o-mini
```

## Testing

Test with mock data:
```bash
uv run python -m src.training.prepare_finetune_data \
  --input test_fixtures/mock_episodes.jsonl \
  --provider openai \
  --output training_data/test_openai.jsonl
```
