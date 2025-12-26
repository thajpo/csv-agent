# Training Data Preparation

Convert episodes to various training formats.

## Formats

### Standard SFT (`sft-standard`)
Classic conversation format for supervised fine-tuning.
```
[system] You are a data analysis assistant...
[user] What is the mean age? Hint: Use df['age'].mean()
[assistant] I'll compute the mean age...\n```python\ndf['age'].mean()```
[user] [stdout]: 54.3
```

### Interleaved SFT (`sft-interleaved`)
Train model to predict interpreter state between code execution (Lucas Beyer style).
```
[system] ...After writing code, predict what values will be computed...
[user] What is the mean age?
[assistant] ```python\nmean_age = df['age'].mean()```
[user] Predict the intermediate values:
[assistant] mean_age = 54.3
[user] [actual output]: 54.3
```

### PRM (`prm`)
Process Reward Model samples - one sample per hook/step with correctness label.
```json
{
  "prefix": "[conversation up to this point]",
  "step_type": "hook",
  "code_line": "mean_age = df['age'].mean()",
  "value": 54.3,
  "label": 1.0
}
```

## Usage

```bash
# Standard SFT
uv run python -m src.training.prepare_finetune_data \
  --input data/episodes/episodes.jsonl \
  --format sft-standard \
  --output data/training/train_sft.jsonl

# Interleaved SFT (state prediction)
uv run python -m src.training.prepare_finetune_data \
  --input data/episodes/episodes.jsonl \
  --format sft-interleaved \
  --output data/training/train_interleaved.jsonl

# PRM samples
uv run python -m src.training.prepare_finetune_data \
  --input data/episodes/episodes.jsonl \
  --format prm \
  --output data/training/train_prm.jsonl
```

## Options

- `--input` (required): Path to episodes JSONL
- `--format`: `sft-standard`, `sft-interleaved`, or `prm` (default: `sft-standard`)
- `--output`: Output path (default: auto-generated from input + format)
- `--include-unverified`: Include unverified episodes (default: verified only)

## Python API

```python
from src.training.prepare_finetune_data import (
    load_episodes,
    to_sft_standard,
    to_sft_interleaved,
    to_prm_samples,
)

episodes = load_episodes("data/episodes/episodes.jsonl")

for ep in episodes:
    # Standard SFT
    sft = to_sft_standard(ep)
    # {"messages": [...]}
    
    # Interleaved SFT
    interleaved = to_sft_interleaved(ep)
    # {"messages": [...]}  (with state prediction turns)
    
    # PRM samples
    prm_samples = to_prm_samples(ep)
    # [{"prefix": ..., "step_type": ..., "label": ...}, ...]
```

## Data Flow

```
episodes.jsonl (structured turns)
        │
        ├──► sft-standard    → Fine-tune for code generation
        ├──► sft-interleaved → Fine-tune for code + state prediction  
        └──► prm             → Train process reward model
```
