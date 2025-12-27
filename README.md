# csv-agent

Synthetic training data generation pipeline for CSV analysis agents. Uses teacher triangulation to create verified question-answer pairs with execution traces.

## Setup

```bash
uv sync
```

## CLI

All commands go through `csvagent`:

```bash
csvagent                    # Interactive menu
csvagent status             # Data inventory
csvagent progress           # Detailed progress with time estimates
csvagent generate ...       # Generate questions or episodes
csvagent run ...            # Full pipeline
csvagent inspect ...        # Inspect data
csvagent validate ...       # Debug single question
csvagent stats              # Coverage report
```

---

## Quick Start

```bash
# 1. Check what data you have
csvagent status

# 2. Quick end-to-end test (~30 seconds)
csvagent run --test

# 3. Generate everything
csvagent run --both
```

---

## Pipeline

### Status & Progress

```bash
csvagent status      # Quick data inventory
csvagent progress    # Detailed progress with time estimates
```

Example output:
```
csv-agent Data Generation Pipeline
  Datasets     77 available
  Questions    synthetic 1,399 (63 datasets) | llm 1,504 (75 datasets)
  Episodes     synthetic 11/11 verified (100%) | llm 0/0 verified (0%)

Next: csvagent generate episodes --llm
```

### Generate Questions

Two paths:

| Path | Speed | Determinism | Use Case |
|------|-------|-------------|----------|
| **Synthetic** | Fast | Deterministic | Scale, reproducibility |
| **LLM** | Slow | Non-deterministic | Exploration, diversity |

```bash
csvagent generate questions --synth     # Template-based (fast)
csvagent generate questions --llm       # LLM exploration (slow)
csvagent generate questions --synth --dry-run  # Preview only
```

### Generate Episodes

Episodes are verified question-answer traces for training.

```bash
# Preview first (always safe)
csvagent generate episodes --synth --dry-run
csvagent generate episodes --llm --dry-run

# Generate (appends by default - won't overwrite existing)
csvagent generate episodes --synth
csvagent generate episodes --llm

# Start fresh (explicit overwrite)
csvagent generate episodes --synth --fresh
```

**Safe defaults:**
- Pre-flight summary shows progress before running
- Append mode by default (skips already-processed questions)
- Use `--fresh` to explicitly overwrite existing data

### Full Pipeline

```bash
csvagent run --both         # Full pipeline (questions + episodes)
csvagent run --synth        # Synthetic only
csvagent run --llm          # LLM only
csvagent run --triangulate  # Episodes only (skip question gen)
csvagent run --test         # Quick e2e test (~30 seconds)
```

---

## Inspection & Debugging

```bash
# Coverage report
csvagent stats
csvagent stats --gaps       # Show missing data

# Inspect outputs
csvagent inspect questions                    # Preview questions
csvagent inspect questions --show-hint        # With hints
csvagent inspect episodes --verified          # Show verified episodes
csvagent inspect trace abc123                 # Deep-dive single episode

# Debug single question
csvagent validate \
    --csv data/csv/data.csv \
    --questions-file data/questions_synthetic/dataset/questions.json \
    --index 0 \
    --show-code
```

---

## Configuration

Settings are in `src/core/config.py` (Pydantic models). Key fields:

| Setting | Default | Description |
|---------|---------|-------------|
| `teacher_model` | `openai/gpt-oss-120b` | Model for episode generation |
| `question_gen_model` | `openai/gpt-oss-120b` | Model for question generation |
| `max_turns` | `10` | Max conversation turns per episode |
| `n_consistency` | `7` | Number of consistency traces for triangulation |
| `n_question_slots` | `4` | Parallel questions per container |
| `float_tolerance` | `0.1` | Tolerance for float comparison |
| `dynamic_triangulation` | `true` | Adjust consistency by difficulty |
| `triangulation_by_difficulty` | `{EASY: 2, MEDIUM: 2, HARD: 4, VERY_HARD: 6}` | Per-difficulty consistency |

---

## Adding Datasets from Kaggle

```bash
uv sync --extra kaggle

# Download datasets
uv run python scripts/kaggle/download_datasets.py --limit 10
```

---

## Upload to HuggingFace

```bash
huggingface-cli login  # one-time

uv run python scripts/upload_hf.py --repo your-username/csv-agent-episodes
uv run python scripts/upload_hf.py --repo your-username/csv-agent-episodes --private
```

---

## Tests

```bash
uv run pytest tests/ -v
```
