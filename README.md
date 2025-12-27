# csv-agent

Synthetic training data generation pipeline for CSV analysis agents. Uses teacher triangulation to create verified question-answer pairs with execution traces.

## Setup

```bash
uv sync
```

## Configuration

Settings are in `src/core/config.py` (Pydantic models). Key fields:
- `teacher_model` / `question_gen_model` - Model identifiers
- `max_turns` - Max conversation turns per episode
- `n_consistency` - Number of verification traces for triangulation
- `float_tolerance` - Tolerance for float comparison in answer matching

## Pipeline

Two paths for question generation:

| Path | Speed | Determinism | Use Case |
|------|-------|-------------|----------|
| **Synthetic** | Fast | Deterministic | Scale, reproducibility |
| **LLM** | Slow | Non-deterministic | Exploration, diversity |

---

### Stage 1: Generate Questions

**Synthetic (recommended for scale):**
```bash
uv run python -m src.datagen.synthetic.generator
uv run python -m src.datagen.synthetic.generator --csv path/to/data.csv
```

**LLM-based exploration:**
```bash
uv run python -m src.datagen.question_gen
```

Output: `data/questions_synthetic/<dataset>/questions.json`

---

### Stage 2: Generate Training Episodes

**Synthetic questions** (single teacher trace, validates against ground truth):
```bash
uv run python -m src.datagen.synthetic_episodes \
    --questions-dir data/questions_synthetic \
    --output data/episodes/synthetic.jsonl

# Parallel mode (process multiple datasets concurrently)
uv run python -m src.datagen.synthetic_episodes \
    --questions-dir data/questions_synthetic \
    --output data/episodes/synthetic.jsonl \
    --parallel --n-workers 4
```

**LLM questions** (full triangulation: gold + N consistency traces):
```bash
uv run python -m src.datagen.episode_gen
uv run python -m src.datagen.episode_gen --parallel
```

Output: `data/episodes/*.jsonl`

---

### Stage 3: Upload to HuggingFace

```bash
huggingface-cli login  # one-time

uv run python scripts/upload_hf.py --repo your-username/csv-agent-episodes
uv run python scripts/upload_hf.py --repo your-username/csv-agent-episodes --private
```

---

## Smoke Test

Validate the pipeline end-to-end:

```bash
uv run python scripts/smoke_test.py                    # Full pipeline
uv run python scripts/smoke_test.py --stage questions  # Question gen only
uv run python scripts/smoke_test.py --n-questions 3    # More questions per CSV
```

---

## Adding Datasets from Kaggle

```bash
uv sync --extra kaggle

# Download datasets
uv run python scripts/kaggle/download_datasets.py --limit 10

# Use downloaded datasets
uv run python -m src.datagen.synthetic.generator --csv data/kaggle/*/data.csv
```

---

## Tests

```bash
uv run pytest tests/ -v
```

