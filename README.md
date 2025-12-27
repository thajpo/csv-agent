# csv-agent

Synthetic training data generation pipeline for CSV analysis agents. Uses teacher triangulation to create verified question-answer pairs with execution traces.

## Setup

```bash
uv sync
```

## Configuration

Settings are in `src/core/config.py` (Pydantic models). Key fields:

| Setting | Default | Description |
|---------|---------|-------------|
| `teacher_model` | `openai/gpt-oss-120b` | Model for episode generation |
| `question_gen_model` | `openai/gpt-oss-120b` | Model for question generation |
| `max_turns` | `10` | Max conversation turns per episode |
| `n_consistency` | `7` | Number of consistency traces for triangulation |
| `n_question_slots` | `4` | Parallel questions per container (tune for speed) |
| `float_tolerance` | `0.1` | Tolerance for float comparison in answer matching |

## Pipeline

### Quick Start (Recommended)

Run the full pipeline with a single command:

```bash
uv run python -m src.datagen.run_all                       # Full pipeline (both)
uv run python -m src.datagen.run_all --synth               # Synthetic only
uv run python -m src.datagen.run_all --llm                 # LLM only
uv run python -m src.datagen.run_all --triangulate         # Episodes only (existing questions)
uv run python -m src.datagen.run_all --synth --triangulate # Synthetic episodes only
uv run python -m src.datagen.run_all --test                # Quick e2e test (~30 seconds)
```

---

### Manual Pipeline

Two paths for question generation:

| Path | Speed | Determinism | Use Case |
|------|-------|-------------|----------|
| **Synthetic** | Fast | Deterministic | Scale, reproducibility |
| **LLM** | Slow | Non-deterministic | Exploration, diversity |

#### Stage 1: Generate Questions

**Synthetic (recommended for scale):**
```bash
uv run python -m src.datagen.synthetic.generator
```

**LLM-based exploration:**
```bash
uv run python -m src.datagen.question_gen
```

#### Stage 2: Generate Training Episodes

**For synthetic questions:**
```bash
uv run python -m src.datagen.synthetic_episodes \
    --questions-dir data/questions_synthetic \
    --output data/episodes/episodes_synthetic.jsonl
```

**For LLM questions (with triangulation):**
```bash
uv run python -m src.datagen.episode_gen \
    --questions-dir data/questions_llm \
    --output data/episodes/episodes_llm.jsonl \
    --skip-difficulty-filter
```

**Options:**
```bash
--skip-difficulty-filter  # Use all questions (recommended for LLM)
--max-questions N         # Limit questions per dataset (for testing)
--n-consistency N         # Override consistency traces (default: 7)
--difficulties HARD VERY_HARD  # Filter by difficulty
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

## Debugging & Inspection

**Inspect outputs:**
```bash
uv run python -m src.utils.inspect questions               # Preview questions
uv run python -m src.utils.inspect questions --show-hint   # With hints
uv run python -m src.utils.inspect episodes --verified     # Show verified episodes
uv run python -m src.utils.inspect trace abc123            # Deep-dive single episode
```

**Validate single question:**
```bash
uv run python -m src.datagen.validate_question \
    --csv data/csv/data.csv \
    --questions-file data/questions_synthetic/dataset/questions.json \
    --index 0 \
    --show-code
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

