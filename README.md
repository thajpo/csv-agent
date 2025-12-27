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

### Stage 2: Generate Training Episodes (Triangulation)

Triangulation verifies questions by running:
1. **Gold trace** - teacher WITH hint
2. **Consistency traces** - teacher WITHOUT hint (N times, in parallel)
3. **Verification** - gold answer must match majority of consistency answers

**For synthetic questions:**
```bash
uv run python -m src.datagen.episode_gen \
    --questions-dir data/questions_synthetic \
    --output data/episodes/episodes_synthetic.jsonl \
    --parallel
```

**For LLM questions:**
```bash
uv run python -m src.datagen.episode_gen \
    --questions-dir data/questions_llm \
    --output data/episodes/episodes_llm.jsonl \
    --parallel \
    --skip-difficulty-filter
```

**Options:**
```bash
--parallel                # Process multiple CSVs concurrently
--skip-difficulty-filter  # Use all questions (recommended for LLM questions)
--max-questions N         # Limit questions per dataset (for testing)
--n-consistency N         # Override consistency traces (default: 7)
--difficulties HARD VERY_HARD  # Filter by difficulty
```

> **Note:** LLM-generated questions often don't match the expected difficulty distribution (30% EASY, 30% MEDIUM, 20% HARD, 20% VERY_HARD). Use `--skip-difficulty-filter` to process all questions. Synthetic questions don't have this issue.

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

