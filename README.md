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
csvagent run --all
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

Next: csvagent generate episodes --llm-gen
```

### Generate Questions

Two paths:

| Path | Speed | Determinism | Use Case |
|------|-------|-------------|----------|
| **Synthetic** | Fast | Deterministic | Scale, reproducibility |
| **LLM** | Slow | Non-deterministic | Exploration, diversity |

```bash
csvagent generate questions --template     # Template-based
csvagent generate questions --procedural   # Program/procedural
csvagent generate questions --llm-gen      # LLM exploration
csvagent generate questions --all --dry-run
```

### Generate Episodes

Episodes are verified question-answer traces for training.

```bash
# Preview first (always safe)
csvagent generate episodes --template --dry-run
csvagent generate episodes --procedural --dry-run
csvagent generate episodes --llm-gen --dry-run

# Generate (appends by default - won't overwrite existing)
csvagent generate episodes --template
csvagent generate episodes --procedural
csvagent generate episodes --llm-gen

# Start fresh (explicit overwrite)
csvagent generate episodes --template --fresh
```

**Safe defaults:**
- Pre-flight summary shows progress before running
- Append mode by default (skips already-processed questions)
- Use `--fresh` to explicitly overwrite existing data

### Full Pipeline

```bash
csvagent run --all          # Full pipeline (questions + episodes)
csvagent run --template     # Template only
csvagent run --procedural   # Procedural only
csvagent run --llm-gen      # LLM only
csvagent run --test         # Quick e2e test (~30 seconds)
```

---

## Inspection & Debugging

```bash
# Coverage report
csvagent stats
csvagent stats --gaps       # Show missing data

# Inspect outputs
csvagent inspect questions --source template  # Preview template questions
csvagent inspect questions --source all --show-hint        # With hints
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

## Documentation

| Document | Purpose |
|----------|---------|
| [current.md](current.md) | Active planning and spec funnel (`Institutional Knowledge`, `Beliefs`, `Brainstormed`, `Specd`) |
| [AGENTS.md](AGENTS.md) | Repo collaboration and execution guardrails |

**Key insight:** Episodes capture raw structured data (traces, hooks, corrections). Training formats (SFT, PRM, DPO) are derived at training time, not pre-baked. This means new training methods can reuse existing episodes without regeneration.

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

## Caching & Incremental Generation

The pipeline uses a manifest (`data/datagen_manifest.jsonl`) to track processed questions. This enables:

- **Skip redundant work** - Already-processed questions are skipped automatically
- **Resume interrupted runs** - Just re-run the command, it picks up where it left off
- **Template change detection** - When template code changes, only affected questions re-run

```bash
# View manifest summary
csvagent manifest

# Force re-run of failed questions
uv run python -m src.datagen.validate_synthetic --questions-dir data/questions_synthetic --output data/episodes/episodes_synthetic.jsonl --retry-failed

# To fully reset cache, delete the manifest file
rm data/datagen_manifest.jsonl
```

The manifest tracks fingerprints based on:
- **Synthetic**: template code + params + dataset content hash
- **LLM**: normalized question text + dataset content hash

Changing template code automatically invalidates cached results for that template.

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
