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

## Prime-RL Adapter

csv-agent does not save a Prime-specific episode format. The canonical
`EpisodeJSONL` records are adapted at training time by the Verifiers environment
exposed as `csv-agent`.

The adapter accepts `csv_root` so absolute `csv_source` paths stored in episodes
can be rebased when training on a remote machine.

```bash
# 1. Split verified episodes
uv run python -m src.training.split_episodes \
  --input data/episodes/template.jsonl \
  --output-dir data/splits/template
```

For remote jobs, upload the split episodes and raw CSV tree to a private
Hugging Face dataset repo:

```bash
uv run python scripts/upload_hf.py \
  --repo-id ThaJpo/csv-agent-template-episodes \
  --include-csvs
```

Prime-RL is intentionally isolated from the main csv-agent environment. It
requires Python 3.12 plus NVIDIA/CUDA Torch/vLLM, so the helper scripts clone it
into `.prime-rl/prime-rl` and expose csv-agent by `PYTHONPATH` when launching.
The standard local RL launcher also expects separate GPU allocations for
inference and training; use at least a 2-GPU NVIDIA box for real RL runs. The
HF-backed config uses Prime-hosted sandboxes for Python execution, so it does
not require Docker on the training pod.

```bash
# 2. On the Prime/NVIDIA box, verify GPU + sandbox support
CSV_AGENT_SANDBOX_BACKEND=prime bash scripts/prime_rl/doctor.sh

# 3. Add secrets on the Prime box
cp configs/prime_rl/secrets.env.example configs/prime_rl/secrets.env
$EDITOR configs/prime_rl/secrets.env

# 4. Create the isolated Prime-RL checkout
bash scripts/prime_rl/bootstrap.sh
# If needed, point at a newer uv binary:
# UV_BIN=/path/to/uv bash scripts/prime_rl/bootstrap.sh
# By default this installs Prime-RL's `flash-attn` extra for NVIDIA training.
# To skip extras for config-only debugging:
# PRIME_RL_UV_EXTRAS= bash scripts/prime_rl/bootstrap.sh

# 5. Validate config resolution without launching training
bash scripts/prime_rl/run.sh configs/prime_rl/csv-agent-hf.toml --dry-run

# 6. Launch a short run
CSV_AGENT_PRIME_RUN_NAME=qwen4b-smoke \
  bash scripts/prime_rl/run.sh configs/prime_rl/csv-agent-hf.toml \
  --max-steps 5 \
  --orchestrator.batch-size 8 \
  --orchestrator.group-size 2 \
  --orchestrator.eval.num-examples 10

# 7. Convert raw rollout JSONL into small repo artifacts
uv run python scripts/prime_rl/plot_run.py \
  --run-dir artifacts/prime_rl/runs/qwen4b-smoke \
  --artifact-dir artifacts/prime_rl/qwen4b-smoke
```

Use [configs/prime_rl/csv-agent-hf.toml](configs/prime_rl/csv-agent-hf.toml) for
remote/NVIDIA training. It loads episodes and CSV files from Hugging Face via
the adapter's `dataset_name` argument. Raw Prime-RL outputs and checkpoints live
under `artifacts/prime_rl/runs/` and are gitignored; reward plots and summary
CSVs under `artifacts/prime_rl/<run-name>/` are intended to be committed and
linked from this README after a run completes.

Latest smoke artifact:
[qwen4b-prime-hf-smoke-20260607-155611](artifacts/prime_rl/qwen4b-prime-hf-smoke-20260607-155611)
completed one Prime-RL step on the Hugging Face dataset with Prime-hosted
Python sandboxes.

Prime-RL currently requires NVIDIA GPUs for actual training. This repo can still
build the environment, split data, and run adapter tests on CPU.

Required keys for the Prime box:

- `HF_TOKEN`: create at [Hugging Face settings/tokens](https://huggingface.co/settings/tokens). Needs read access to `ThaJpo/csv-agent-template-episodes`.
- `WANDB_API_KEY`: create or copy from [Weights & Biases authorizations](https://wandb.ai/authorize). Optional, but recommended.
- Prime auth: run `prime login`, or create/copy an API key from Prime Intellect and set `PRIME_API_KEY`.

Docker note: the local CSV REPL backend uses a CPU-only Docker image, but the
HF-backed Prime-RL config uses `sandbox_backend = "prime"` and does not require
Docker on the training pod.

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
