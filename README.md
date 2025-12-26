# csv-agent

Agent for CSV dataset exploration and training data generation.

## Setup

```bash
uv sync
```

## Configuration

All settings are in `config.yaml`:
- `csv` - Path to dataset
- `teacher_model` / `question_gen_model` - Model identifiers
- `max_turns` - Max conversation turns
- `n_consistency` - Number of verification traces

## Pipeline

### Stage 1: Generate Questions

Explore a dataset and generate questions:

```bash
uv run python -m src.datagen.question_gen
```

Output: `data/questions/<dataset>/questions.json`

---

### Stage 2: Generate Training Data

Validate questions via teacher triangulation:

```bash
uv run python -m src.datagen.episode_gen
```

Output: `data/episodes/episodes.jsonl`

---

### Stage 3: Upload to HuggingFace

Upload verified episodes for training:

```bash
# Login first (one-time)
huggingface-cli login

# Upload splits
uv run python scripts/upload_hf.py --repo your-username/csv-agent-episodes --splits data/episodes/splits

# Private dataset
uv run python scripts/upload_hf.py --repo your-username/csv-agent-episodes --private
```

Then load for training:
```python
from datasets import load_dataset

ds = load_dataset("your-username/csv-agent-episodes")
ds = ds.map(lambda x: {"messages": x["conversation_for_sft"]["messages"]})
```

---

### Adding Datasets from Kaggle

To expand your training data with Kaggle datasets:

1. Install the optional kaggle dependency:
   ```bash
   uv sync --extra kaggle
   ```

2. Set up Kaggle API credentials (see `kaggle/README.md`)

3. Download datasets:
   ```bash
   # Download popular tabular datasets
   uv run python kaggle/download_datasets.py --limit 10

   # Or from a curated list
   uv run python kaggle/download_datasets.py --from-list kaggle/curated_datasets.json
   ```

4. Use downloaded datasets:
   ```bash
   uv run python -m src.datagen.question_gen --csv data/kaggle/uciml_iris.csv
   ```

---

## Tests

```bash
uv run pytest tests/ -v
```

