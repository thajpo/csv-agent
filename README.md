# csv-agent

Agent for CSV dataset exploration and training data generation.

## Setup

```bash
uv sync
```

## Pipeline

### Stage 1: Generate Questions

Explore a dataset and generate questions with hints:

```bash
uv run python scripts/generate_questions.py --csv csv/data.csv
```

Options:
- `--model` - Model to use (default: `openai/gpt-4o`)
- `--max-turns` - Max exploration turns (default: 20)
- `--output-dir` - Output directory (default: `outputs`)

Output: `outputs/<dataset>/questions.json`

---

### Stage 2: Generate Training Data

Validate questions via teacher triangulation and save verified traces:

```bash
uv run python scripts/generate_training_data.py \
    --csv csv/data.csv \
    --questions outputs/data/questions.json
```

Options:
- `--model` - Model to use (default: `openai/gpt-4o`)
- `--n-consistency` - Number of verification traces (default: 3)
- `--max-turns` - Max turns per trace (default: 10)

Output:
- `outputs/<dataset>/traces.json` - All traces
- `outputs/<dataset>/training_data.json` - Verified traces only

---

### Stage 3: Train (future)

Student RL training using verified traces. Not yet implemented.

## Tests

```bash
uv run pytest tests/ -v
```
