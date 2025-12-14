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
uv run python scripts/generate_questions.py
```

Override config:
```bash
uv run python scripts/generate_questions.py --csv other.csv --model gpt-4o
```

Output: `outputs/<dataset>/questions.json`

---

### Stage 2: Generate Training Data

Validate questions via teacher triangulation:

```bash
uv run python scripts/generate_training_data.py --questions outputs/data/questions.json
```

Output:
- `outputs/<dataset>/traces.json` - All traces
- `outputs/<dataset>/training_data.json` - Verified traces only

---

### Stage 3: Train (future)

Student RL training. Not yet implemented.

## Tests

```bash
uv run pytest tests/ -v
```
