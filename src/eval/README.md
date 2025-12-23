# Evaluation Harness

Custom evaluation harness for CSV agent models. Evaluates model performance on test episodes by running inference and comparing answers.

## Architecture

### Components

**`metrics.py`** - Data structures for evaluation results
- `EvalResult`: Single episode evaluation result
- `EvalMetrics`: Aggregate metrics across episodes

**`evaluator.py`** - Main evaluation logic
- `Evaluator.evaluate_episode()`: Evaluate single episode
- `Evaluator.evaluate_batch()`: Parallel batch evaluation
- `Evaluator.compute_metrics()`: Compute aggregate metrics

**`report.py`** - Report generation
- `generate_report()`: Create markdown or JSON reports

### Design Decisions

**Reuse Existing Code**:
- Uses `Environment.rollout()` for execution (same path as training)
- Uses `answers_match()` from `teacher.py` for comparison (float tolerance)
- Ensures consistency between training and evaluation

**Async Throughout**:
- All evaluation is async for parallel processing
- Uses `asyncio.gather()` with semaphore for concurrency control

**Float Tolerance**:
- Default ±0.1 for numeric answers
- Default ±0.002 for p-values
- Handles DataFrames, dicts, lists with element-wise comparison

## Usage

### CLI

```bash
# Evaluate model on test episodes
uv run python -m scripts.evaluate_model \\
  --model openai/gpt-4o-mini \\
  --episodes episodes/test.jsonl \\
  --output eval_results/report.md

# With custom CSV path and JSON output
uv run python -m scripts.evaluate_model \\
  --model openai/gpt-4o-mini \\
  --episodes test_fixtures/mock_episodes.jsonl \\
  --csv mock/data.csv \\
  --format json \\
  --output eval_results/report.json

# Adjust concurrency and tolerance
uv run python -m scripts.evaluate_model \\
  --model openai/gpt-4o-mini \\
  --episodes episodes/test.jsonl \\
  --concurrency 10 \\
  --float-tol 0.05
```

### Python API

```python
import asyncio
from src.eval.evaluator import Evaluator
from src.eval.report import generate_report

# Create evaluator
evaluator = Evaluator(
    model="openai/gpt-4o-mini",
    csv_path="data.csv",  # Optional override
    max_turns=10,
    sampling_args={"temperature": 0.7, "max_tokens": 6000},
    float_tol=0.1,
)

# Load episodes
episodes = evaluator.load_episodes("episodes/test.jsonl")

# Evaluate
results = await evaluator.evaluate_batch(episodes, concurrency=5)

# Compute metrics
metrics = evaluator.compute_metrics(results)

# Generate report
generate_report(
    metrics=metrics,
    results=results,
    output_path="report.md",
    format="markdown",
    model="openai/gpt-4o-mini",
    episodes_path="episodes/test.jsonl",
)
```

## Metrics

### Overall Metrics

- **Accuracy**: Percentage of episodes with correct final answer
- **Execution Success Rate**: Percentage of episodes that submitted an answer
- **Average Turns**: Mean number of conversation turns
- **Average Time**: Mean elapsed time per episode

### Breakdown Metrics

- **Accuracy by Difficulty**: Accuracy split by EASY/MEDIUM/HARD/VERY_HARD
- **Episodes by Difficulty**: Count of episodes per difficulty level
- **Correct by Difficulty**: Count of correct answers per difficulty level

## Report Formats

### Markdown

Human-readable report with:
- Overall metrics summary
- Accuracy breakdown by difficulty
- Per-episode results table
- Failed episodes section (if any)
- Incorrect episodes section (if any)

### JSON

Machine-readable report with:
- Metadata (timestamp, model, episodes path)
- Full metrics object
- Complete results array with all fields

## Evaluation Protocol

For each episode:

1. **Load Episode**: Parse from JSONL file
2. **Extract Question**: Get question_text from episode
3. **Run Inference**: Use `Environment.rollout()` in student mode (no hint)
4. **Compare Answer**: Use `answers_match()` with float tolerance
5. **Record Result**: Create `EvalResult` with correctness and metadata

## Testing

Test with mock episodes:

```bash
uv run python -m scripts.evaluate_model \\
  --model openai/gpt-4o-mini \\
  --episodes test_fixtures/mock_episodes.jsonl \\
  --csv mock/data.csv \\
  --output test_report.md
```

The `test_fixtures/mock_episodes.jsonl` contains 4 episodes with varied difficulty:
- 3 verified episodes (should be correct if model works)
- 1 unverified episode (gold doesn't match majority)

## Future Enhancements

Potential improvements (not implemented in MVP):

- **Hook-level verification**: Check intermediate hooks, not just final answer
- **Confidence intervals**: Bootstrap confidence intervals for metrics
- **Stratified sampling**: Ensure balanced difficulty distribution
- **Error analysis**: Automatic categorization of failure modes
- **Comparative reports**: Compare multiple models side-by-side
