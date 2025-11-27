# CSV Agent: Synthetic DS Trajectory Dataset

## Overview

Generate trajectories of a teacher LLM solving tabular ML tasks on progressively corrupted data. Use these trajectories to train a smaller model via SFT + GRPO.

**Core loop:**
1. Load Kaggle CSV → teacher solves task on clean data → baseline metric
2. Apply corruption level N → teacher re-solves → log trajectory + metric
3. Repeat until teacher fails or max corruption level reached
4. Package all successful trajectories as training data

---

## 1. Environment onstraints

### Hardware Budget (24GB RAM, 3B student model)
| Constraint | Value | Rationale |
|------------|-------|-----------|
| Max context (student) | 4096 tokens | Safe for 4-bit 3B model |
| Max tokens per episode | ~3000 tokens | Leave room for system prompt |
| Max tool calls per episode | 20 | Bounds trajectory length |
| Max tokens per tool output | 500 | Truncate DataFrames, stdout |

### Dataset Size Limits
| Constraint | Value |
|------------|-------|
| Max rows exposed to agent | 20,000 |
| Max columns | 50 |
| If larger → subsample | Random sample, preserve target distribution |

### Runtime Limits
| Constraint | Value |
|------------|-------|
| Timeout per `run_code` call | 10 seconds |
| Timeout exceeded → | Return error, count as failed step |
| Same error repeated 2x → | End episode, mark as failed |

### Allowed Libraries (whitelist)
```python
ALLOWED_IMPORTS = {
    "pandas", "numpy", "sklearn", "scipy.stats",
    "matplotlib", "seaborn",  # for EDA only
}
```

---

## 2. Tool API (6 functions)

The teacher/student interacts with the Jupyter-like environment through these tools only:

### 2.1 `get_dataset_info`
```python
def get_dataset_info() -> dict:
    """
    Returns schema and summary stats (no raw data).
    
    Returns:
        {
            "n_rows": int,
            "n_cols": int,
            "columns": [
                {"name": str, "dtype": str, "n_missing": int, "n_unique": int, "sample_values": list[3]},
                ...
            ],
            "target_column": str,  # pre-selected
            "task_type": "classification" | "regression",
            "metric": "accuracy" | "f1" | "rmse",
        }
    """
```

### 2.2 `get_data_sample`
```python
def get_data_sample(n_rows: int = 5, columns: list[str] | None = None) -> str:
    """
    Returns a truncated df.head() as markdown table.
    Max 10 rows, max 10 columns per call.
    """
```

### 2.3 `run_code`
```python
def run_code(code: str) -> dict:
    """
    Execute Python code in the stateful Jupyter kernel.
    
    Returns:
        {
            "success": bool,
            "stdout": str,        # truncated to 500 chars
            "stderr": str,        # truncated to 500 chars  
            "error_type": str | None,
            "error_message": str | None,
            "execution_time_ms": int,
        }
    
    Constraints:
        - 10 second timeout
        - Only whitelisted imports allowed
        - Variables persist across calls within episode
    """
```

### 2.4 `get_variable_info`
```python
def get_variable_info(var_name: str) -> dict:
    """
    Inspect a variable in the kernel namespace.
    
    Returns:
        {
            "exists": bool,
            "type": str,
            "shape": tuple | None,  # for arrays/dataframes
            "head": str | None,     # first few rows if dataframe
            "value": str | None,    # repr if small scalar/list
        }
    """
```

### 2.5 `submit_prediction`
```python
def submit_prediction(predictions_var: str) -> dict:
    """
    Submit final predictions for evaluation.
    The variable must be an array-like of predictions on the test set.
    
    Returns:
        {
            "success": bool,
            "metric_name": str,
            "metric_value": float,
            "error": str | None,
        }
    
    Note: This is the ONLY way to get a reward signal.
    Evaluated against ground truth (hidden from agent).
    """
```

### 2.6 `give_up`
```python
def give_up(reason: str) -> dict:
    """
    Agent declares it cannot solve the task.
    Ends episode with reward = 0.
    """
```

---

## 3. Corruption Ladder (v0: 4 levels)

Applied cumulatively. Each level adds to previous.

| Level | Name | Implementation |
|-------|------|----------------|
| 0 | Clean | No corruption (baseline) |
| 1 | Missingness | Inject NaN in 2-3 feature columns, 15-25% rate, MCAR |
| 2 | + Label Noise | Flip 5-10% of labels (classification) or add N(0, 0.1*std) to target (regression) |
| 3 | + Schema Noise | Rename 2-3 columns to generic names ("col_1"), cast 1-2 numeric cols to string |

```python
def apply_corruption(df_clean: pd.DataFrame, level: int, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Returns (df_corrupt, corruption_metadata)
    
    corruption_metadata = {
        "level": int,
        "missingness": {"columns": [...], "rate": float},
        "label_noise": {"rate": float, "type": "flip" | "gaussian"},
        "schema_noise": {"renamed": {old: new, ...}, "cast_to_string": [...]},
    }
    """
```

---

## 4. Task Specification (v0: supervised prediction only)

For v0, keep tasks simple:

```python
@dataclass
class TaskSpec:
    dataset_id: str              # Kaggle dataset identifier
    target_column: str           # Column to predict
    task_type: Literal["classification", "regression"]
    metric: Literal["accuracy", "f1_macro", "rmse", "mae"]
    train_test_split: float      # e.g., 0.8
    random_seed: int
    
    # Natural language prompt (can be LLM-generated)
    problem_statement: str
    # e.g., "Predict whether a customer will churn based on their usage patterns."
```

**Task selection heuristic:**
1. Load Kaggle CSV
2. Identify candidate target columns:
   - Classification: categorical with 2-10 unique values, >100 samples per class
   - Regression: numeric, non-constant, reasonable variance
3. Pick one, generate `TaskSpec`
4. Run teacher on clean data → if metric > threshold, keep this task

---

## 5. Episode Schema

One episode = one (dataset, task, corruption_level) combination.

```python
@dataclass
class Episode:
    # Identifiers
    episode_id: str
    dataset_id: str
    task_spec: TaskSpec
    corruption_level: int
    corruption_metadata: dict
    
    # Trajectory
    messages: list[dict]  # OpenAI-style messages with tool calls
    # Example message:
    # {"role": "assistant", "content": "...", "tool_calls": [{"name": "run_code", "arguments": {...}}]}
    # {"role": "tool", "name": "run_code", "content": "{...}"}
    
    # Outcomes
    n_tool_calls: int
    total_tokens: int           # Approximate, for filtering
    final_metric: float | None  # None if gave up or failed
    baseline_metric: float      # Teacher's score on clean data (level 0)
    reward: float               # final_metric / baseline_metric, or 0 if failed
    
    # Metadata
    success: bool
    failure_reason: str | None  # "timeout", "max_steps", "gave_up", "repeated_error"
    generation_timestamp: str
    teacher_model: str          # e.g., "gpt-4o"
```

---

## 6. Data Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     OUTER LOOP: Per Dataset                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load Kaggle CSV                                              │
│ 2. Subsample if needed (max 20k rows, 50 cols)                  │
│ 3. Select target column → create TaskSpec                       │
│ 4. Generate problem_statement (LLM or template)                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. CLEAN RUN (level=0)                                          │
│    - Reset Jupyter kernel                                       │
│    - Load df_clean into kernel                                  │
│    - Run teacher agent with TaskSpec                            │
│    - If success → baseline_metric = final_metric                │
│    - If fail → discard this dataset                             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. CORRUPTION LOOP: for level in [1, 2, 3]                      │
│    - df_corrupt = apply_corruption(df_clean, level, config)     │
│    - Reset kernel, load df_corrupt                              │
│    - Run teacher agent with same TaskSpec                       │
│    - Compute reward = final_metric / baseline_metric            │
│    - Log Episode                                                │
│    - If reward < threshold (e.g., 0.5) → stop corruption loop   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. POST-PROCESSING                                              │
│    - Filter episodes by total_tokens < 3000                     │
│    - Optionally compress verbose outputs                        │
│    - Save to JSONL                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Error Loop Prevention

Track errors per episode:

```python
class ErrorTracker:
    def __init__(self, max_repeats: int = 2):
        self.error_counts: dict[tuple[str, str], int] = {}  # (error_type, code_hash) -> count
        self.max_repeats = max_repeats
    
    def record_error(self, error_type: str, code: str) -> bool:
        """Returns True if episode should be terminated."""
        key = (error_type, hash(code.strip()))
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        return self.error_counts[key] >= self.max_repeats
```

If `record_error()` returns True → inject a system message: "You have repeated the same error. Episode terminated." → mark episode as failed.

---

## 8. Training Pipeline (Sketch)

### Phase 1: SFT
```python
# Use all successful episodes (reward > 0.7)
# Format as ChatML conversations
# Train with standard SFT (TRL's SFTTrainer)
```

### Phase 2: GRPO
```python
# Reward function:
def compute_reward(episode: Episode) -> float:
    if not episode.success:
        return 0.0
    
    metric_reward = episode.final_metric / episode.baseline_metric
    length_penalty = max(0, 1 - episode.n_tool_calls / 30)  # Prefer shorter
    
    return 0.8 * metric_reward + 0.2 * length_penalty
```

---

## 9. Project Structure

```
csv-agent/
├── PLAN.md                    # This file
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── jupyter_kernel.py  # Stateful code execution
│   │   ├── tools.py           # Tool implementations
│   │   └── constraints.py     # Timeouts, whitelists
│   ├── corruption/
│   │   ├── __init__.py
│   │   └── operators.py       # apply_corruption, etc.
│   ├── data_gen/
│   │   ├── __init__.py
│   │   ├── kaggle_loader.py   # Dataset fetching & prep
│   │   ├── task_generator.py  # TaskSpec creation
│   │   └── teacher_runner.py  # Run teacher through episodes
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft.py
│   │   └── grpo.py
│   └── schemas.py             # Pydantic models (Episode, TaskSpec, etc.)
├── scripts/
│   ├── generate_episodes.py   # Main data generation script
│   └── train.py               # Training script
├── data/
│   ├── raw/                   # Downloaded Kaggle CSVs
│   ├── episodes/              # Generated episodes (JSONL)
│   └── models/                # Trained checkpoints
└── notebooks/
    └── exploration.ipynb      # For debugging
```

---

## 10. Implementation Order

1. **Environment first**: `jupyter_kernel.py`, `tools.py`, `constraints.py`
   - Get a working stateful Python executor with timeouts
   - Implement the 6 tools
   - Test manually

2. **Corruption operators**: `operators.py`
   - Implement the 3 corruption types
   - Test on a sample CSV

3. **Data loading**: `kaggle_loader.py`, `task_generator.py`
   - Fetch datasets via Kaggle API or manual download
   - Target column selection heuristics
   - TaskSpec generation

4. **Teacher runner**: `teacher_runner.py`
   - Wire up teacher LLM (OpenAI API or similar)
   - Implement episode loop
   - Error tracking

5. **Generate first batch**: `generate_episodes.py`
   - Run on 5-10 Kaggle datasets
   - Inspect outputs, iterate on prompts/constraints

6. **Training**: `sft.py`, `grpo.py`
   - Start with pure SFT
   - Add GRPO later

---

## 11. Open Questions (Decisions for You)

| Question | Options | Tradeoff |
|----------|---------|----------|
| Teacher model | GPT-4o / Claude / local 70B | Cost vs. quality of trajectories |
| Kaggle API vs manual | API is cleaner, manual is simpler to start | Time investment |
| Corruption config randomization | Fixed configs vs. random per dataset | Reproducibility vs. diversity |
| Student model | Qwen2.5-3B / Llama-3.2-3B / Phi-3 | Different base capabilities |
| Metric threshold for "success" | 0.6? 0.7? 0.8? | Purity vs. data quantity |

---

## 12. Quick Start (After Implementation)

```bash
# 1. Generate episodes
python scripts/generate_episodes.py \
    --datasets data/raw/*.csv \
    --teacher gpt-4o \
    --output data/episodes/v0.jsonl \
    --max-episodes-per-dataset 12  # 4 levels × 3 seeds

# 2. Train SFT
python scripts/train.py sft \
    --data data/episodes/v0.jsonl \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output data/models/sft-v0

# 3. Train GRPO
python scripts/train.py grpo \
    --data data/episodes/v0.jsonl \
    --model data/models/sft-v0 \
    --output data/models/grpo-v0
```

