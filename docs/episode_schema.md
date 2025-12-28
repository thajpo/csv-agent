# Episode JSONL Schema

> **Purpose**: Canonical schema for training data episodes.  
> **Location**: `data/episodes/episodes.jsonl` (one JSON object per line)

---

## Top-Level Episode

```json
{
    "episode_id": "uuid-string",
    "timestamp": "2025-12-21T16:00:00.000000",
    "csv_source": "path/to/data.csv",
    
    "question": { /* QADict */ },
    "gold_trace": { /* TraceDict - teacher WITH hint */ },
    "consistency_traces": [ /* list[TraceDict] - teacher WITHOUT hint */ ],
    
    "verified": true,
    "triangulation": { /* TriangulationMetadataDict */ },
    "timing": { /* TimingMetadataDict */ }
}
```

---

## QADict (Question + Answer)

Contains the question and ground truth for evaluation.

```json
{
    "id": "16-char-hex",
    "question_text": "What is the mean age of patients with heart disease?",
    "hint": "Filter by target==1, then compute mean of age column",
    "difficulty": "EASY | MEDIUM | HARD | VERY_HARD",
    "n_steps": 3,
    
    "template_name": "filter_aggregate",
    "template_params": {"column": "age", "filter_col": "target"},
    "output_type": "float",
    "output_schema": null,
    
    "ground_truth": 54.3,
    "ground_truth_hash": "a1b2c3d4e5f67890"
}
```

**Note**: `ground_truth` is included for synthetic (template-based) questions. For LLM-generated questions, this may be null and correctness is determined by triangulation.

---

## TraceDict

A complete execution trace = sequence of turns + final outcome.

```json
{
    "turns": [ /* list[TurnDict] */ ],
    "final_answer": 54.3,
    "final_answer_hash": "a1b2c3d4e5f67890",
    "success": true
}
```

### TurnDict

Single turn = model reasoning + code + execution result.

```json
{
    "turn_index": 0,
    "reasoning": "I need to filter the dataframe for patients with heart disease...",
    "code": "df_filtered = df[df['target'] == 1]\nmean_age = df_filtered['age'].mean()\nsubmit(mean_age)",
    "execution": { /* ExecutionResultDict */ }
}
```

### ExecutionResultDict

Result of executing one code cell.

```json
{
    "success": true,
    "stdout": "54.3\n✓ Submitted: {\"__csv_agent_answer__\": 54.3, \"hooks\": [...]}",
    "stderr": "",
    "hooks": [ /* list[HookDict] */ ],
    "submitted_answer": 54.3
}
```

### HookDict

Intermediate checkpoint for PRM/RL training.

**Value storage policy:**
- Scalars (int, float, str, bool, None): Stored in full
- DataFrame/Series: Bounded summary (shape, dtypes, head rows, numeric stats)
- Other complex types (dict, list): Stored if < 100KB, else type+size metadata
- `value_hash` always computed on full normalized value for verification

**DataFrame summary example:**
```json
{
    "variable_name": "df_filtered",
    "code_line": "df_filtered = df[df['target'] == 1]",
    "value": {
        "type": "DataFrame",
        "shape": [303, 14],
        "columns": ["age", "sex", "cp", "trestbps", "..."],
        "dtypes": {"age": "int64", "sex": "int64", "...": "..."},
        "head": [[63, 1, 3, 145], [37, 1, 2, 130], [41, 0, 1, 130]],
        "numeric_summary": {
            "age": {"mean": 54.3, "min": 29, "max": 77}
        }
    },
    "value_hash": "b2c3d4e5f6789012",
    "depends_on": [],
    "description": "Filtered dataframe for heart disease patients"
}
```

---

## Metadata Dicts

### TriangulationMetadataDict

```json
{
    "n_consistency_runs": 3,
    "n_consistency_succeeded": 3,
    "majority_answer_hash": "a1b2c3d4e5f67890",
    "majority_count": 3,
    "gold_matches_majority": true
}
```

### TimingMetadataDict

```json
{
    "gold_elapsed": 12.5,
    "consistency_elapsed": [8.2, 9.1, 7.8],
    "total_elapsed": 37.6,
    "avg_elapsed": 9.4
}
```

---

## Verified Episode Guarantees

An episode with `verified: true` means:

1. Gold trace's `final_answer` matches majority of consistency traces (within float tolerance)
2. Teacher successfully called `submit()` in gold trace
3. At least one consistency trace also submitted a matching answer

---

## Training Format Derivation

Training formats are derived at training time from the structured data, not pre-baked.

### Standard SFT

Derive from `gold_trace.turns`:
```python
messages = [{"role": "system", "content": system_prompt}]
messages.append({"role": "user", "content": question_text + hint})
for turn in gold_trace["turns"]:
    messages.append({"role": "assistant", "content": f"{turn['reasoning']}\n```python\n{turn['code']}\n```"})
    messages.append({"role": "user", "content": f"[stdout]:\n{turn['execution']['stdout']}"})
```

### Interleaved SFT (State Prediction)

Train model to predict interpreter state between code execution:
```python
for turn in gold_trace["turns"]:
    # Model writes code
    messages.append({"role": "assistant", "content": f"```python\n{turn['code']}\n```"})
    # Model predicts what hooks will be captured
    for hook in turn["execution"]["hooks"]:
        messages.append({"role": "user", "content": "Predict the value of " + hook["variable_name"]})
        messages.append({"role": "assistant", "content": str(hook["value"])})
    # Reveal actual execution result
    messages.append({"role": "user", "content": f"[actual stdout]:\n{turn['execution']['stdout']}"})
```

### PRM (Process Reward Model)

Score each intermediate step:
```python
samples = []
for turn in gold_trace["turns"]:
    for hook in turn["execution"]["hooks"]:
        samples.append({
            "prefix": "...",  # conversation up to this point
            "step": hook["code_line"],
            "value": hook["value"],
            "label": 1.0 if verified else 0.0
        })
```

---

## Migration from Old Schema

The old schema had:
- `teacher_gold_trace` → now `gold_trace`
- `conversation_for_sft` → removed (derive at training time)
- `rl_verification_data` → removed (use `gold_trace.final_answer_hash`)

Use `src.training.prepare_finetune_data` to convert episodes to training formats.
