# Debug Log: Procedural Question Ground Truth Bug

**Date:** 2026-01-30
**Status:** Investigation in progress
**Issue:** Procedural questions showing `'group': '?'` or empty ground truth instead of actual computed values

---

## Symptoms

1. **Pass rate test shows 0%** because ground truth doesn't match computed answers
2. **Ground truth values are malformed:**
   - Expected: `{'group': 'PP_333_20g/L', 'count': 680}` (actual computed answer)
   - Got: `{'group': '?', 'count': 2795}` or `{}` (from template, not execution)

3. **Old questions have proper ground truth:**
   - Stored in `_ground_truth` field with actual values
   - Example: `"target_column": "Cholesterol", "test_statistic": 7.97`

---

## Investigation Findings

### 1. Program Generator Flow

**File:** `src/datagen/synthetic/programs/program_generator.py`

**Execution phase (lines 129-164):**
```python
answer = submission.get("__csv_agent_answer__")
if answer is None:
    continue

executed.append({
    "program": spec,
    "answer": answer,  # <-- This should be the real answer
    ...
})
```

**Record creation (lines 257, 274-276):**
```python
answer = program["answer"]  # <-- Gets answer from executed

record = {
    "ground_truth": answer,
    "ground_truth_hash": hash_artifact(answer),
}
```

**Finding:** The code looks correct - it should store the actual answer from execution.

### 2. Cascading Template Analysis

**File:** `src/datagen/synthetic/programs/semantic_long_chains.py`

**Template chain:**
```python
ops = [
    OpInstance("select_numeric_cols", {}),
    OpInstance("bind_numeric_col", {"selected_col": col_a}),
    OpInstance("filter_by_threshold", {"selected_col": col_a, "threshold": threshold_a}),
    OpInstance("bind_numeric_col", {"selected_col": col_b}),
    OpInstance("filter_by_threshold", {"selected_col": col_b, "threshold": threshold_b}),
    OpInstance("select_categorical_cols", {}),
    OpInstance("bind_binary_cat_col", {"cat_col": cat_col}),
    OpInstance("groupby_count", {}),        # <-- Creates group_counts
    OpInstance("argmax_group_count", {}),   # <-- Uses group_counts, calls submit()
]
```

**Operator verification:**
- `groupby_count`: Creates `group_counts` variable, outputs "Dict"
- `argmax_group_count`: Consumes `group_counts`, calls `submit({"group": ..., "count": ...})`

**Finding:** Template operators are correctly defined and should produce proper output.

### 3. Operator Type Signatures

**File:** `src/datagen/synthetic/programs/operators.py`

**groupby_count:**
- Inputs: `["Table", "CatCol"]`
- Outputs: `["Dict"]`
- Produces: `["group_counts"]`

**argmax_group_count:**
- Inputs: `["Dict"]`
- Outputs: `["Dict"]`
- Consumes: `["group_counts"]`
- Emits answer: `True`

**Finding:** Type signatures look correct for chaining.

---

## Hypotheses

### Hypothesis 1: Execution is failing silently
- The cascading chain might be failing during execution
- No submission is produced, so question is skipped
- But we see questions being generated...

### Hypothesis 2: Answer format mismatch
- The answer is being submitted but in wrong format
- Parser can't extract it properly
- But we see `{'group': '?', ...}` which suggests template default

### Hypothesis 3: Template instantiation bug
- The template is not being properly instantiated with actual column values
- Placeholders like `?` remain in the output
- This matches the symptom exactly

### Hypothesis 4: Ground truth is from template, not execution
- The template defines `output_schema: '{"group": "<category>", "count": 0}'`
- This might be getting used as ground truth instead of actual execution result
- **Most likely hypothesis**

---

## Next Steps to Investigate

1. **Check actual execution output:**
   ```python
   # Add debug logging in program_generator.py around line 143-151
   print(f"Submission: {submission}")
   print(f"Answer: {answer}")
   ```

2. **Verify template execution:**
   - Run a single cascading template manually
   - Check if it produces correct output
   - Compare to stored ground truth

3. **Check filter_programs:**
   - The `filter_programs()` function might be modifying answers
   - Line 168 in program_generator.py

4. **Compare with working templates:**
   - Old questions (like ANOVA) have proper ground truth
   - What's different about their generation path?

---

## Related Files

- `src/datagen/synthetic/programs/program_generator.py` - Main generation logic
- `src/datagen/synthetic/programs/semantic_long_chains.py` - Template definitions
- `src/datagen/synthetic/programs/operators.py` - Operator implementations
- `src/datagen/synthetic/programs/compiler.py` - Code compilation

---

## Test Command

```bash
# Run single question debug
uv run python -c "
import asyncio
from src.datagen.synthetic.programs.program_generator import run_pipeline
import tempfile

async def test():
    with tempfile.TemporaryDirectory() as tmpdir:
        questions = await run_pipeline(
            csv_path='data/csv/data.csv',
            max_programs=1,
            max_verbalize=1,
            output_dir=tmpdir,
        )
        if questions:
            print('Ground truth:', questions[0].get('ground_truth'))
            print('Program name:', questions[0].get('program_name'))

asyncio.run(test())
"
```

---

## Notes

- Old template-based questions (anova_discovered_groups) have proper `_ground_truth` field
- New procedural questions have empty or malformed ground truth
- The issue seems specific to cascading filter templates
- Simple 3-step aggregations (mean) might work correctly - need to verify
