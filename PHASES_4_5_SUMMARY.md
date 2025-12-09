# Phases 4-5 Implementation Summary

## Phase 4: Teacher Triangulation Protocol

### Overview
Implements self-consistency verification to filter out bad questions and ambiguous hints.

### Implementation: `src/teacher.py`

#### Key Functions

**`execute_teacher_trace()`**
- Runs a single teacher execution (with or without hint)
- Creates fresh Jupyter kernel for isolated execution
- Extracts code cells, artifacts, and final answer
- Returns `TeacherTrace` object

**`triangulate_teacher()`**
- Runs 1 gold trace (with hint) + N consistency traces (without hint)
- Uses majority voting on final answer hashes
- Returns `(gold_trace, consistency_traces, verified)` tuple

**`batch_triangulate()`**
- Processes multiple questions in batch
- Returns list of triangulation results for each question

### Verification Logic

```
1. Gold Trace (with hint):
   Question: "What is mean TL for control?"
   Hint: "Filter to control, then calculate mean"
   → Executes with hint, gets answer: 1.234
   → Hash: "a1b2c3d4"

2. Consistency Traces (without hint):
   Trace 1: Answer = 1.234, Hash = "a1b2c3d4" ✓
   Trace 2: Answer = 1.234, Hash = "a1b2c3d4" ✓
   Trace 3: Answer = 2.567, Hash = "x9y8z7w6" ✗

3. Majority Voting:
   Hash "a1b2c3d4": 2 votes (majority)
   Hash "x9y8z7w6": 1 vote

4. Verification:
   Gold hash matches majority? YES → VERIFIED ✓
```

### Why This Matters

**Filters out:**
- Misleading hints
- Ambiguous questions
- Questions the dataset doesn't support
- Questions with multiple valid interpretations

**Example of filtered question:**
```
Question: "Which treatment has better growth?"
Hint: "Compare EL_500 and PP_333"

Gold (with hint): "EL_500" (hash: abc123)
Consistency 1 (no hint): "PP_333" (hash: def456)
Consistency 2 (no hint): "control" (hash: ghi789)
Consistency 3 (no hint): "PP_333" (hash: def456)

Majority: "PP_333" (hash: def456)
Verified: NO ✗ - Hint was misleading!
```

---

## Phase 5: Episode Generation Pipeline

### Overview
Orchestration script that loads questions, runs triangulation, and saves episodes.

### Implementation: `scripts/generate_episodes.py`

#### Features

1. **Load questions** from `questions.json`
2. **Run batch triangulation** on all questions
3. **Create Episode objects** with full trace data
4. **Save verified episodes** as JSON files

#### Usage

```bash
# Generate episodes from questions
python -m scripts.generate_episodes \
  --csv data.csv \
  --questions questions.json \
  --output episodes/ \
  --n-consistency 3 \
  --verified-only

# Arguments:
#   --csv: Path to CSV file
#   --questions: Path to questions JSON
#   --output: Directory to save episodes
#   --n-consistency: Number of consistency traces (default: 3)
#   --verified-only: Only save verified episodes
#   --model: Teacher model (default: openai/gpt-oss-120b)
#   --temperature: Sampling temperature (default: 0.7)
```

#### Output Format

Each episode is saved as `{uuid}.json`:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:00",
  "question": "What is the mean TL for control?",
  "hint": "Filter to control, then calculate mean",
  "teacher_trace": {
    "question": "...",
    "hint": "...",
    "code_cells": ["df_control = df[df['TR'] == 'control']", "..."],
    "artifacts": {
      "df_control": {
        "name": "df_control",
        "hash": "a1b2c3d4",
        "type": "DataFrame"
      },
      "mean_tl": {
        "name": "mean_tl",
        "hash": "e5f6g7h8",
        "type": "scalar"
      }
    },
    "final_answer": 1.234,
    "final_answer_hash": "e5f6g7h8",
    "execution_success": true
  },
  "consistency_traces": [
    { /* trace 1 */ },
    { /* trace 2 */ },
    { /* trace 3 */ }
  ],
  "verified": true,
  "difficulty": "EASY"
}
```

---

## Testing

### Quick Test: `test_triangulation.py`

Tests a single question to verify triangulation logic:

```bash
python test_triangulation.py
```

**Expected output:**
```
Running teacher triangulation...
Question: What is the mean TL (total length) for the control group?
Hint: Filter the data to the control group first, then calculate the mean.

============================================================
RESULTS
============================================================

Gold Trace (with hint):
  Success: True
  Final Answer: 1.234
  Final Hash: a1b2c3d4e5f6g7h8
  Code Cells: 2
  Artifacts: 3

Consistency Traces (without hint):
  Trace 1:
    Success: True
    Final Answer: 1.234
    Final Hash: a1b2c3d4e5f6g7h8
  Trace 2:
    Success: True
    Final Answer: 1.234
    Final Hash: a1b2c3d4e5f6g7h8

Verification: ✓ PASSED

The gold trace matches the majority of consistency traces!
This question is verified and ready for training.
```

---

## Architecture Notes

### Why Fresh Kernels Per Trace?

Each trace gets a fresh `JupyterKernel()` to ensure:
- No state pollution between traces
- Isolated execution environments
- Clean artifact snapshots

### Why Majority Voting?

Using majority voting (instead of unanimous agreement) handles:
- **Stochastic models**: Same question, different valid approaches
- **Ties**: If 2/3 traces agree, that's good enough
- **Robustness**: One outlier trace doesn't invalidate the question

### Hash-Based Comparison

We compare **hashes** instead of raw values because:
- Handles floating point precision issues
- Works for complex DataFrames
- Deterministic and fast

---

## What's Next (Phase 6)

Phase 6 will add:
- Episode-based student training loop
- Scavenger hunt reward calculation during training
- Episode loading/batching utilities
- Training metrics and logging

This is a "different beast" (as you noted) because it involves:
- RL training infrastructure
- Reward shaping
- Episode replay
- Policy gradient optimization

We can discuss the design for Phase 6 when you're ready!

---

## Summary

**Phase 4** gives us **quality control** - only verified questions make it to training.

**Phase 5** gives us **scalability** - batch process hundreds of questions into episodes.

Together, they form the **data pipeline** that feeds the student training loop.
