# Data Capture Architecture

> **Philosophy:** Capture raw data once, derive training formats later.

This document explains what data is captured during episode generation and how it supports multiple training objectives.

---

## Overview

Each episode captures a complete execution trace with enough detail to derive training samples for:

| Training Method | What It Uses | Purpose |
|-----------------|--------------|---------|
| **SFT** | Conversation turns | Teach model to write code given questions |
| **PRM** | Hook values + hashes | Score intermediate reasoning steps |
| **ORM** | Final answer + verified flag | Score final outcomes |
| **DPO/RLHF** | Gold vs consistency traces | Preference learning from trace pairs |
| **Self-correction** | Correction metadata | Train error recovery behavior |

---

## What We Capture

### 1. Turn-Level Data

Each turn in a trace contains:

```
TurnDict:
├── turn_index: int           # Position in conversation
├── reasoning: str            # Model's thinking before code
├── code: str                 # The code block executed
├── execution:
│   ├── success: bool         # Did code run without error?
│   ├── stdout: str           # Execution output
│   ├── stderr: str           # Error messages (if any)
│   ├── hooks: [HookDict]     # Intermediate values (see below)
│   └── submitted_answer: Any # Final answer if submit() called
└── correction: CorrectionDict | null  # Self-correction metadata (if applicable)
```

### 2. Hook Values (for PRM)

Hooks capture intermediate computation states. Storage policy:

| Value Type | What's Stored | Size |
|------------|---------------|------|
| **Scalars** (int, float, str, bool) | Full value | Exact |
| **DataFrame** | Structured summary | ~1-2 KB |
| **Series** | Structured summary | ~500 B |
| **Large dicts/lists** (>100KB) | Type + size metadata | ~50 B |

**DataFrame summary includes:**
- `shape`: [rows, cols]
- `columns`: Column names
- `dtypes`: Column types
- `head`: First 3 rows
- `numeric_summary`: Mean/min/max for numeric columns

**Why summaries?** A PRM can't predict 100K rows, but it CAN predict "shape should be ~300 rows after filtering" or "mean should be around 54.3". Summaries are learnable targets.

**Verification:** `value_hash` is always computed on the full normalized value, so correctness can still be verified even when only a summary is stored.

### 3. Self-Correction Metadata

When a turn successfully fixes a previous failure, we capture:

```
CorrectionDict:
├── corrects_turn: int        # Which failed turn this fixes
├── error_type: str           # "KeyError", "ValueError", etc.
├── error_message: str        # The specific error
├── attempts_since_error: int # Usually 1
└── code_diff:
    ├── removed_lines: [str]  # Lines in failed code
    └── added_lines: [str]    # Lines in fixed code
```

This enables training models to:
1. Recognize errors from feedback
2. Diagnose what went wrong
3. Generate appropriate fixes

### 4. Triangulation Data (for DPO)

Each episode contains:
- `gold_trace`: Teacher run WITH hint (preferred)
- `consistency_traces`: N teacher runs WITHOUT hint

Preference pairs can be derived:
- Gold vs failed consistency → (preferred, dispreferred)
- Successful consistency vs failed → more pairs

---

## Deriving Training Samples

### Standard SFT

```python
for turn in gold_trace["turns"]:
    messages.append({"role": "assistant", "content": f"{turn['reasoning']}\n```python\n{turn['code']}\n```"})
    messages.append({"role": "user", "content": f"[stdout]:\n{turn['execution']['stdout']}"})
```

### PRM (Process Reward Model)

```python
for turn in gold_trace["turns"]:
    for hook in turn["execution"]["hooks"]:
        samples.append({
            "step": hook["code_line"],
            "value": hook["value"],  # Structured summary
            "value_hash": hook["value_hash"],  # For verification
            "label": 1.0 if verified else 0.0
        })
```

### Self-Correction

```python
for turn in gold_trace["turns"]:
    if turn.get("correction"):
        c = turn["correction"]
        failed_turn = gold_trace["turns"][c["corrects_turn"]]
        samples.append({
            "failed_code": failed_turn["code"],
            "error_feedback": failed_turn["execution"]["stderr"],
            "fixed_code": turn["code"],
            "code_diff": c["code_diff"],
        })
```

### DPO/Preference Learning

```python
# Gold (with hint) vs failed consistency (without hint)
for consistency_trace in consistency_traces:
    if not consistency_trace["success"]:
        pairs.append({
            "preferred": gold_trace,
            "dispreferred": consistency_trace,
        })
```

---

## Design Principles

### 1. Capture Once, Train Many

Episodes store raw structured data. Training formats are derived at training time, not pre-baked. This means:
- New training methods can reuse existing episodes
- Schema changes are additive (new optional fields)
- No need to regenerate data for new training approaches

### 2. Hash Everything

Every value has a hash computed on the full normalized representation:
- `value_hash` for hooks
- `final_answer_hash` for answers
- `ground_truth_hash` for synthetic questions

This enables verification without storing full values.

### 3. Bounded Sizes

Large objects (DataFrames, long lists) are summarized to bounded sizes. This keeps episodes manageable while preserving the training signal.

### 4. Explicit Over Implicit

Self-correction data was always captured (failed turns followed by successful turns), but we made it explicit with `CorrectionDict`. This makes it easy to:
- Filter episodes with corrections
- Extract error→fix pairs
- Analyze recovery patterns

---

## File Locations

```
data/
├── episodes/
│   └── episodes.jsonl    # Main training data (one episode per line)
├── questions/
│   ├── synthetic/        # Template-generated questions
│   └── llm/              # LLM-generated questions
└── csv/                  # Source datasets
```

---

## Schema Reference

See [episode_schema.md](episode_schema.md) for detailed field definitions and examples.
