# Episode JSONL Schema

> **Purpose**: This document defines the canonical schema for training data episodes.  
> **Location**: `episodes/episodes.jsonl` (one JSON object per line)

---

## Top-Level Episode

```json
{
    "episode_id": "uuid-string",
    "timestamp": "2025-12-21T16:00:00.000000",
    "verified": true,
    
    "question": { /* Question Schema */ },
    "teacher_gold_trace": { /* ExecutionTrace Schema */ },
    "consistency_traces": [ /* list of ExecutionTrace */ ],
    
    "conversation_for_sft": {
        "system_prompt": "string",
        "messages": [ /* OpenAI message format */ ]
    },
    
    "rl_verification_data": {
        "expected_final_answer_hash": "16-char-hex",
        "expected_final_answer": "any JSON value"
    },
    
    "triangulation_metadata": {
        "n_consistency_runs": 5,
        "n_consistency_succeeded": 4,
        "majority_answer_hash": "16-char-hex",
        "majority_count": 4,
        "gold_matches_majority": true
    }
}
```

---

## Question Schema

```json
{
    "id": "16-char-hex",           // Deterministic hash of question_text + hint
    "question_text": "string",
    "hint": "string | null",
    "difficulty": "EASY | MEDIUM | HARD | VERY_HARD | null",
    "n_steps": "int | null",
    "created_at": "datetime | null"
}
```

**ID Generation**: `sha256(f"{question_text}|{hint or ''}")[:16]`

---

## ExecutionTrace Schema

```json
{
    "code_cells": ["string"],       // Raw Python code per turn
    "final_answer": "any",          // The submit() value
    "final_answer_hash": "16-char-hex | null",
    "execution_success": true,
    
    "hooks": [                      // Intermediate checkpoints for RL
        {
            "code_line": "string",
            "variable_name": "string | null",
            "value_hash": "16-char-hex",
            "description": "string | null",
            "depends_on": ["hook_name", ...]  // DAG edges
        }
    ],
    
    "submission_metadata": {},      // Raw metadata from submit()
    "total_turns": 0,
    "archived_turn_count": 0
}
```

---

## Hook Schema (Detail)

| Field | Type | Description |
|-------|------|-------------|
| `code_line` | string | The code that produced this checkpoint |
| `variable_name` | string? | Variable name (e.g., `df_filtered`) |
| `value_hash` | string | 16-char hex hash of normalized value |
| `description` | string? | Semantic description of the checkpoint |
| `depends_on` | list[string] | Names of hooks this depends on (DAG ordering) |

**Hash Generation**: `sha256(json.dumps(normalize_value(x), sort_keys=True))[:16]`

---

## Verified Episode Guarantees

An episode with `verified: true` means:

1. Gold trace's `final_answer_hash` matches majority of consistency traces
2. Teacher successfully called `submit()` in gold trace
3. At least one consistency trace also matched

---

## Usage in Training

### SFT (Supervised Fine-Tuning)
Use `conversation_for_sft`:
- `system_prompt`: Set as system message
- `messages`: User/assistant turns for next-token prediction

### RL (Reinforcement Learning)  
Use `rl_verification_data` + `teacher_gold_trace.hooks`:
- Compare student's final answer hash against `expected_final_answer_hash`
- Compare student's intermediate hooks against stored hook `value_hash` values
- Use `depends_on` to validate ordering (optional, for dense reward)
