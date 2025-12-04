# Implementation Plan: Two-Phase Question Generation

> For use in a new chat session. This captures the architecture decisions from the Dec 4 planning discussion.

## Overview

Split the current single-phase pipeline into two decoupled phases:
1. **Exploration Phase**: Generate questions + reasoning paths (no formal hooks)
2. **Answering Phase**: Formalize reasoning into hooks, execute, get ground truth

Then validate with best-of-N filtering.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXPLORATION PHASE                           │
│  Input: Dataset + tools                                         │
│  Output: Questions + reasoning paths (natural language)         │
│  Model does: Explore data, discover patterns, design questions  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ANSWERING PHASE (Guided)                    │
│  Input: Question + reasoning path                               │
│  Output: Formal hooks → execute → ground truth values           │
│  Model does: Translate reasoning into hook DAG                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ANSWERING PHASE (Blind) × N                 │
│  Input: Question only (no reasoning hint)                       │
│  Output: Hooks → execute → values                               │
│  Model does: Independently solve the question                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FILTER                                      │
│  Compare: Guided answer vs Blind answers                        │
│  Keep if: ≥2 blind attempts match guided answer (within tol)    │
│  Discard: Ambiguous or poorly-specified questions               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Insight: The Executor Already Exists

The "executor" is just tool calling in sequence. `run_tool()` already works.

What we need:
```python
def execute_hooks(hooks: list[dict], df: pd.DataFrame) -> dict[str, Any]:
    """Execute hooks in dependency order, return {hook_id: value}."""
    ordered = topo_sort(hooks)  # sort by depends_on
    results = {}
    
    for hook in ordered:
        params = resolve_references(hook["params"], results)  # substitute prior values
        output = run_tool(hook["tool"], df, params)
        results[hook["id"]] = parse_value(output)  # extract numeric from "14.25 (n=414)"
    
    return results
```

Components needed:
- `topo_sort(hooks)` - standard topological sort (~15 lines)
- `resolve_references(params, results)` - for chain tools, substitute hook refs with values (~20 lines)
- `parse_value(output)` - extract numeric/string from tool output format (~15 lines)

---

## Output Schemas

### Exploration Phase Output
```json
{
  "question_text": "Which treatment shows the most consistent growth (lowest CV of TL), and how does it compare to control?",
  "difficulty": "HARD",
  "reasoning_path": "1. For each treatment, compute mean and std of TL. 2. Derive CV = std/mean for each. 3. Find treatment with minimum CV. 4. Compute control's CV. 5. Calculate percentage difference.",
  "key_columns": ["TR", "TL"],
  "expected_steps": 5
}
```

No hooks. No computed values. Just the strategy.

### Answering Phase Output
```json
{
  "question_text": "...",
  "hooks": [
    {"id": "ctrl_mean", "tool": "group_stat", "params": {...}, "depends_on": []},
    {"id": "ctrl_std", "tool": "group_stat", "params": {...}, "depends_on": []},
    {"id": "ctrl_cv", "tool": "combine", "params": {"expr": "s/m", "vars": {"s": "ctrl_std", "m": "ctrl_mean"}}, "depends_on": ["ctrl_std", "ctrl_mean"]},
    ...
  ],
  "computed_answers": {
    "ctrl_mean": 14.25,
    "ctrl_std": 5.82,
    "ctrl_cv": 0.408,
    ...
  }
}
```

Formal hooks + executed values.

---

## Implementation Steps

### Step 1: Executor Functions (~50 lines)

Create `src/executor.py`:

```python
def topo_sort(hooks: list[dict]) -> list[dict]:
    """Order hooks so dependencies come first."""
    # Kahn's algorithm or DFS
    
def resolve_references(params: dict, results: dict) -> dict:
    """For chain tools, substitute hook references with actual values."""
    # Handle 'vars' in combine, 'group_hook' in lookup, 'hooks' in aggregate_hooks
    
def parse_value(output: str) -> float | str:
    """Extract numeric value from tool output like '14.25 (n=414)'."""
    # Regex to pull first number, or return string if not numeric
    
def execute_hooks(hooks: list[dict], df: pd.DataFrame) -> dict[str, Any]:
    """Execute all hooks, return computed values."""
```

### Step 2: Exploration Prompt

Create `build_exploration_prompt()` in `src/prompts.py`:
- Same data exploration setup as current
- Output schema: questions + reasoning paths (NO hooks, NO teacher_answers)
- Emphasize: describe HOW to solve, not the formal tool calls

### Step 3: Answering Prompt (Guided)

Create `build_answering_prompt()` in `src/prompts.py`:
- Input: question + reasoning path
- Task: formalize into hooks
- Model outputs hooks, we execute them to get values
- No "teacher_answers" from model - we compute them

### Step 4: Answering Prompt (Blind)

Same as guided but without the reasoning path hint.

### Step 5: Pipeline Refactor

Refactor `src/rich_pipeline.py` to support modes:
- `--mode explore` → exploration phase
- `--mode answer --input questions.jsonl` → answering phase
- `--mode validate --input answered.jsonl --n 4` → best-of-N

### Step 6: Filtering Logic

```python
def filter_episodes(episodes: list, n_attempts: int = 4, min_matches: int = 2):
    """Keep episodes where blind attempts converge to guided answer."""
    for ep in episodes:
        guided = ep["computed_answers"]
        blind_results = [run_blind(ep["question_text"]) for _ in range(n_attempts)]
        matches = sum(1 for b in blind_results if answers_match(b, guided))
        if matches >= min_matches:
            yield ep
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/executor.py` | NEW - topo_sort, resolve_references, parse_value, execute_hooks |
| `src/prompts.py` | ADD - build_exploration_prompt(), build_answering_prompt() |
| `src/rich_pipeline.py` | REFACTOR - support explore/answer/validate modes |
| `src/types.py` | NEW (optional) - TypedDict for Question, Hook, Episode |

---

## Testing Strategy

1. **Test executor first**: Hand-craft a simple hook DAG, verify execute_hooks produces correct values
2. **Test exploration prompt**: Does it output questions + reasoning without hooks?
3. **Test answering prompt**: Given question + reasoning, does it produce valid hooks?
4. **End-to-end**: Run full pipeline, check filter catches bad episodes

---

## Questions to Decide

- [ ] Tolerance for numeric comparison? (suggest: 5% relative or 0.01 absolute)
- [ ] Min matches for best-of-N? (suggest: 2 of 4)
- [ ] Store intermediate results (exploration output, guided output) or pipeline end-to-end?
- [ ] Same model for all phases, or different models?

---

## Priority Order

1. **Executor** - foundation for everything
2. **Exploration prompt** - get questions + reasoning flowing
3. **Answering prompt (guided)** - formalize + execute
4. **Best-of-N filtering** - validate quality
5. **Pipeline integration** - tie it together

Start with executor. It's ~50 lines and unlocks everything else.

