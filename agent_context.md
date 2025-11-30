# CSV Agent Plan

## Human Snapshot (read this first)
- Goal: generate a dataset of rich data-science episodes (question + hooks) validated by an oracle, then train a small model to behave like a disciplined analyst.
- Shape: each episode is a multi-hook DAG (2-4 hooks) over one CSV; oracle runs hooks deterministically and compares to teacher answers.
- V1 scope: built-in tools (`group_stat`, `correlation`, `count_filter`, `model_eval` with fixed seeds) plus `python_code` that only touches previous hook results. No hidden hooks, no tool evolution, no corruption variants in V1.
- Outcome: validated episodes in JSONL ready for SFT, followed by GRPO using hook-level rewards.

## V1 Scope & Guardrails
- Hooks: 2-4 per episode; DAG with explicit `depends_on`; `python_code` can only read prior hook results and must use declared deps.
- Determinism: seeds on any modeling; tools are stateless and operate only on `(df, params)`.
- Validation: oracle ground truth is final; numeric tolerance ≈5% relative; partial credit via per-hook correctness.
- Data limits: cap rows/cols when necessary; avoid fragile index-based logic.
- Out of scope for V1: hidden eval hooks, behavioral regularizers/coverage metrics, corruption variants, tool-voting/evolution pipelines, multi-model zoo.

## Milestones
1) Oracle core: dataclasses, built-in tools, `run_hooks` with topo sort, validation helper.  
2) Teacher integration: prompt/schema update, JSON parsing + retries, logging of validation errors.  
3) Episode packaging: JSONL export with metadata; small hand-authored set for sanity checks.  
4) Training: SFT on validated traces; GRPO with hook-average reward (post-V1 if needed).

## LLM Reference Context

### Core Loop
1. Bootstrap EDA on CSV (shape, dtypes, describe, value counts) and show to teacher.
2. Teacher explores, then emits: `question_text`, `difficulty`, `hooks`, `teacher_answers`, `solution_trace`.
3. Oracle runs hooks on df → `oracle_results`; compare to `teacher_answers`; reject on mismatch or execution error.
4. Package accepted episodes; optionally later generate corruption variants and re-run oracle.

### Schemas (authoritative)
```python
from dataclasses import dataclass, field
from typing import Literal, Any

@dataclass
class HookSpec:
    id: str
    tool: Literal["group_stat", "correlation", "model_eval", "count_filter", "python_code"]
    params: dict = field(default_factory=dict)   # for df tools
    code: str | None = None                      # only for python_code
    depends_on: list[str] = field(default_factory=list)

@dataclass
class OracleResult:
    value: float | str | bool | int
    metadata: dict = field(default_factory=dict)

@dataclass
class Episode:
    episode_id: str
    dataset_id: str
    question_text: str
    difficulty: Literal["MEDIUM", "HARD", "VERY_HARD"]
    hooks: list[HookSpec]
    ground_truth: dict[str, Any]      # oracle outputs per hook
    teacher_answers: dict[str, Any]   # teacher claims, compared to oracle
    solution_trace: str
    n_turns: int
    total_tokens: int
    generation_timestamp: str
    teacher_model: str
    corruption_level: int = 0
    corruption_metadata: dict = field(default_factory=dict)
```

### Tool Semantics (V1)
- `group_stat`: optional `filter_expr`, `target_col`, `group_col`, `group_val`, `agg` (`mean`, `median`, `sum`, `count`, `std`); returns stat + n. Reject empty groups.
- `correlation`: optional `filter_expr`, `col_a`, `col_b`, `method` (`pearson`/`spearman`); requires numeric, returns r/p/n.
- `count_filter`: `filter_expr`; returns count.
- `model_eval`: `filter_expr`, `target_col`, `feature_cols`, `model` (start with `linear_regression`), `metric` (`mse`, `mae`, `r2`, or `accuracy` for classification), `seed`; fixed train/test split.
- `python_code`: pure function over `results` from depends_on; no df access, no randomness, no I/O; must reference its inputs.

### Validation / Reward
- Execute hooks in topo order; fail fast on missing deps or invalid params.
- Compare oracle vs teacher answers per hook with float tolerance; episode validity = all hooks match.
- Training reward = average per-hook correctness; keep deterministic seeds to avoid drift.

### Future (post-V1, optional)
- Hidden eval hooks for anti-hacking; corruption variants (noise/missing/shuffle/schema); behavioral coverage metrics; tool-evolution via offline voting.

# Design: Multi-Hook Oracle

## Core idea

One episode = one DS question + a small DAG of hooks (2-4).

Hooks are the verification surface. The question contextualizes them but the oracle only sees hooks.

```
Episode
├── question_text: "Compare treatment effects on TL..."
├── hooks: [h1, h2, h3]  ← what gets verified
├── teacher_answers: {h1: 0.73, h2: 0.41, h3: true}  ← claims
├── ground_truth: {h1: 0.73, h2: 0.41, h3: true}  ← from oracle
└── solution_trace: "code + reasoning"
```

Oracle runs hooks, produces ground truth. Teacher answers are just claims to validate.

## Execution flow

1. Teacher explores CSV → emits question, hooks, answers, trace
2. Oracle runs `run_hooks(df, hooks)` in topological order
3. Compare oracle vs teacher (5% tolerance for floats)
4. Valid if all hooks match → save with oracle's ground truth

Reward = mean per-hook correctness. Not binary pass/fail.

## Dataclasses

```python
@dataclass
class HookSpec:
    id: str
    tool: Literal["group_stat", "correlation", "model_eval", "count_filter", "python_code"]
    params: dict = field(default_factory=dict)
    code: str | None = None  # only for python_code
    depends_on: list[str] = field(default_factory=list)

@dataclass
class OracleResult:
    value: float | str | bool | int
    metadata: dict = field(default_factory=dict)

@dataclass
class Episode:
    episode_id: str
    dataset_id: str
    question_text: str
    difficulty: Literal["MEDIUM", "HARD", "VERY_HARD"]
    hooks: list[HookSpec]
    ground_truth: dict[str, Any]
    teacher_answers: dict[str, Any]
    solution_trace: str
    n_turns: int
    total_tokens: int
    generation_timestamp: str
    teacher_model: str
    corruption_level: int = 0
    corruption_metadata: dict = field(default_factory=dict)
```

## Hook execution

- Topological sort by `depends_on`
- Built-in tools get `(df, params)`
- `python_code` gets `results` dict from deps only—no df, no randomness, no I/O

## Tool semantics (V1)

| Tool | Params | Returns |
|------|--------|---------|
| `group_stat` | filter_expr, target_col, group_col, group_val, agg | stat, n |
| `correlation` | filter_expr, col_a, col_b, method | r, p, n |
| `count_filter` | filter_expr | count |
| `model_eval` | filter_expr, target_col, feature_cols, model, metric, seed | metric, n_train, n_test |
| `python_code` | code, depends_on | value |

All deterministic. Seeds required for model_eval. Empty groups rejected.

## Validation

```python
def values_match(oracle, teacher, rel_tol=0.05):
    if isinstance(oracle, float):
        return math.isclose(oracle, teacher, rel_tol=rel_tol)
    return oracle == teacher
```

Episode valid only if every hook matches.

## Safety

- No tool sees question text or other hooks' params
- No bulk-leak tools (each returns one measurement)
- `python_code` must use its declared deps
- Determinism prevents drift

## Deferred

- Hidden eval hooks
- Corruption variants
- Coverage metrics / behavioral regularizers
- Tool evolution pipeline

# Ongoing

## Where we are
- Bootstrap EDA + teacher exploration loop work.
- Docs aligned to multi-hook design.
- Oracle and tools: not yet implemented.

## V1 scope
- 2-4 hooks per episode, DAG with explicit deps
- Tools: `group_stat`, `correlation`, `count_filter`, `model_eval`, `python_code`
- `python_code` operates on hook results only, not df
- Oracle = ground truth, ~5% float tolerance, partial credit per hook

## Deferred
- Hidden eval hooks, corruption variants
- Behavioral regularizers, coverage metrics
- Tool evolution pipeline

## Next
1. `HookSpec`, `OracleResult`, `Episode` dataclasses
2. Built-in tools (group_stat, correlation, count_filter, model_eval w/ seeds)
3. `run_hooks` with topo sort + validation
4. Teacher prompt update + JSON parsing
5. Hand-author a few episodes to sanity-check oracle

# Tool Design

## Why tools?

Tools constrain the action space. That's the point.

- **Verifiable**: oracle can say correct/incorrect definitively
- **Analyzable**: we can see what the model does (tool distribution, filter usage)
- **Bounded**: no arbitrary code, no I/O, no side effects

The tradeoff is flexibility. The model won't learn general Python. It learns to use *this* tool vocabulary well. For disciplined DS behavior, that's fine.

## Design rules

1. **Inputs**: `(df, params)` only. No question text, no other hooks' answers.
2. **Outputs**: one measurement (stat, metric, count). No tables, no free-form blobs.
3. **Deterministic**: stateless, seed randomness, no caches.
4. **Sandboxed**: `python_code` gets prior hook results only, not df.

## V1 tools

| Tool | What it does | Returns |
|------|--------------|---------|
| `group_stat` | filter → group → aggregate | stat + n |
| `correlation` | Pearson/Spearman between cols | r, p, n |
| `count_filter` | count rows matching filter | count |
| `model_eval` | train/test with fixed seed | metric + split sizes |
| `python_code` | compose over hook results | value |

## Anti-hacking

- Tools can't see question text or emit arbitrary payloads
- Reject degenerate hooks (empty groups, `python_code` ignoring inputs)
- Vary episode design (effect directions, null effects) so "always answer X" fails

## Future

Tool set will evolve. But offline—log traces, find gaps, add tools in batches. Not mid-training.