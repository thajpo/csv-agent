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
