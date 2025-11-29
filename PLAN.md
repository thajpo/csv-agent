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
