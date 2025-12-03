# CSV Agent Roadmap

> Consolidated from ongoing.md, agent_context.md, and planning discussions.
> Last updated: Dec 3, 2025

## Where We Are

**Working:**
- Tool library (`src/tools.py`) - exploration + data query + hook-chaining tools
- Main loop (`src/rich_pipeline.py`) - teacher explores, calls tools, outputs episodes
- Prompt system (`src/prompts.py`) - guides teacher through exploration → episode generation
- Text extraction (`src/text_extraction.py`) - parses `<code>` blocks and JSON output

**The gap:** We generate episodes, but we can't *verify* them. The teacher writes `teacher_answers` but nothing checks if they're actually correct. This is the critical missing piece.

---

## Priority 1: Hook Executor + Verifier

This is the backbone. Everything else (best-of-N, RL rewards, filtering) needs this.

### What it does
```python
# Input: episode dict with hooks, dataframe
# Output: computed answers, comparison to teacher_answers, pass/fail

result = execute_episode(episode, df)
# result.computed_answers = {"ctrl_mean": 14.25, "ctrl_std": 3.82, ...}
# result.matches = {"ctrl_mean": True, "ctrl_std": True, ...}  # within tolerance
# result.valid = True  # all invariants passed
```

### Concrete tasks

**Task 1.1: Define Episode TypedDict** (~30 min)
```python
# src/types.py (new file)
from typing import TypedDict, Literal

class HookParams(TypedDict, total=False):
    group_col: str
    target_col: str
    agg: str
    # ... all possible params

class Hook(TypedDict):
    id: str
    tool: str
    params: HookParams
    depends_on: list[str]

class Episode(TypedDict):
    question_text: str
    difficulty: Literal["MEDIUM", "HARD", "VERY_HARD"]
    hooks: list[Hook]
    teacher_answers: dict[str, str]
    solution_trace: str
```

**Task 1.2: Topological sort for DAG execution** (~30 min)
```python
def topo_sort(hooks: list[Hook]) -> list[Hook]:
    """Order hooks so dependencies come first."""
    # Kahn's algorithm or DFS
```

**Task 1.3: Execute single hook** (~1 hr)
```python
def execute_hook(hook: Hook, df: pd.DataFrame, prior_results: dict[str, Any]) -> Any:
    """
    Run one hook. For chain tools (combine, lookup, aggregate_hooks),
    resolve references from prior_results.
    """
    tool_name = hook["tool"]
    params = hook["params"].copy()
    
    # If chain tool, substitute hook references with actual values
    if tool_name == "combine":
        # Replace vars like {"s": "ctrl_std"} with actual values
        ...
    
    # Call the tool
    return run_tool(tool_name, df, params)
```

**Task 1.4: Execute full episode** (~1 hr)
```python
def execute_episode(episode: Episode, df: pd.DataFrame) -> ExecutionResult:
    """Execute all hooks in order, collect results."""
    hooks = topo_sort(episode["hooks"])
    results = {}
    
    for hook in hooks:
        try:
            results[hook["id"]] = execute_hook(hook, df, results)
        except Exception as e:
            results[hook["id"]] = f"ERROR: {e}"
    
    return ExecutionResult(
        computed=results,
        matches=compare_to_teacher(results, episode["teacher_answers"]),
        valid=check_invariants(results)
    )
```

**Task 1.5: Comparison with tolerance** (~30 min)
```python
def values_match(computed: str, teacher: str, rel_tol: float = 0.05) -> bool:
    """
    Compare computed vs teacher answer.
    Handle: "14.25 (n=414)" vs "14.2500 (n=414)"
    Extract numeric part, compare with tolerance.
    """
```

**Task 1.6: Invariant checks** (~30 min)
```python
def check_invariants(results: dict) -> tuple[bool, list[str]]:
    """
    - Counts >= 0
    - Correlations in [-1, 1]
    - Percentages in [0, 100]
    - No NaN where unexpected
    """
```

### Deliverable
New file: `src/executor.py` with tests in `tests/test_executor.py`

---

## Priority 2: Separate Q&A in Main Loop

Current flow: Teacher explores → proposes questions → answers them (all in one pass)

Problem: Teacher can hallucinate answers that "sound right" but don't match tool output.

New flow:
1. **Explore phase**: Teacher explores data, proposes questions + hook DAGs
2. **Execute phase**: We run the hooks ourselves, get ground truth
3. **Filter phase**: Compare teacher_answers to computed, keep good episodes

### Concrete tasks

**Task 2.1: Split prompt into explore-only** (~1 hr)
- New prompt that stops at "output your questions and hook DAGs as JSON"
- No `teacher_answers` field yet

**Task 2.2: Compute answers via executor** (~30 min)
- After getting proposed episodes, run `execute_episode` on each
- Fill in `computed_answers` field

**Task 2.3: Best-of-N sampling** (~1 hr)
- Generate N=4 versions of each question's hook DAG
- Execute all, keep the ones that agree / pass invariants

### Deliverable
Refactored `rich_pipeline.py` or new `src/pipeline_v2.py`

---

## Priority 3: Output Format for Training

Decide exactly what the JSONL looks like for downstream SFT/RL.

### Questions to answer
- Do we include the full conversation or just final episodes?
- Do we include `computed_answers` (ground truth) or `teacher_answers`?
- What metadata? (dataset_id, timestamp, teacher_model, n_turns)
- Separate format for SFT vs RL?

### Proposed format (SFT)
```json
{
  "dataset_id": "tree_growth",
  "question_text": "Which treatment has lowest CV for TL?",
  "difficulty": "HARD",
  "hooks": [...],
  "ground_truth": {"hook_id": "value", ...},
  "solution_trace": "...",
  "metadata": {"teacher": "grok-4.1-fast", "generated": "2025-12-03"}
}
```

### Proposed format (RL)
Same, but during training you:
1. Give model the question + dataset
2. Let it generate hooks
3. Execute hooks, compute reward = mean(hook correctness)

---

## Priority 4: Context Management

For long exploration runs, context can blow up.

### Already in place
- `TOOL_SPECS` has `summarize: bool` flag per tool
- Large outputs should be truncated

### What's missing
- Actually implementing summarization in the loop
- External memory / conversation buffer
- Caching tool outputs to avoid re-running

### Concrete tasks

**Task 4.1: Implement output truncation** (~30 min)
```python
def maybe_summarize(tool_name: str, output: str, max_lines: int = 20) -> str:
    if TOOL_SPECS[tool_name].get("summarize") and output.count("\n") > max_lines:
        lines = output.split("\n")
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output
```

**Task 4.2: Add to pipeline** (~15 min)
- Call `maybe_summarize` before appending tool output to conversation

---

## Lower Priority / Future

### Model training tool
- Add a simple `baseline_model` tool with fixed split, fixed algo (logistic/linear)
- Returns: accuracy/R², maybe feature importance
- **Only if** the dataset has a clear target variable
- Keep it dead simple: no hyperparams, seeded randomness

### More tools from feedback
- `melt_pattern`: High value for wide datasets (INTERNODE_* columns)
- `row_validate`: Useful but tricky implementation

### Infrastructure
- Unit tests for new tools
- CI pipeline
- Multiple dataset support

---

## Suggested Order of Attack

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Executor basics | `src/executor.py` with topo_sort, execute_hook, execute_episode |
| 1 | Tolerant comparison | `values_match()` handles numeric extraction + tolerance |
| 2 | Invariant checks | `check_invariants()` catches obvious errors |
| 2 | Test on real episodes | Run executor on your existing tool_requests.jsonl output |
| 3 | Split Q&A loop | New pipeline that proposes → executes → filters |
| 3 | Best-of-N | Generate multiple, keep consensus |
| 4 | Output format | Finalize JSONL schema, add validator |
| 4 | Context management | Truncation + caching |

---

## Quick Wins (< 1 hour each)

If you want momentum, start here:

1. **Write the `Episode` TypedDict** - Forces you to think through the schema
2. **Write `topo_sort`** - Classic algorithm, satisfying to implement
3. **Add output truncation** - Immediately useful, easy to test
4. **Add a test for `multi_group_stat`** - You added the tool, now verify it works

---

## Questions to Resolve

- [ ] Should chain tools (`combine`, `lookup`, `aggregate_hooks`) be hookable, or only usable in DAGs?
- [ ] What's the tolerance for float comparison? 5% relative? Absolute?
- [ ] Do we want to support multiple datasets, or stay single-dataset for V1?
- [ ] SFT first, then RL? Or interleave from the start?

