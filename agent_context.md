# CSV Agent: Verified Tool-Learning for Data Science

## Core Thesis

Train a small model to solve data science tasks using constrained tools, with RL on verified execution outcomes.

**Key insight**: Ground truth comes from the *environment* (tool execution), not the teacher's reasoning. The teacher generates task structure; the oracle provides correctness. This means the student can surpass the teacher in execution quality while learning from teacher-generated tasks.

**Why tools, not pandas?**
- Verifiable: scalar outputs, deterministic execution
- Bounded action space: RL can explore effectively
- Atomic operations: clear credit assignment per hook

## Architecture

```
Teacher LLM → generates (question, hook DAG, tool calls)
                            ↓
Environment → executes tools, produces ground truth
                            ↓
Filter → best-of-N, invariant checks, discard bad episodes
                            ↓
Training → SFT warm-start → on-policy GRPO with hook-level rewards
```

## V1 Scope

### Tools (defined in TOOL_SPECS - see src/tools.py)

**Exploration tools** (may produce large output, summarized for context):
| Tool | Purpose | Summarize |
|------|---------|-----------|
| `inspect` | View df structure (head/tail/shape/dtypes/columns/missing/info) | Yes |
| `describe` | Statistical summary for numeric/object/all columns | Yes |
| `value_counts` | Frequency counts for a column | Yes |
| `unique` | List unique values in a column | Yes |
| `sort_values` | Sort by column and show top rows | Yes |
| `crosstab` | Cross-tabulation of two categorical columns | Yes |

**Data query tools** (produce scalar/small output, query dataframe):
| Tool | Params | Returns |
|------|--------|---------|
| `group_stat` | group_col, target_col, agg, filter_expr?, group_val | scalar (with group_val) |
| `group_extremum` | group_col, target_col, agg, extremum, return_what | group name or value |
| `correlation` | col_a, col_b, method?, filter_expr? | r value with interpretation |
| `count_filter` | filter_expr? | count + percentage |
| `quantile` | col, q, filter_expr? | scalar (single q) or series |
| `derive_stat` | formula, group_col, agg, filter_expr?, group_val | derived scalar |

**Hook-chaining tools** (operate on previous hook results, enable multi-step reasoning):
| Tool | Params | Purpose |
|------|--------|---------|
| `combine` | expr, vars (map var→hook_id) | Arithmetic + boolean on hook results (ratios, %, a > b) |
| `lookup` | group_hook, group_col, target_col, agg | Chain group_extremum → group_stat dynamically |
| `aggregate_hooks` | hooks (list), agg | Reduce multiple hooks (min/max/mean/sum/range) |

**Design rules** (from TOOL_SPECS):
- `TOOL_SPECS` in `tools.py` is the source of truth for tool definitions
- Each spec includes `summarize: bool` for context compression  
- Hook-chaining tools have `chain: True` flag
- Auto-coerce '?' to NaN for numeric operations
- Tools are stateless and deterministic

### Episodes
- MEDIUM: 3-4 hooks, HARD: 4-6 hooks, VERY_HARD: 5-10 hooks
- DAG with explicit `depends_on` (required for hook-chaining tools)
- Each hook produces one verifiable value
- Hook-chaining tools enable true multi-step reasoning chains
- Reward = average per-hook correctness

## Data Generation Strategy

### Best-of-N Filtering
For each question:
1. Sample N teacher traces (N=4-8)
2. Execute all, verify hooks
3. Discard if: execution error, invariant fails, hooks contradict
4. Keep traces where ≥2 agree on key values

**Why**: False positives (wrong but rewarded) poison the model. False negatives (correct but no reward) just slow learning. Bias toward throwing away ambiguous episodes.

### Invariant Checks
- Counts non-negative
- Correlations in [-1, 1]
- Proportions in [0, 1]
- Cross-hook consistency

### Difficulty Curriculum
- MEDIUM: 3-4 hooks, linear chain  
- HARD: 4-6 hooks, some branching, uses combine/compare
- VERY_HARD: 5-10 hooks, complex DAG with lookup/aggregate_hooks

Start training on MEDIUM, gradually add harder tasks.

## Training Strategy

### Phase 1: SFT Warm-Start
- Train on filtered teacher traces
- Goal: model can solve easy/medium tasks
- ~5k-20k episodes

### Phase 2: On-Policy GRPO
- Sample tasks from pool (teacher provides structure)
- Roll out student in environment
- Compute hook-level rewards (oracle vs student outputs)
- Update policy toward correct executions
- Generate new rollouts from updated student (on-policy)

**Key**: RL shapes behavior in error regions the teacher never explored. SFT teaches patterns; RL teaches what actually works.

### Reward Design
- Per-hook correctness (dense signal)
- Float tolerance ~5% relative
- Episode reward = mean hook correctness
- Optional: KL penalty to stay near SFT policy

## Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Teacher DAG is semantically wrong | Best-of-N + consensus filtering |
| Teacher uses unstated domain knowledge | Keep questions self-contained; vet with second LLM |
| Systematic false positives | Aggressive filtering, invariant checks |
| Student learns teacher's bad habits | RL reward is environment-based, not teacher-matching |

## What's NOT in V1
- Web search / external knowledge
- Hidden eval hooks
- Corruption variants
- Multi-model ensembles
- Tool evolution mid-training

## Current Status

**Done**:
- Tools: group_stat, group_extremum, derive_stat, correlation, count_filter, etc.
- Teacher exploration prompt
- Episode generation pipeline

**Next**:
1. Simplify tools (ensure scalar outputs)
2. Build filtering pipeline (best-of-N, invariants)
3. Generate pilot batch (~500 episodes)
4. Estimate noise rate empirically
5. SFT baseline
