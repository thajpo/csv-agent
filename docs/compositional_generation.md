# Compositional Program Generation - Implementation Summary

## What Was Implemented

### Phase 1: Core Infrastructure (COMPLETE)

#### 1.1 Counter-Based Type System
**File:** `src/datagen/synthetic/programs/types.py`

**Change:** Replaced `Types = frozenset[str]` with `Types = Counter[str]`

**Why:** The old system only tracked presence ("has NumCol"), not quantity ("has 2 NumCols"). This blocked multi-input operators like correlation that need two distinct columns.

**Example:**
```python
# Old (set-based) - can't distinguish 1 vs 2 NumCols
Types = {"Table", "NumCol"}  # Has at least 1 NumCol

# New (Counter-based) - tracks exact counts
Types = Counter({"Table": 1, "NumCol": 2})  # Has exactly 2 NumCols
```

**Key Functions:**
- `is_subtype(actual, required)`: Checks if actual has ≥ required counts
- `add_types(current, outputs)`: Adds output types to current state
- `types_from_list(type_list)`: Creates Counter from list (handles duplicates)

#### 1.2 General Binding System
**File:** `src/datagen/synthetic/programs/enumerate.py`

**Change:** Replaced hardcoded bindings with general binding enumeration

**How It Works:**
1. Inspect operator chain to discover all required binding keys
2. Map each key to appropriate column pool (numeric, categorical, etc.)
3. Generate cartesian product of valid combinations
4. Skip combinations where same column bound to multiple keys

**Example:**
```python
# Chain: select_numeric_cols → bind_num_col_1 → bind_num_col_2 → correlation
# Binding keys discovered: {"num_col_1", "num_col_2"}
# Column pool: ["A", "B", "C"]
# Generated combinations: (A,B), (A,C), (B,A), (B,C), (C,A), (C,B)
```

#### 1.3 Multi-Input Operator Support
**Files:** `src/datagen/synthetic/programs/operators.py`, `grammar.py`

**New Operators:**
- `bind_num_col_1`: Binds first numeric column
- `bind_num_col_2`: Binds second numeric column (must be different)
- `correlation`: Uses two explicitly bound columns
- `ratio`: Computes col1/col2 (another multi-input example)
- `min`, `max`: Additional aggregations

**Design Pattern:**
```python
# Multi-input operator declaration
correlation = Op(
    name="correlation",
    inputs=["NumCol", "NumCol"],  # Needs 2 NumCols (count-based)
    outputs=["Dict"],
    attributes=["analysis"],
    emit=correlation_emit,
    precondition=lambda _profile, _s: True,  # Structure only
    reads={"num_col_1", "num_col_2"},
    writes={"answer"},
    emits_answer=True,
    requires_bindings={},  # Bindings from previous ops
)
```

**Key Insight:** Separation of concerns
- **Grammar:** Validates structural feasibility (types, operator order)
- **Enumeration:** Validates semantic constraints (distinct columns)

### Phase 2: Operator Library (IN PROGRESS)

**Current Operators:** 42 total

**Selector Operators (8):**
- `select_numeric_cols`, `select_categorical_cols`
- `bind_numeric_col`, `bind_binary_cat_col`
- `bind_num_col_1`, `bind_num_col_2` (NEW)
- `pick_numeric_by_variance`, `pick_numeric_by_skew`
- `pick_categorical_by_cardinality`

**Transform Operators (4):**
- `zscore`, `log1p`, `abs`
- `filter_greater_than`

**Analysis Operators (18):**
- Single-input: `mean`, `median`, `std`, `variance`, `sum`, `min`, `max`
- Multi-input: `correlation`, `ratio` (NEW)
- Grouped: `groupby_mean`, `groupby_var`, `groupby_median`, `groupby_std`, `groupby_count`
- Argmax/Argmin: 8 variants for different metrics

**Evidence Operators (2):**
- `shapiro_p`, `levene_p`

**Decision Operators (1):**
- `choose_test`

**Test Operators (1):**
- `ttest_ind`

## How to Add New Operators

### Step 1: Define the Emit Function

```python
def my_operator_emit(state: State) -> str:
    """Emit Python code for the operator.
    
    Access bound columns via state.bindings dict.
    Use hook() for intermediate values.
    End with submit() for final answer.
    """
    col1 = state.bindings.get("num_col_1")
    col2 = state.bindings.get("num_col_2")
    
    return (
        f"# Compute something with {col1} and {col2}\n"
        f"result = df['{col1}'] + df['{col2}']\n"
        'submit({"result": float(result.mean())})'
    )
```

### Step 2: Add to OPERATORS Dict

```python
"my_operator": Op(
    name="my_operator",
    inputs=["NumCol", "NumCol"],  # Use counts for multi-input
    outputs=["Dict"],  # Or "Table", "Scalar", etc.
    attributes=["analysis"],  # Or "selector", "transform", etc.
    emit=my_operator_emit,
    update=lambda s: None,  # Or update state if needed
    precondition=lambda _profile, _s: True,  # Keep simple
    reads={"num_col_1", "num_col_2"},  # What bindings it reads
    writes={"answer"},  # What bindings it writes
    emits_answer=True,  # True if this op produces final answer
    requires_bindings={},  # Or {"my_param": True} if needs binding
),
```

### Step 3: Test

```python
# Check operator exists
from src.datagen.synthetic.programs.operators import OPERATORS
assert "my_operator" in OPERATORS

# Check grammar finds chains
from src.datagen.synthetic.programs.grammar import search_programs
chains = search_programs(profile, max_depth=5)
my_chains = [c for c in chains if any(op.op_name == "my_operator" for op in c)]
print(f"Found {len(my_chains)} chains with my_operator")

# Check full pipeline
from src.datagen.synthetic.programs.program_generator import run_pipeline
questions = await run_pipeline(csv_path="data.csv", max_programs=50)
my_questions = [q for q in questions if "my_operator" in q.get("program_ops", [])]
print(f"Generated {len(my_questions)} questions")
```

## Key Design Decisions

### 1. Counter-Based Types
**Decision:** Use `Counter[str]` instead of `set[str]`
**Rationale:** Multi-input operators need to track quantity, not just presence
**Trade-off:** Slightly more complex type checking, but enables rich composition

### 2. Explicit Bindings
**Decision:** Bindings filled during enumeration, not selected inside operators
**Rationale:** No arbitrary selection, all valid combinations enumerated
**Trade-off:** More programs generated, but all are valid and reproducible

### 3. Separation of Concerns
**Decision:** Grammar validates structure, enumeration validates semantics
**Rationale:** Grammar can find valid chains without knowing specific column values
**Trade-off:** Two-phase process, but cleaner architecture

### 4. Type Accumulation
**Decision:** Types accumulate through chain (not consumed)
**Rationale:** Later operators can use any previously produced type
**Trade-off:** Types can grow large, but enables flexible chaining

## Testing Strategy

### Unit Tests
- Type system: `is_subtype`, `add_types`, `types_from_list`
- Enumeration: `_discover_binding_keys`, `_generate_binding_combinations`
- Operators: Each emit function produces valid Python

### Integration Tests
- Grammar search finds expected chains
- Enumeration produces valid bindings
- Full pipeline generates executable questions
- All operators appear in output at least once

### Coverage Tests
- Operator usage distribution
- Chain length distribution
- Binding combination coverage

## Future Extensions

### Easy Additions
- More aggregations: `percentile`, `nunique`, `count_missing`
- More transforms: `normalize`, `clip`, `fill_na`
- More multi-input: `covariance`, `regression_slope`

### Medium Complexity
- Time-series operators: `rolling_mean`, `lag`, `diff`
- Conditional operators: `if_then_else` (requires branching)
- Multi-table: `join` (requires multiple dataframes)

### Hard Additions
- Looping constructs: `for_each_group` (requires iteration)
- Recursive patterns: `until_convergence` (requires termination)
- External data: `lookup_in_table` (requires external reference)

## Performance Considerations

### Current Limits
- Grammar search: 2000 iterations max (safety limit)
- Chain depth: 6 operators max
- Programs generated: ~200 per dataset

### Scaling Strategies
1. **Sampling:** Sample from binding combinations instead of full enumeration
2. **Filtering:** Filter chains by predicted difficulty/interest before execution
3. **Caching:** Cache compiled programs and execution results
4. **Parallelism:** Execute programs in parallel (already supported)

## Summary

The compositional program generation system now supports:
- ✅ Multi-input operators (correlation, ratio)
- ✅ Explicit binding enumeration (no arbitrary selection)
- ✅ Extensible operator library (42 operators, easy to add more)
- ✅ Counter-based type system (tracks multiplicity)
- ✅ Separation of structure (grammar) and semantics (enumeration)
- ✅ Full pipeline from operators → questions → ground truth

Next steps for scaling:
- Add more operator families (time-series, conditional, etc.)
- Implement sampling strategies for large datasets
- Add difficulty tagging and filtering
- Create coverage reports and tuning loops
