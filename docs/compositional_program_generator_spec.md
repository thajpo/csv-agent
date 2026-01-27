# Compositional Program Generator Spec (v2, Typed Grammar Search)

Status: implementation plan for Option B (grammar search + enumeration)
Scope: True compositional generation, no arbitrary selection, 50 programs post-filter target

---

## 1) Core Requirements

- Programs are discovered via **typed grammar search**, not hardcoded schemas
- All selection is **property-based or enumeration-only** - no arbitrary "first column"
- Operator chains built via BFS/DFS with **depth limit = 6**
- Enumerate **all eligible column bindings** (no cap unless compute requires)
- Target: **50 programs post-filter** (fallback 20 if insufficient)

## 2) Non-goals (Phase-1)

- No ambiguity bands or multi-outcome acceptance
- No repair policies (fail -> reject)
- No learned sampling policies
- No LLM verbalization (mechanical question framing only)
- No arbitrary selection by index or position

---

## 3) Program Generation Method

**Typed grammar search:**
- Operators have input/output types and preconditions
- Build operator chains via BFS/DFS, respecting type compatibility
- Depth limit: 6
- Must include >=1 selector + >=1 analysis
- Decision operators in subset of chains (System-2 coverage)

**Enumeration (critical):**
- If multiple eligible columns exist, enumerate ALL
- Programs are built per binding: each (cat_col, num_col) pair -> separate ProgramSpec
- This yields 20-50+ programs per dataset (over-generation)

**Why enumeration:**
- Prevents teaching arbitrary "first" selections to models
- Compositional programs are not hand-coded; they are discovered

---

## 4) Structure Types

- Table, NumCols, CatCols, NumCol, CatCol, Groups
- Scalar, Bool, Dict
- Used only for pruning/validation

---

## 5) Operator Library (expanded)

Selectors:
- select_numeric_cols: Table -> NumCols
- select_categorical_cols: Table -> CatCols
- bind_numeric_col: NumCols -> NumCol (enumerates all numeric cols)
- bind_binary_cat_col: CatCols -> CatCol (enumerates all binary cat cols with support)
- pick_numeric_by_variance(max|min): NumCols -> NumCol
- pick_numeric_by_skew(max|min): NumCols -> NumCol
- pick_categorical_by_cardinality(min|max): CatCols -> CatCol

Grouping/Aggregation:
- groupby_mean
- groupby_var
- argmax_group / argmin_group

Evidence/Test:
- shapiro_p
- levene_p
- ttest_ind
- mannwhitney

Decision:
- choose_test

Attributes: selector, analysis, evidence, decision, test

---

## 6) Grammar Search

BFS/DFS algorithm:
- Start with Table state
- At each step, extend chain if operator.inputs ⊆ current_types and precondition passes
- Stop at depth 6 or when no valid extensions
- Track chain composition: must include selector + analysis

Decision coverage:
- At least 20% of chains should include decision operators (System-2 requirement)

---

## 7) Post-Execution Filters

**Validity gates (always applied):**
- No NaN/empty outputs
- Variance > 0 for numeric targets
- Min group size >= 20
- Min rows >= 30

**Signal filters:**
- Group diff: p_value <= 0.05
- Ranking: unique winner (no ties within tolerance)

**Program count policy:**
- If >=50 programs post-filter, keep top-K by interestingness (effect size, abs(corr), gap)
- If 20-49, keep all
- If <20, keep all and log insufficient count

---

## 8) Guardrails (Tests prevent drift)

**Required tests:**
- No arbitrary selection: if >=2 eligible binary cat cols -> >=2 programs differ by cat_col
- Program count floor: >=50 post-filter (or >=20 if insufficient)
- System-2 coverage: at least one decision-based program per dataset
- Compilation + execution: at least one program executes

**Static checks:**
- Ban "sorted()[0]" in selection unless preceded by property ranking
- Ban "first eligible" logic in sampler

---

## 3) Core Model

Programs are typed operator chains. Valid composition is determined by:
- Type compatibility (outputs of op[i] satisfy inputs of op[i+1])
- Operator preconditions (dataset profile + compile-time state)
- Composition rules (must include selector + analysis; optional decision)

### ProgramSpec (minimal)

```
@dataclass
class ProgramSpec:
    name: str
    ops: list[OpInstance]
    output_type: str
    output_schema: str
    difficulty: str
    tags: list[str] = field(default_factory=list)
```

Notes:
- No required "family" field. Families may appear later as optional metadata.
- ProgramSpec must be serializable and stable for caching/hashing.

### Op + OpInstance

```
@dataclass
class Op:
    name: str
    inputs: list[str]
    outputs: list[str]
    attributes: list[str]
    precondition: Callable[[Profile, State], bool]
    emit: Callable[[State], str]
    update: Callable[[State], None]

@dataclass
class OpInstance:
    op_name: str
    params: dict[str, Any]
```

### State (compile-time)

State stores bindings and metadata only (not runtime data):
- selected columns (names)
- group sizes / counts
- evidence values computed earlier (p-values, outlier scores)
- variable names for emitted code

---

## 4) Types (Phase-1 Minimal)

- Table
- NumCols
- CatCols
- NumCol
- CatCol
- Groups
- Vec
- Scalar
- Bool
- Dict

Types are abstract and only used for pruning/validation.

---

## 5) Operator Attributes (replace "families")

Attributes are semantic tags used for composition rules:

- selector: selects columns or subsets
- transform: changes values (not used in Phase-1)
- analysis: computes final or intermediate stats
- evidence: produces evidence for a decision (p-values, etc.)
- decision: chooses method or path based on evidence
- test: statistical test (uses decision output)

Composition policy uses attributes instead of a family label.

---

## 6) Phase-1 Operator Library

Minimal set for smoke tests (add only as needed):

Selectors
- select_numeric_cols: Table -> NumCols
- select_categorical_cols: Table -> CatCols
- pick_numeric_by_variance(strategy=max|min): NumCols -> NumCol
- pick_binary_category(min_count=20): CatCols -> CatCol

Evidence / Decision
- shapiro_p: NumCol -> Scalar (p-value)
- levene_p: Groups -> Scalar (p-value)
- choose_test: Scalar(normality_p) + Scalar(equal_var_p) -> Dict {"test": "ttest"|"mwu"}

Analysis / Tests
- groupby_values: Table + CatCol + NumCol -> Groups
- ttest_ind: Groups -> Dict {"stat": float, "p_value": float}
- mannwhitney: Groups -> Dict {"stat": float, "p_value": float}
- mean: NumCol -> Scalar
- groupby_mean: Table + CatCol + NumCol -> Dict {group -> mean}
- argmax_group: Dict -> Dict {"group": str, "value": float}

Notes:
- Tie-breaking must be deterministic (alphabetical column names).
- ID-like columns must be excluded using existing profile logic.

---

## 7) Composition Rules (Phase-1)

- Program length: 3-8 ops
- Must include at least one selector
- Must include at least one analysis op
- If a test op is present, required evidence ops must precede it
- If a decision op is present, it must consume evidence ops and feed the test op
- No repair ops, no ambiguity ops

---

## 8) Phase-1 Program Catalog (Smoke Tests)

These are concrete operator chains the sampler must be able to build and compile.

### P1: Most-Variable Mean (System-1)

Chain:
- select_numeric_cols
- pick_numeric_by_variance(max)
- mean

Output schema:
```
{"column": "<name>", "mean": 0.0}
```

### P2: Group Max Mean (System-1)

Chain:
- select_categorical_cols
- pick_binary_category (or pick_categorical_by_cardinality if available)
- select_numeric_cols
- pick_numeric_by_variance(max)
- groupby_mean
- argmax_group

Output schema:
```
{"group": "<name>", "mean": 0.0}
```

### P3: Evidence-Conditioned Group Test (System-2)

Chain:
- select_categorical_cols
- pick_binary_category
- select_numeric_cols
- pick_numeric_by_variance(max)
- groupby_values
- shapiro_p
- levene_p
- choose_test
- ttest_ind OR mannwhitney (chosen by decision)

Output schema:
```
{"test": "ttest|mwu", "stat": 0.0, "p_value": 0.0}
```

Phase-1 sampler must produce at least one P3 program per dataset where preconditions are met.

---

## 9) Compiler Requirements

Compiler inputs:
- ProgramSpec
- dataset profile

Compiler outputs:
- Executable code string with hooks

Rules:
- Use deterministic variable naming (v0, v1, v2, ...)
- Exclude ID-like columns using profile-derived list
- Apply deterministic tie-breaks (sort col names; pick first on ties)
- Emit hook() for evidence + decision values
- Final line must call submit(answer) with required schema

Hook guidance:
- Evidence hooks: hook(value, name="normality_p")
- Decision hooks: hook(choice, name="chosen_test")

No repairs or ambiguity logic in Phase-1.

---

## 10) Sampler (Phase-1)

Phase-1 sampler is deterministic, not learned:

1) Build profile
2) Determine eligible numeric/categorical columns (exclude ID-like)
3) For each program template in the Phase-1 catalog:
   - Check preconditions (e.g., binary categorical exists, numeric cols exist)
   - Instantiate OpInstances with fixed params
   - Emit ProgramSpec

If preconditions fail, skip program. No repairs.

---

## 11) Question Generation (Mechanical, Hypothesis Framing)

Question text must express intent, not method. No step leakage.

Rules:
- Avoid method words: "compute", "test", "correlation", "regression", etc.
- Use hypothesis phrasing: "Is there evidence that...", "Which group seems..."
- Never mention column names
- Always append output schema example

Example for P3:
"Is there evidence that the two groups differ in their typical values? Return as JSON, e.g.: {\"test\": \"ttest\", \"stat\": 0.0, \"p_value\": 0.0}"

---

## 9) Compiler Requirements

Compiler inputs:
- ProgramSpec
- dataset profile

Compiler outputs:
- Executable code string with hooks

Rules:
- Use deterministic variable naming (v0, v1, v2, ...)
- Exclude ID-like columns using profile-derived list
- Apply deterministic tie-breaks (sort col names; pick first on ties)
- Emit hook() for evidence + decision values
- Final line must call submit(answer) with required schema

Hook guidance:
- Evidence hooks: hook(value, name="normality_p", code_line="...")
- Decision hooks: hook(choice, name="chosen_test", code_line="...")
- Column hooks (optional): hook(value, name="cat_col", code_line="...")

No repairs or ambiguity logic in Phase-1.

---

## 10) Sampler (v2 - grammar search)

Sampler uses typed grammar search:

1) Build profile
2) Enumerate eligible columns:
   - numeric: variance > 0, not ID-like
   - categorical: binary and min support >= 20
3) Run BFS/DFS to depth 6
4) For each chain, enumerate ALL valid bindings
5) Yield ProgramSpec for each binding
6) Compile, execute, filter

No hardcoded program catalog.

---

## 11) Question Generation (Mechanical, Hypothesis Framing)

Question text must express intent, not method. No step leakage.

Rules:
- Avoid method words: "compute", "test", "correlation", "regression", etc.
- Use hypothesis phrasing: "Is there evidence that...", "Which group seems..."
- Never mention column names
- Always append output schema example

---

## 12) Episode Integration

Use existing execution + episode pipeline:
- Execute compiled code in LocalCSVAnalysisEnv
- Collect hooks and final answer
- Build episodes that pass tests/test_episode_contract.py

Schema compatibility:
- Store program name in question.template_name
- Include program ops list in question.template_params

---

## 13) Smoke Test Criteria

Success = all of the following:
- At least 20 programs compile on data/csv/data.csv
- At least 1 program executes successfully
- Generated episodes pass tests/test_episode_contract.py
- At least one decision-based program is present when preconditions allow

---

## 14) Implementation Checklist

Phase-1 (Option B implementation):
- [ ] Expand operator library (12-15 operators)
- [ ] Implement typed grammar search (grammar.py)
- [ ] Implement enumeration of column bindings (no arbitrary selection)
- [ ] Implement deterministic filters (filter.py)
- [ ] Add guardrail tests (no arbitrary selection, count floor)
- [ ] Integrate with generator.py (--question-source programs flag)

---

## 15) Deferred (Phase-2+)

Ambiguity bands + multi-outcome hashes
Repair policies (dropna, impute, log shifts)
Learned or probabilistic sampling
LLM verbalization
Meta-reasoning programs (policy robustness)
