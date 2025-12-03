"""Prompts and templates used throughout the data pipeline."""

import pandas as pd
from src.tools import format_tool_docs, inspect, describe, value_counts


def generate_bootstrap_output(csv_path: str = "data.csv") -> str:
    """
    Generate bootstrap exploration output for initial data inspection.
    
    Returns formatted string showing shape, head, dtypes, summary, missing values,
    and categorical value counts.
    """
    df = pd.read_csv(csv_path)
    lines = []
    
    for label, call in [
        ("SHAPE", lambda: inspect(df, "shape")),
        ("HEAD", lambda: inspect(df, "head", 5)),
        ("DTYPES", lambda: inspect(df, "dtypes")),
        ("NUMERIC SUMMARY", lambda: describe(df, "number")),
        ("MISSING", lambda: inspect(df, "missing")),
    ]:
        lines.append(f"=== {label} ===\n{call()}\n")
    
    lines.append("=== CATEGORICAL VALUE COUNTS ===")
    for col in df.select_dtypes(include=['object']).columns:
        lines.append(f"\n{col}:\n{value_counts(df, col, 5)}")
    
    return "\n".join(lines)


# JSON schema for episodes - shown to the model
# Tools: data query tools + hook-chaining tools
EPISODE_SCHEMA = """
{
  "question_text": "string - multi-step exam question requiring reasoning (NOT a simple lookup)",
  "difficulty": "MEDIUM | HARD | VERY_HARD",
  "hooks": [
    {
      "id": "string - unique, lowercase with underscores (e.g. 'ctrl_mean', 'cv_ratio')",
      "tool": "DATA: group_stat|group_extremum|correlation|count_filter|quantile|derive_stat  CHAIN: combine|lookup|aggregate_hooks",
      "params": {
        "// DATA TOOLS query the dataframe": "",
        "// group_stat": "group_col, target_col, agg?, filter_expr?, group_val",
        "// group_extremum": "group_col, target_col, agg?, extremum?, return_what?, filter_expr?",
        "// correlation": "col_a, col_b, method?, filter_expr?",
        "// count_filter": "filter_expr?",
        "// quantile": "col, q, filter_expr?",
        "// derive_stat": "formula, group_col, agg?, filter_expr?, group_val",
        "// CHAIN TOOLS operate on prior hook results": "",
        "// combine": "expr (arithmetic or boolean), vars (map var→hook_id)",
        "// lookup": "group_hook, group_col, target_col, agg?",
        "// aggregate_hooks": "hooks (list), agg (min|max|mean|sum|range)"
      },
      "depends_on": ["required for chain tools - list hook ids this depends on"]
    }
  ],
  "// hooks requirement": "MEDIUM: 3-4 hooks, HARD: 4-6 hooks, VERY_HARD: 5-10 hooks",
  "teacher_answers": {
    "hook_id": "value - computed answer (number or string)"
  },
  "solution_trace": "string - step-by-step explanation"
}
""".strip()


# Hook tool documentation for episodes (subset of TOOL_SPECS that produce scalar outputs)
HOOK_TOOLS_DOC = """
HOOK TOOLS (for building multi-step reasoning chains):

═══ DATA QUERY TOOLS (operate on dataframe) ═══

**group_stat**: Aggregate a column grouped by another (e.g., mean salary by department)
  - params: group_col*, target_col?, agg (mean|sum|median|std|min|max|count|nunique|condition_rate|missing_rate), filter_expr?, group_val*, condition?
  - group_val is REQUIRED for scalar output (returns "value (n=count)" or "% (...)")
  - condition_rate: provide condition (e.g., "TL > 50") to get % within group
  - missing_rate: target_col is the column to check missingness on
  - Example: {{"tool": "group_stat", "params": {{"group_col": "TR", "target_col": "TL", "agg": "mean", "group_val": "control"}}}}
  - Example (condition rate): {{"tool": "group_stat", "params": {{"group_col": "TR", "agg": "condition_rate", "condition": "TL > 50", "group_val": "control"}}}}
  - Example (missing rate): {{"tool": "group_stat", "params": {{"group_col": "TR", "target_col": "INTERNODE_5", "agg": "missing_rate", "group_val": "control"}}}}

**multi_group_stat**: Aggregate multiple targets with multiple aggs by group
  - params: group_col*, target_cols* (list), aggs* (list), filter_expr?, top_n_cols?
  - returns: compact table with columns like target_agg, one row per group
  - Example: {{"tool": "multi_group_stat", "params": {{"group_col": "TR", "target_cols": ["TL", "IN"], "aggs": ["mean", "std"]}}}}

**group_extremum**: Find which group has highest/lowest aggregated value
  - params: group_col*, target_col*, agg?, extremum (max|min), return_what (group|value), filter_expr?
  - return_what='group' → group name, return_what='value' → the extreme value
  - Example: {{"tool": "group_extremum", "params": {{"group_col": "TR", "target_col": "TL", "extremum": "max", "return_what": "group"}}}}

**correlation**: Correlation coefficient between two numeric columns
  - params: col_a*, col_b*, method (pearson|spearman|kendall), filter_expr?
  - returns: "r_value (strength direction correlation)"
  - Example: {{"tool": "correlation", "params": {{"col_a": "TL", "col_b": "IN"}}}}

**count_filter**: Count rows matching a condition
  - params: filter_expr?
  - returns: "count rows (pct% of total)"
  - Example: {{"tool": "count_filter", "params": {{"filter_expr": "TL > 50 and TR == 'control'"}}}}

**filter_stat**: Aggregate a numeric column with optional filtering (no grouping)
  - params: target_col?, agg (mean|sum|median|std|min|max|count|nunique), filter_expr?
  - agg='count' returns count + percent of total
  - Example: {{"tool": "filter_stat", "params": {{"target_col": "TL", "agg": "mean", "filter_expr": "TR == 'control'"}}}}
  - Example (count): {{"tool": "filter_stat", "params": {{"agg": "count", "filter_expr": "TL > 50 and TR == 'control'"}}}}

**quantile**: Calculate percentile for a column
  - params: col*, q (single float 0-1 for scalar), filter_expr?
  - Example: {{"tool": "quantile", "params": {{"col": "TL", "q": 0.9}}}}

**group_value_counts**: Top-N value counts per group for a categorical column
  - params: group_col*, target_col*, top_n?, normalize?
  - returns: stacked table with group, value, count, pct (if normalize)
  - Example: {{"tool": "group_value_counts", "params": {{"group_col": "TR", "target_col": "IN", "top_n": 5}}}}

**derive_stat**: Compute derived metric from formula, aggregate by group
  - params: formula*, group_col*, agg?, filter_expr?, group_val*
  - formula uses column names (e.g., "TL / IN", "price * quantity")
  - Example: {{"tool": "derive_stat", "params": {{"formula": "TL / IN", "group_col": "TR", "agg": "mean", "group_val": "control"}}}}

═══ HOOK-CHAINING TOOLS (operate on previous hook results) ═══

**combine**: Arithmetic or boolean expressions on hook results
  - params: expr* (expression), vars* (map variable names → hook IDs)
  - Arithmetic: ratios, differences, percentages → returns number
  - Boolean: comparisons (>, <, ==, >=, <=) → returns "true"/"false"
  - Example: {{"tool": "combine", "params": {{"expr": "s / m", "vars": {{"s": "ctrl_std", "m": "ctrl_mean"}}}}}}
  - Example: {{"tool": "combine", "params": {{"expr": "(a - b) / b * 100", "vars": {{"a": "treat_val", "b": "ctrl_val"}}}}}}
  - Example: {{"tool": "combine", "params": {{"expr": "a > b", "vars": {{"a": "ctrl_cv", "b": "treat_cv"}}}}}}

**lookup**: Chain a group result into another query
  - params: group_hook* (hook that returned a group), group_col*, target_col*, agg?
  - Use for: "find best treatment, then get its correlation with X"
  - Example: {{"tool": "lookup", "params": {{"group_hook": "best_tr", "group_col": "TR", "target_col": "IN", "agg": "std"}}}}

**aggregate_hooks**: Reduce multiple hooks to one value
  - params: hooks* (list of hook IDs), agg* (min|max|mean|sum|range)
  - Use for: "which of these 6 treatment means is highest?"
  - Example: {{"tool": "aggregate_hooks", "params": {{"hooks": ["cv_1", "cv_2", "cv_3"], "agg": "max"}}}}

All tools are atomic, deterministic, and verifiable.
""".strip()


def build_prompt(dataset_description: str, bootstrap_output: str) -> str:
    """Build system prompt with dataset context and initial exploration results."""
    tool_docs = format_tool_docs()
    
    return f"""You are a senior data scientist designing a **final exam** for an advanced data analysis course. Your questions must require multi-step reasoning chains—not simple lookups.

DATASET CONTEXT:
{dataset_description}

INITIAL EXPLORATION:
```
{bootstrap_output}
```

{tool_docs}

HOW TO EXPLORE:
Call tools by placing JSON in <code>...</code> tags. One tool call per block.
<code>{{"tool": "inspect", "aspect": "head|tail|shape|dtypes|columns|missing", "n": 10}}</code>
<code>{{"tool": "describe", "include": "number|object|all"}}</code>
<code>{{"tool": "value_counts", "col": "TR", "top_n": 20}}</code>
<code>{{"tool": "unique", "col": "TREE", "top_n": 50}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean|sum|median|std|min|max|count", "filter_expr": "IN > 5"}}</code>
<code>{{"tool": "multi_group_stat", "group_col": "TR", "target_cols": ["TL", "IN"], "aggs": ["mean", "std"]}}</code>
<code>{{"tool": "filter_stat", "target_col": "TL", "agg": "mean|median|std|min|max|count", "filter_expr": "TR == 'control' and IN > 5"}}</code>
<code>{{"tool": "correlation", "col_a": "TL", "col_b": "IN", "method": "pearson|spearman|kendall", "filter_expr": ""}}</code>
<code>{{"tool": "count_filter", "filter_expr": "TL > 50 and TR == 'control'"}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "agg": "condition_rate", "condition": "TL > 50", "group_val": "control"}}</code>
<code>{{"tool": "sort_values", "col": "TL", "ascending": false, "top_n": 10}}</code>
<code>{{"tool": "quantile", "col": "TL", "q": [0.1, 0.25, 0.5, 0.75, 0.9]}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "INTERNODE_5", "agg": "missing_rate", "group_val": "control"}}</code>
<code>{{"tool": "group_value_counts", "group_col": "TR", "target_col": "IN", "top_n": 5}}</code>
<code>{{"tool": "crosstab", "col_a": "TR", "col_b": "TREE", "normalize": "index|columns|all|"}}</code>

YOUR TASK:
1. **Explore** the dataset deeply—find patterns, anomalies, relationships, and edge cases
2. **Brainstorm broadly**—propose MANY candidate questions (aim for 15-20+ ideas), then refine
3. **Select the best 10** that require genuine multi-step reasoning chains
4. **Solve** each by defining 3-10 hooks that build on each other

═══════════════════════════════════════════════════════════════════════════════
WHAT MAKES A GOOD EXAM QUESTION (CRITICAL)
═══════════════════════════════════════════════════════════════════════════════

BAD (lookup questions—AVOID THESE):
- "What is the mean TL for control?" (1 step: just groupby + mean)
- "How many rows have IN > 10?" (1 step: just filter + count)
- "What is the correlation between TL and IN?" (1 step: just df.corr())

GOOD (multi-step reasoning chains—THIS IS WHAT WE WANT):
- "Identify which treatment has the highest proportion of 'long' branches (TL > 50cm), then compute how much higher the mean INTERNODE_5 is in those long branches compared to short branches within that treatment, expressed as a percentage."
  → Step 1: For each treatment, count branches with TL > 50
  → Step 2: For each treatment, count total branches
  → Step 3: Compute proportions, find max
  → Step 4: Filter to winning treatment + long branches, get mean INTERNODE_5
  → Step 5: Filter to winning treatment + short branches, get mean INTERNODE_5
  → Step 6: Compute percentage difference
  
- "A researcher hypothesizes a 'threshold effect': branches with >10 internodes show different early vs late growth patterns. Test this by computing the ratio of mean(INTERNODE_1:5) to mean(INTERNODE_6:10) for branches with IN>10 vs IN<=10 in the control group. What is the difference in these ratios?"
  → Multiple filters, multiple aggregations, ratio computations, comparison

- "Which tree (TREE) shows the most consistent response to PP_333_20g/L treatment, measured by having the lowest coefficient of variation (std/mean) in TL? Report the tree ID and its CV."
  → Group by tree within treatment, compute both std and mean, derive CV, find min

DIFFICULTY LEVELS (based on reasoning chain length):
- MEDIUM (3-4 hooks): Multi-step but linear. Filter→aggregate→compare. Example: "Compare mean TL between the two most common IN values"
- HARD (4-6 hooks): Branching logic or derived metrics. Example: "Find treatment with highest TL variance, then compare its correlation TL~IN to the treatment with lowest variance"
- VERY_HARD (5-10 hooks): Complex chains with multiple derivations. Example: "Identify outlier branches (TL > 2 std above treatment mean) in each treatment, compute what % of their internode measurements are missing ('?'), compare this % between treatments"

{HOOK_TOOLS_DOC}

EPISODE SCHEMA:
```json
{EPISODE_SCHEMA}
```

═══════════════════════════════════════════════════════════════════════════════
WORKED EXAMPLE: VERY_HARD EPISODE (10 hooks with proper chaining)
═══════════════════════════════════════════════════════════════════════════════

Question: "Compare growth consistency (coefficient of variation = std/mean of TL) between the control group and the treatment with the highest mean TL. Which has lower CV (more consistent growth), and by what percentage is it lower?"

```json
{{
  "question_text": "Compare growth consistency (CV = std/mean of TL) between control and the treatment with highest mean TL. Which has lower CV (more consistent), and by what percentage is it lower?",
  "difficulty": "VERY_HARD",
  "hooks": [
    {{"id": "ctrl_std", "tool": "group_stat", "params": {{"group_col": "TR", "target_col": "TL", "agg": "std", "group_val": "control"}}, "depends_on": []}},
    {{"id": "ctrl_mean", "tool": "group_stat", "params": {{"group_col": "TR", "target_col": "TL", "agg": "mean", "group_val": "control"}}, "depends_on": []}},
    {{"id": "ctrl_cv", "tool": "combine", "params": {{"expr": "s / m", "vars": {{"s": "ctrl_std", "m": "ctrl_mean"}}}}, "depends_on": ["ctrl_std", "ctrl_mean"]}},
    {{"id": "best_tr", "tool": "group_extremum", "params": {{"group_col": "TR", "target_col": "TL", "agg": "mean", "extremum": "max", "return_what": "group"}}, "depends_on": []}},
    {{"id": "best_std", "tool": "lookup", "params": {{"group_hook": "best_tr", "group_col": "TR", "target_col": "TL", "agg": "std"}}, "depends_on": ["best_tr"]}},
    {{"id": "best_mean", "tool": "lookup", "params": {{"group_hook": "best_tr", "group_col": "TR", "target_col": "TL", "agg": "mean"}}, "depends_on": ["best_tr"]}},
    {{"id": "best_cv", "tool": "combine", "params": {{"expr": "s / m", "vars": {{"s": "best_std", "m": "best_mean"}}}}, "depends_on": ["best_std", "best_mean"]}},
    {{"id": "lower_cv", "tool": "aggregate_hooks", "params": {{"hooks": ["ctrl_cv", "best_cv"], "agg": "min"}}, "depends_on": ["ctrl_cv", "best_cv"]}},
    {{"id": "cv_diff", "tool": "combine", "params": {{"expr": "a - b", "vars": {{"a": "ctrl_cv", "b": "best_cv"}}}}, "depends_on": ["ctrl_cv", "best_cv"]}},
    {{"id": "pct_lower", "tool": "combine", "params": {{"expr": "(high - low) / high * 100", "vars": {{"high": "ctrl_cv", "low": "best_cv"}}}}, "depends_on": ["ctrl_cv", "best_cv"]}}
  ],
  "teacher_answers": {{
    "ctrl_std": "18.4500 (n=45)",
    "ctrl_mean": "45.2300 (n=45)",
    "ctrl_cv": "0.4079",
    "best_tr": "control (mean=45.2300, n=45)",
    "best_std": "18.4500 (n=45)",
    "best_mean": "45.2300 (n=45)",
    "best_cv": "0.4079",
    "lower_cv": "0.4079 (from ctrl_cv)",
    "cv_diff": "0.0000",
    "pct_lower": "0.0000"
  }},
  "solution_trace": "1-2) Get std and mean for control. 3) Compute control CV = std/mean. 4) Find treatment with highest mean TL. 5-6) Look up std and mean for that treatment. 7) Compute its CV. 8) Find which CV is lower using aggregate_hooks. 9-10) Compute difference and percentage. Note: In this case control IS the best treatment, so CVs are equal."
}}
```

DAG structure (hooks build on each other):
```
ctrl_std ──┬──→ ctrl_cv ──┬──→ lower_cv
ctrl_mean ─┘              ├──→ cv_diff
                          └──→ pct_lower
best_tr ──┬──→ best_std ──┬──→ best_cv ──┘
          └──→ best_mean ─┘
```

Key patterns demonstrated:
- **combine**: Arithmetic + boolean on hook results (CV = std/mean, comparisons, percentages)
- **lookup**: Dynamically query stats for the group returned by group_extremum
- **aggregate_hooks**: Find min/max across multiple hooks (reports which hook won)
- Proper depends_on forming a real DAG, not parallel independent lookups

REQUIREMENTS FOR GOOD QUESTIONS:
- MEDIUM questions need 3-4 hooks minimum
- HARD questions need 4-6 hooks minimum  
- VERY_HARD questions need 5-10 hooks minimum
- Hooks MUST build on each other via depends_on—form a real DAG, not parallel independent lookups
- You MUST compute and provide teacher_answers for EVERY hook
- Use hook-chaining tools (combine, lookup, aggregate_hooks) to build on prior results
- Answers must be specific (exact numbers, not ranges or approximations)
- Questions should sound like exam questions, not "what is the value of X"

TURN STRUCTURE (during exploration):
1. **Interpretation**: What patterns or anomalies do you see? (1-2 sentences)
2. **Brainstorm**: List 5-10 potential question ideas—be creative, think about what would challenge a student
3. **Refine**: Star (★) the most promising ideas that require real reasoning chains
4. **Next exploration**: <code> blocks to dig deeper into promising directions

FINAL OUTPUT (when ready):
When you have explored enough, write DONE on its own line, then output your final 10 episodes as a JSON array:

```json
[
  {{episode 1}},
  {{episode 2}},
  ...
  {{episode 10}}
]
```

Each episode must include question_text, difficulty, hooks (3-10 based on difficulty), teacher_answers (one per hook), and solution_trace.

Begin by sharing your interpretation, brainstorm broadly, then explore. Signal DONE when ready to output final episodes."""


def build_tool_feedback_prompt(dataset_description: str, bootstrap_output: str) -> str:
    """Build prompt for tool feedback mode - identify missing/friction tools."""
    tool_docs = format_tool_docs()
    
    return f"""You are a senior data scientist evaluating a tool library for data analysis. Your goal is to identify gaps and friction points.

DATASET CONTEXT:
{dataset_description}

INITIAL EXPLORATION:
```
{bootstrap_output}
```

{tool_docs}

HOW TO EXPLORE:
Call tools by placing JSON in <code>...</code> tags. One tool call per block.

YOUR TASK:
1. **Explore** the dataset using the available tools
2. **Note friction**: When you wish you had a different tool, or find the current tools awkward, mark it with <TOOL_WISH> tags:
   <TOOL_WISH>I wish I could compute rolling averages across internodes</TOOL_WISH>
3. **Document patterns**: What operations do you find yourself repeating? What would make analysis faster?

After exploring, write DONE and output a JSON array of tool recommendations.

CRITICAL: Output VALID JSON only. Do NOT use placeholders like `{{...}}` or `...` - use real example values.

```json
[
  {{
    "name": "suggested_tool_name",
    "priority": "high|medium|low",
    "why": "explanation of the gap this fills",
    "example_call": {{"tool": "example_tool", "params": {{"col": "TL", "group_col": "TR"}}}},
    "returns": "description of expected output"
  }}
]
```

Focus on tools that would:
- Reduce multi-step workflows to single calls
- Handle common data patterns more naturally
- Provide atomic, verifiable outputs (scalar or structured)

Begin exploring and note any friction you encounter."""


# Dataset description (manual for now, could come from Kaggle API later)
DEFAULT_DATASET_DESCRIPTION = """
Tree branch growth measurements from an agricultural experiment.
- TR: Treatment (control, methanol_control, PP_333_4g/L, PP_333_20g/L, EL_500_4g/L, EL_500_20g/L)
- TREE: Tree identifier (e.g., G28, M33)
- BR: Branch label (A-J)
- TL: Total branch length (cm)
- IN: Number of internodes
- INTERNODE_1 to INTERNODE_29: Length of each internode (cm), '?' = missing

Goal: Understand how treatments affect branch growth patterns.
""".strip()
