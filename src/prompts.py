"""Prompts and templates used throughout the data pipeline."""

from src.tools import format_tool_docs

# Initial exploration code - runs automatically before Turn 1 so the model sees real data
BOOTSTRAP_CODE = """
import pandas as pd
from src.tools import inspect, describe, value_counts

df = pd.read_csv("data.csv")

for label, call in [
    ("SHAPE", lambda: inspect(df, "shape")),
    ("HEAD", lambda: inspect(df, "head", 5)),
    ("DTYPES", lambda: inspect(df, "dtypes")),
    ("NUMERIC SUMMARY", lambda: describe(df, "number")),
    ("MISSING", lambda: inspect(df, "missing")),
]:
    print(f"=== {label} ===\\n{call()}\\n")

print("=== CATEGORICAL VALUE COUNTS ===")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\\n{col}:\\n{value_counts(df, col, 5)}")
""".strip()


# JSON schema for episodes - shown to the model
EPISODE_SCHEMA = """
{
  "question_text": "string - multi-step exam question requiring reasoning (NOT a simple lookup)",
  "difficulty": "MEDIUM | HARD | VERY_HARD",
  "hooks": [
    {
      "id": "string - unique, lowercase with underscores only (e.g. 'ctrl_mean_tl', 'compare_ratio')",
      "tool": "group_stat | correlation | count_filter | python_code",
      "params": {
        "// group_stat": "filter_expr?, target_col, group_col, group_val, agg (mean|median|sum|count|std|min|max|nunique)",
        "// correlation": "filter_expr?, col_a, col_b, method? (pearson|spearman)",
        "// count_filter": "filter_expr",
        "// python_code": "code, depends_on[] - MUST match hook's depends_on exactly"
      },
      "depends_on": ["hook ids this depends on - forms acyclic DAG"]
    }
  ],
  "teacher_answers": {"hook_id": "scalar value (number|string|bool) - see INVARIANTS"},
  "solution_trace": "step-by-step explanation"
}

INVARIANTS (your output MUST satisfy these):
- Every hook.id is unique and appears exactly once as a key in teacher_answers
- For group_stat: teacher_answers[id] = the 'stat' value (number)
- For correlation: teacher_answers[id] = the 'r' value (number)
- For count_filter: teacher_answers[id] = the 'count' value (number)
- For python_code: teacher_answers[id] = the returned scalar; params.depends_on MUST equal hook's depends_on
- All depends_on refs must point to earlier hooks (acyclic)
- Only use columns from INITIAL EXPLORATION - do not invent column names
- Minimum 2 hooks (must be multi-step), no maximum
""".strip()


# Hook tool documentation for the final output
HOOK_TOOLS_DOC = """
HOOK TOOLS (for episodes only - not exploration):

**group_stat** → returns {{stat, n}} → teacher_answers gets 'stat'
  params: filter_expr?, target_col, group_col, group_val, agg (mean|median|sum|count|std|min|max|nunique)

**correlation** → returns {{r, p, n}} → teacher_answers gets 'r'
  params: filter_expr?, col_a, col_b, method? (pearson|spearman)

**count_filter** → returns {{count}} → teacher_answers gets 'count'
  params: filter_expr

**python_code** → returns scalar → teacher_answers gets that scalar
  params: code, depends_on[] (MUST match hook's depends_on exactly)
  `results` dict has {{hook_id: full_result_dict}} from depends_on hooks
  Examples:
    {{"code": "results['h1']['stat'] - results['h2']['stat']", "depends_on": ["h1", "h2"]}}
    {{"code": "results['h1']['stat'] / results['h2']['stat']", "depends_on": ["h1", "h2"]}}
    {{"code": "'A' if results['h1']['stat'] > results['h2']['stat'] else 'B'", "depends_on": ["h1", "h2"]}}
    {{"code": "round(100 * results['h1']['count'] / results['h2']['count'], 2)", "depends_on": ["h1", "h2"]}}
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

EXPLORATION TOOLS (use these now via <code> blocks, one per block):
<code>{{"tool": "inspect", "aspect": "head|tail|shape|dtypes|columns|missing", "n": 10}}</code>
<code>{{"tool": "describe", "include": "number|object|all"}}</code>
<code>{{"tool": "value_counts", "col": "TR", "top_n": 20}}</code>
<code>{{"tool": "unique", "col": "TREE", "top_n": 50}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean"}}</code>  // all groups
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean", "group_val": "control"}}</code>  // ONE group (atomic)
<code>{{"tool": "group_extremum", "group_col": "TR", "target_col": "TL", "agg": "mean", "extremum": "max", "return_what": "group|value"}}</code>
<code>{{"tool": "derive_stat", "formula": "TL / IN", "group_col": "TR", "agg": "mean"}}</code>  // all groups
<code>{{"tool": "derive_stat", "formula": "TL / IN", "group_col": "TR", "agg": "mean", "group_val": "control"}}</code>  // ONE group (atomic)
<code>{{"tool": "correlation", "col_a": "TL", "col_b": "IN", "method": "pearson|spearman|kendall", "filter_expr": ""}}</code>
<code>{{"tool": "count_filter", "filter_expr": "TL > 50 and TR == 'control'"}}</code>
<code>{{"tool": "sort_values", "col": "TL", "ascending": false, "top_n": 10}}</code>
<code>{{"tool": "quantile", "col": "TL", "q": 0.9}}</code>  // single quantile (atomic)
<code>{{"tool": "crosstab", "col_a": "TR", "col_b": "TREE", "normalize": "index|columns|all|"}}</code>
Note: Add group_val or single q for atomic scalar output. '?' auto-coerced to NaN.

YOUR TASK:
1. **Explore** the dataset deeply—find patterns, anomalies, relationships, and edge cases
2. **Brainstorm broadly**—propose MANY candidate questions (aim for 15-20+ ideas), then refine
3. **Select the best 10** that require genuine multi-step reasoning chains
4. **Solve** each by defining hooks that build on each other (use as many as needed)

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

DIFFICULTY LEVELS (based on reasoning complexity):
- MEDIUM: Multi-step but linear chain. Filter→aggregate→compare.
- HARD: Branching logic, derived metrics, or conditional reasoning.
- VERY_HARD: Complex chains with multiple derivations, comparisons across groups, or iterative computations.

{HOOK_TOOLS_DOC}

EPISODE SCHEMA:
```json
{EPISODE_SCHEMA}
```

═══════════════════════════════════════════════════════════════════════════════
WORKED EXAMPLE: VERY_HARD EPISODE (7 hooks, proper DAG)
═══════════════════════════════════════════════════════════════════════════════

Question: "Which treatment shows the largest 'growth efficiency gap'? Define growth efficiency as TL/IN (total length per internode). Compare the mean efficiency of 'long' branches (IN >= 10) vs 'short' branches (IN < 5) within each treatment. Report which treatment has the largest gap and what that gap is."

```json
{{
  "question_text": "Which treatment shows the largest 'growth efficiency gap'? Define growth efficiency as TL/IN (total length per internode). Compare the mean efficiency of long branches (IN >= 10) vs short branches (IN < 5) within each treatment. Report which treatment has the largest gap and what that gap value is.",
  "difficulty": "VERY_HARD",
  "hooks": [
    {{"id": "ctrl_long_tl", "tool": "group_stat", "params": {{"filter_expr": "TR == 'control' and IN >= 10", "target_col": "TL", "group_col": "TR", "group_val": "control", "agg": "mean"}}, "depends_on": []}},
    {{"id": "ctrl_long_in", "tool": "group_stat", "params": {{"filter_expr": "TR == 'control' and IN >= 10", "target_col": "IN", "group_col": "TR", "group_val": "control", "agg": "mean"}}, "depends_on": []}},
    {{"id": "ctrl_short_tl", "tool": "group_stat", "params": {{"filter_expr": "TR == 'control' and IN < 5", "target_col": "TL", "group_col": "TR", "group_val": "control", "agg": "mean"}}, "depends_on": []}},
    {{"id": "ctrl_short_in", "tool": "group_stat", "params": {{"filter_expr": "TR == 'control' and IN < 5", "target_col": "IN", "group_col": "TR", "group_val": "control", "agg": "mean"}}, "depends_on": []}},
    {{"id": "ctrl_long_eff", "tool": "python_code", "params": {{"code": "results['ctrl_long_tl']['stat'] / results['ctrl_long_in']['stat']", "depends_on": ["ctrl_long_tl", "ctrl_long_in"]}}, "depends_on": ["ctrl_long_tl", "ctrl_long_in"]}},
    {{"id": "ctrl_short_eff", "tool": "python_code", "params": {{"code": "results['ctrl_short_tl']['stat'] / results['ctrl_short_in']['stat']", "depends_on": ["ctrl_short_tl", "ctrl_short_in"]}}, "depends_on": ["ctrl_short_tl", "ctrl_short_in"]}},
    {{"id": "ctrl_gap", "tool": "python_code", "params": {{"code": "round(results['ctrl_long_eff'] - results['ctrl_short_eff'], 3)", "depends_on": ["ctrl_long_eff", "ctrl_short_eff"]}}, "depends_on": ["ctrl_long_eff", "ctrl_short_eff"]}}
  ],
  "teacher_answers": {{
    "ctrl_long_tl": 61.64,
    "ctrl_long_in": 14.17,
    "ctrl_short_tl": 4.82,
    "ctrl_short_in": 2.73,
    "ctrl_long_eff": 4.35,
    "ctrl_short_eff": 1.77,
    "ctrl_gap": 2.58
  }},
  "solution_trace": "1) Get mean TL and IN for long branches (IN>=10) in control. 2) Get mean TL and IN for short branches (IN<5) in control. 3) Compute efficiency (TL/IN) for each group. 4) Compute gap = long_eff - short_eff. Note: Full question would repeat for all treatments and find max gap."
}}
```

Notice the DAG structure:
- h1,h2 → h5 (long efficiency depends on long TL and IN)
- h3,h4 → h6 (short efficiency depends on short TL and IN)  
- h5,h6 → h7 (gap depends on both efficiencies)

This is ONE treatment. A full VERY_HARD question would compute this for all 6 treatments (more hooks) and find the max. The key is that each hook builds on prior results—no hook stands alone as a simple lookup.

TURN STRUCTURE (during exploration):
Each turn, THINK OUT LOUD before running tools:

1. **What I see**: Describe what's interesting in the data or last result. Why does it matter?
2. **Question ideas**: Jot down 2-3 potential exam questions this suggests.
3. **Next step**: What do you want to explore next and why?
4. **Tool calls**: Then run <code> blocks.

Example turn:
"The control group has much higher variance in TL (std=15.2) than PP_333 treatments (std=4.1). 
This suggests the treatment stabilizes growth. Question ideas: (1) Which treatment reduces TL 
variance most? (2) Is low variance correlated with fewer internodes? Let me check the 
internode distribution across treatments..."

Do NOT just emit code—explain your reasoning first.

FINAL OUTPUT:
When ready, output exactly this:
1. A line containing only: DONE
2. A JSON array of exactly 10 episodes (no other text):
```json
[{{"question_text": "...", "difficulty": "...", "hooks": [...], "teacher_answers": {{...}}, "solution_trace": "..."}}, ...]
```

Begin exploring. Think out loud as you go. Signal DONE when ready."""


def build_tool_feedback_prompt(dataset_description: str, bootstrap_output: str) -> str:
    """Build prompt for tool improvement feedback session."""
    tool_docs = format_tool_docs()
    
    return f"""You are a senior data scientist evaluating a data exploration toolkit. Your goal is to explore this dataset thoroughly and identify **what tools are missing or could be improved**.

DATASET CONTEXT:
{dataset_description}

INITIAL EXPLORATION:
```
{bootstrap_output}
```

{tool_docs}

EXPLORATION TOOLS (use these now via <code> blocks, one per block):
<code>{{"tool": "inspect", "aspect": "head|tail|shape|dtypes|columns|missing", "n": 10}}</code>
<code>{{"tool": "describe", "include": "number|object|all"}}</code>
<code>{{"tool": "value_counts", "col": "TR", "top_n": 20}}</code>
<code>{{"tool": "unique", "col": "TREE", "top_n": 50}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean"}}</code>  // all groups
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean", "group_val": "control"}}</code>  // ONE group (atomic)
<code>{{"tool": "group_extremum", "group_col": "TR", "target_col": "TL", "agg": "mean", "extremum": "max", "return_what": "group|value"}}</code>
<code>{{"tool": "derive_stat", "formula": "TL / IN", "group_col": "TR", "agg": "mean"}}</code>  // all groups
<code>{{"tool": "derive_stat", "formula": "TL / IN", "group_col": "TR", "agg": "mean", "group_val": "control"}}</code>  // ONE group (atomic)
<code>{{"tool": "correlation", "col_a": "TL", "col_b": "IN", "method": "pearson|spearman|kendall", "filter_expr": ""}}</code>
<code>{{"tool": "count_filter", "filter_expr": "TL > 50 and TR == 'control'"}}</code>
<code>{{"tool": "sort_values", "col": "TL", "ascending": false, "top_n": 10}}</code>
<code>{{"tool": "quantile", "col": "TL", "q": 0.9}}</code>  // single quantile (atomic)
<code>{{"tool": "crosstab", "col_a": "TR", "col_b": "TREE", "normalize": "index|columns|all|"}}</code>
Note: Add group_val or single q for atomic scalar output. '?' auto-coerced to NaN.

YOUR TASK:
Explore this dataset as if preparing a detailed analysis. As you work, pay attention to:
1. **What questions you want to answer** but struggle to express with current tools
2. **What operations require multiple awkward steps** that could be one tool
3. **What information you wish the tools returned** (format, additional stats, etc.)
4. **What common patterns** you find yourself repeating

TURN STRUCTURE:
Each turn, do the following:

1. **Explore**: Run 2-5 tool calls to investigate something
2. **Reflect**: What did you learn? What did you want to do but couldn't?
3. **Tool Wish**: If you hit friction, describe the tool you wish existed:
   
   {{TOOL_WISH}}
   name: <tool_name>
   why: <what you were trying to do>
   params: <what parameters it would take>
   returns: <what it would return>
   example: <example call and result>
   {{/TOOL_WISH}}

Keep exploring until you've thoroughly examined the dataset (at least 5-6 turns). Focus on the kinds of multi-step analytical questions a data scientist would ask.

FINAL OUTPUT:
When done exploring, write DONE and then summarize your top tool recommendations as a JSON array:
```json
[
  {{
    "name": "tool_name",
    "priority": "HIGH|MEDIUM|LOW",
    "why": "what problem it solves",
    "params": {{"param": "description"}},
    "returns": "what it returns",
    "example_call": {{}},
    "example_result": {{}}
  }}
]
```

Begin exploring. Think out loud about what you're trying to accomplish and where the tools fall short."""


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
