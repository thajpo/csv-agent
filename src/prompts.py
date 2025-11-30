"""Prompts and templates used throughout the data pipeline."""

from src.tools import format_tool_docs

# Initial exploration code - runs automatically before Turn 1 so the model sees real data
BOOTSTRAP_CODE = """
import pandas as pd
import numpy as np

# Load and get first impressions
df = pd.read_csv("data.csv")
print("=== SHAPE ===")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")

print("\\n=== FIRST 5 ROWS ===")
print(df.head().to_string())

print("\\n=== DATA TYPES ===")
print(df.dtypes.to_string())

print("\\n=== NUMERIC SUMMARY ===")
print(df.describe().to_string())

print("\\n=== CATEGORICAL COLUMNS ===")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].nunique()} unique → {df[col].value_counts().head(3).to_dict()}")

print("\\n=== MISSING VALUES ===")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0].to_string())
else:
    print("No null values (but check for placeholder strings like '?' or 'NA')")
""".strip()


# JSON schema for episodes - shown to the model
EPISODE_SCHEMA = """
{
  "question_text": "string - a multi-step exam question requiring reasoning (NOT a simple lookup)",
  "difficulty": "MEDIUM | HARD | VERY_HARD",
  "hooks": [
    {
      "id": "string - unique identifier like 'h1', 'filter_control', 'compute_ratio', etc.",
      "tool": "group_stat | correlation | count_filter | model_eval | python_code",
      "params": {
        "// For group_stat": "filter_expr?, target_col, group_col, group_val, agg (mean|median|sum|count|std)",
        "// For correlation": "filter_expr?, col_a, col_b, method? (pearson|spearman)",
        "// For count_filter": "filter_expr",
        "// For model_eval": "filter_expr?, target_col, feature_cols[], model, metric, seed",
        "// For python_code": "code (string), depends_on (list of hook ids) - USE THIS TO CHAIN RESULTS"
      },
      "depends_on": ["list of hook ids this depends on - BUILD A REAL DAG"]
    }
  ],
  "// hooks requirement": "MEDIUM: 3-4 hooks, HARD: 4-6 hooks, VERY_HARD: 5-10 hooks",
  "teacher_answers": {
    "hook_id": "value - your computed answer for this hook (number, string, bool, list)"
  },
  "solution_trace": "string - step-by-step explanation of the reasoning chain"
}
""".strip()


# Hook tool documentation for the final output
HOOK_TOOLS_DOC = """
HOOK TOOLS (for building multi-step reasoning chains):

**group_stat**: Compute aggregated statistic for a group
  - params: filter_expr (optional), target_col, group_col, group_val, agg (mean|median|sum|count|std)
  - returns: {stat: number, n: count}
  - Example: {"tool": "group_stat", "params": {"filter_expr": "TR == 'control'", "target_col": "TL", "group_col": "TREE", "group_val": "G28", "agg": "mean"}}

**correlation**: Pearson/Spearman correlation between two columns
  - params: filter_expr (optional), col_a, col_b, method (pearson|spearman, default: pearson)
  - returns: {r: correlation, p: p-value, n: count}
  - Example: {"tool": "correlation", "params": {"col_a": "TL", "col_b": "IN", "method": "pearson"}}

**count_filter**: Count rows matching a filter expression
  - params: filter_expr
  - returns: {count: number}
  - Example: {"tool": "count_filter", "params": {"filter_expr": "TL > 100 and TR == 'control'"}}

**model_eval**: Train/test a model with fixed seed
  - params: filter_expr (optional), target_col, feature_cols[], model (linear_regression), metric (mse|mae|r2|accuracy), seed
  - returns: {metric: value, n_train: count, n_test: count}
  - Example: {"tool": "model_eval", "params": {"target_col": "TL", "feature_cols": ["IN"], "model": "linear_regression", "metric": "r2", "seed": 42}}

**python_code**: THE KEY TOOL FOR CHAINING - Compose over prior hook results
  - params: code (string), depends_on (list of hook ids)
  - The code receives a `results` dict with {{hook_id: result_dict}} from depends_on
  - Must return a single value (number, string, bool, list, dict)
  - USE THIS TO: compare values, compute ratios, find max/min, derive metrics
  - Examples:
    - Difference: {{"code": "results['h1']['stat'] - results['h2']['stat']", "depends_on": ["h1", "h2"]}}
    - Ratio: {{"code": "results['h1']['stat'] / results['h2']['stat']", "depends_on": ["h1", "h2"]}}
    - Percentage: {{"code": "100 * results['h1']['count'] / results['h2']['count']", "depends_on": ["h1", "h2"]}}
    - Comparison: {{"code": "'treatment' if results['h1']['stat'] > results['h2']['stat'] else 'control'", "depends_on": ["h1", "h2"]}}
    - Max of multiple: {{"code": "max([results['h1']['stat'], results['h2']['stat'], results['h3']['stat']])", "depends_on": ["h1", "h2", "h3"]}}
    - CV (std/mean): {{"code": "results['std_hook']['stat'] / results['mean_hook']['stat']", "depends_on": ["std_hook", "mean_hook"]}}
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
Call exploration tools by placing JSON in <code>...</code> tags. One tool call per block.
Examples:
<code>{{"tool": "inspect", "aspect": "head", "n": 10}}</code>
<code>{{"tool": "value_counts", "col": "TR"}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean"}}</code>
<code>{{"tool": "correlation", "col_a": "TL", "col_b": "IN"}}</code>

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

REQUIREMENTS FOR GOOD QUESTIONS:
- MEDIUM questions need 3-4 hooks minimum
- HARD questions need 4-6 hooks minimum  
- VERY_HARD questions need 5-10 hooks minimum
- Hooks MUST build on each other via depends_on—form a real DAG, not parallel independent lookups
- You MUST compute and provide teacher_answers for EVERY hook
- Use python_code hooks liberally to combine, compare, and derive from prior results
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
