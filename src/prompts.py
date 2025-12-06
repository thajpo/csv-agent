"""Prompts and templates used throughout the data pipeline."""

import pandas as pd
from src.tools import format_tool_docs, inspect, describe, value_counts
from dataclasses import dataclass
from typing import Callable

@dataclass
class ModeConfig:
    """Mode-specific configuration for the pipeline."""
    system_prompt: str
    extractor: Callable[[str], list[dict] | None]
    success_label: str
    parse_error_msg: str
    continue_msg: str
    final_msg: str


def get_mode_config(
    mode: str,
    dataset_description: str,
    bootstrap_output: str,
    target_questions: int,
) -> ModeConfig:
    """Build mode-specific config. Call this once at pipeline start."""
    
    if mode == "explore":
        return ModeConfig(
            system_prompt=build_question_generation_prompt(dataset_description, bootstrap_output, target_questions),
            extractor=extract_question_plans,
            success_label="question plans",
            parse_error_msg="[red]✗ Failed to parse question plans[/red]",
            continue_msg="\n\nContinue exploring the dataset. When done, write DONE and output your question plans as JSON.",
            final_msg="You've reached the turn limit. Please output your question plans now as a JSON array. Write DONE then the JSON.",
        )
    
    elif mode == "episodes":
        return ModeConfig(
            system_prompt=build_prompt(dataset_description, bootstrap_output),
            extractor=extract_json_episodes,
            success_label="episodes",
            parse_error_msg="[red]✗ Failed to parse episodes[/red]",
            continue_msg="\n\nAbove are the actual tool results from your calls. Use these real values to inform your next exploration steps. Continue exploring the dataset - make 3-8 parallel tool calls per turn to explore broadly (different treatments, columns, relationships simultaneously). Observe patterns, brainstorm questions. Do NOT output episodes yet - you need multiple turns of exploration first. When you have thoroughly explored (typically 5-8 turns), then write DONE and output your final 10 episodes as JSON.",
            final_msg="You've reached the turn limit. Please output your final 10 episodes now as a JSON array. Write DONE then the JSON.",
        )
    
    elif mode == "tool-feedback":
        return ModeConfig(
            system_prompt=build_tool_feedback_prompt(dataset_description, bootstrap_output),
            extractor=extract_json_array,
            success_label="tool recommendations",
            parse_error_msg="[red]✗ Failed to parse tool recommendations (check for invalid JSON like {...} placeholders)[/red]",
            continue_msg="\n\nContinue exploring. Note any tool friction with <TOOL_WISH>...</TOOL_WISH> tags. When done, write DONE and output your tool recommendations as JSON.",
            final_msg="You've reached the turn limit. Please output your tool recommendations now as a JSON array. Write DONE then the JSON.",
        )
    
    else:
        valid_modes = "explore, episodes, tool-feedback"
        raise ValueError(f"Unknown mode '{mode}' (expected one of: {valid_modes})")

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

QUESTION_PLAN_SCHEMA = """
{
  "question_text": "string - multi-step exam question requiring reasoning (NOT a simple lookup)",
  "difficulty": "MEDIUM | HARD | VERY_HARD",
  "reasoning_path": "string - numbered steps describing how to solve using available tools; no code or hooks",
  "key_columns": ["string", "string"],
  "expected_steps": 4
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

HOW TO EXPLORE (CRITICAL - READ CAREFULLY):
This is an **iterative process**. You will have multiple turns. In each turn:
1. **Call multiple tools in parallel** - Use multiple <code>...</code> blocks to explore different aspects simultaneously
2. **WAIT** for all tool results to be returned to you
3. Use those actual results to inform your next exploration steps
4. **DO NOT** predict what tools will return - you will see the real outputs

**Efficient exploration strategy**: Make 3-8 tool calls per turn to explore broadly. For example:
- Call tools for different treatments in parallel
- Query multiple columns simultaneously
- Explore different relationships at once
- Get overview stats and detailed breakdowns together

Call tools by placing JSON in <code>...</code> tags. You can have multiple <code> blocks in one turn - they will all execute and return results together.
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

**Example of parallel exploration** (make multiple calls in one turn):
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean"}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "std"}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "IN", "agg": "mean"}}</code>
<code>{{"tool": "correlation", "col_a": "TL", "col_b": "IN", "filter_expr": "TR == 'control'"}}</code>
<code>{{"tool": "correlation", "col_a": "TL", "col_b": "IN", "filter_expr": "TR == 'PP_333_20g/L'"}}</code>

All 5 tools above execute in parallel, giving you a comprehensive view in one turn.

YOUR TASK (MULTI-TURN PROCESS):
**Turn 1-3**: Explore the dataset deeply—call tools, observe results, find patterns
**Turn 4-6**: Brainstorm candidate questions based on what you've actually discovered
**Turn 7-9**: Refine questions, test feasibility with tools, verify your hook chains work
**Final turn**: Only after multiple rounds of exploration, output your final 10 episodes

**DO NOT** output episodes in your first turn. You must explore first and see actual tool results.

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

TURN STRUCTURE (during exploration - THIS IS ITERATIVE):
Each turn follows this pattern:
1. **Call multiple tools in parallel**: Use 3-8 <code> blocks to explore different aspects simultaneously
   - Example: Query stats for all treatments at once, check correlations for multiple pairs, get value counts for several columns
   - This allows efficient broad exploration rather than one narrow query at a time
2. **Wait for results**: The system will execute all your tools and return actual outputs
3. **Interpret results**: What do the numbers tell you? What patterns emerge across the parallel queries?
4. **Plan next steps**: Based on what you learned, what should you explore next in parallel?

**IMPORTANT**: You will receive tool results between turns. Use those actual values to inform your exploration. Do NOT make up or predict tool outputs.

**Early turns (1-3)**: Make parallel queries to understand data structure, distributions, relationships across multiple dimensions
**Middle turns (4-6)**: Start brainstorming questions based on patterns you've discovered, test multiple hypotheses in parallel
**Later turns (7-9)**: Refine questions, test hook chains, verify feasibility with parallel tool calls
**Final turn**: Only after thorough exploration, output episodes

FINAL OUTPUT (only after multiple exploration turns):
When you have explored enough (typically 5-8 turns), write DONE on its own line, then output your final 10 episodes as a JSON array:

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


def build_question_generation_prompt(
    dataset_description: str,
    bootstrap_output: str,
    target_questions: int = 10,
) -> str:
    """Build prompt for exploration-only phase that outputs question plans + reasoning paths."""
    tool_docs = format_tool_docs()
    
    return f"""You are a senior data scientist designing exam questions for a two-phase pipeline.

This is PHASE 1: You propose question blueprints only. No hooks, no computed answers.
Each blueprint pairs a question with a reasoning path—the step-by-step strategy to solve it.

DATASET CONTEXT:
{dataset_description}

INITIAL EXPLORATION:
```
{bootstrap_output}
```

{tool_docs}

═══════════════════════════════════════════════════════════════════════════════
EXPLORATION RULES (READ CAREFULLY)
═══════════════════════════════════════════════════════════════════════════════

You MUST explore for at least 3 turns before outputting questions.
DONE emitted before turn 3 will be REJECTED.

Each turn has TWO parts:

PART 1 - REASONING (required):
Briefly explain what you want to learn and why. What patterns are you looking for?
What hypothesis are you testing? This grounds your exploration.

PART 2 - TOOL CALLS:
Call 3-8 tools that address your stated goal.
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean"}}</code>

Then STOP. Your turn ends after tool calls.

Example turn:
```
I want to understand how treatments affect branch length variability, not just means.
High variability might indicate inconsistent treatment response.

<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean"}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "std"}}</code>
```

Example for computing CV (coefficient of variation):
```
I need to compute CV = std/mean for each treatment to measure growth consistency.

<code>{{"tool": "multi_group_stat", "group_col": "TR", "target_cols": ["TL"], "aggs": ["mean", "std"]}}</code>
```
(Then compute CV = std/mean from the returned table in your next turn)

CRITICAL: After emitting <code> blocks, STOP IMMEDIATELY.
Do not interpret results you haven't seen. Do not output DONE in the same turn as tool calls.

TOOL SYNTAX NOTES:
- filter_expr uses Python boolean syntax, NOT SQL:
  ✓ CORRECT: "TL >= 5 and TL <= 20" or "IN >= 10" or "TR == 'control' and TL > 50"
  ✗ WRONG: "TL BETWEEN 5 AND 20" (SQL syntax - will fail)
- Use Python operators: and, or, ==, !=, >=, <=, >, <
- String comparisons: TR == 'control' (use single quotes around strings)
- derive_stat formula uses actual column names (e.g., "TL / IN"), not aggregate names like "std" or "mean"

COMPUTING DERIVED STATISTICS:
- Coefficient of Variation (CV = std/mean): Use multi_group_stat to get both mean and std, then compute CV manually from the results:
  ✓ CORRECT: multi_group_stat with aggs=["mean", "std"], then CV = std/mean from the output
  ✗ WRONG: derive_stat with formula="std / mean" (std and mean are not column names)
- Other ratios (e.g., TL/IN): Use derive_stat with column names: derive_stat(formula="TL / IN", ...)

ERROR HANDLING:
If a tool call fails, acknowledge the error in your next turn's PART 1 reasoning.
Explain what went wrong and how you're adjusting your approach.
Do not repeat the same failed syntax.

═══════════════════════════════════════════════════════════════════════════════
WHAT MAKES A GOOD QUESTION
═══════════════════════════════════════════════════════════════════════════════

Each question must have ONE CONCRETE, VERIFIABLE ANSWER.
The answer should be a specific value: a number, a treatment name, a percentage, or a single comparison result.
Avoid "and" clauses that require separate answers—each question should produce ONE result.

COMPARISON METRICS (be explicit):
When comparing values, you MUST specify the metric explicitly:
- ✓ "What is the RATIO of X to Y?"
- ✓ "What is the PERCENTAGE DIFFERENCE between X and Y?"
- ✓ "What is the ABSOLUTE DIFFERENCE between X and Y?"
- ✗ "What is X compared to Y?" (too vague - doesn't specify ratio/difference/percentage)
- ✗ "How does X compare to Y?" (too vague)

If your question asks for a comparison, state the exact metric in the question text.

BAD (vague, multi-part, no single answer):
- "Investigate the relationship between TL and IN across treatments and discuss how correlation varies."
- "Analyze how treatment effects differ between early and late internode development."
- "What is the 90th percentile of TL across all treatments, AND how many treatments have max IN ≥ this percentile?"
  (This asks for TWO separate answers: a percentile value AND a count - split into two questions)
- "What is the ratio of mean TL to mean IN for methanol_control compared to PP_333_20g/L, AND which treatment has higher average internode length?"
  (This asks for two separate answers: a ratio AND a comparison - split into two questions)
- "For branches with TL > 20 cm, what is the mean INTERNODE_1 length compared to branches with TL < 5 cm?"
  (Vague: "compared to" doesn't specify the metric - use "ratio", "difference", or "percentage difference")

GOOD (specific, produces ONE concrete answer):
- "Which treatment has the lowest coefficient of variation (std/mean) for TL, and what is that CV value?"
  → Answer: "PP_333_4g/L, CV = 0.32" (one treatment + one number)
- "For branches with IN ≥ 10, which treatment shows the largest ratio of mean INTERNODE_5 to mean INTERNODE_1?"
  → Answer: "EL_500_20g/L, ratio = 1.45" (one treatment + one ratio)
- "What is the percentage difference in mean TL between control and PP_333_20g/L?"
  → Answer: "Control is 289% higher than PP_333_20g/L" (one comparison result)
- "What is the ratio of mean INTERNODE_1 for branches with TL > 20 cm to mean INTERNODE_1 for branches with TL < 5 cm?"
  → Answer: "Ratio = 2.01" (explicit metric: "ratio")
- "What is the absolute difference in mean INTERNODE_1 between branches with TL > 20 cm and branches with TL < 5 cm?"
  → Answer: "Difference = 0.38 cm" (explicit metric: "absolute difference")

DIFFICULTY (MUST match expected_steps):
- MEDIUM: 3-4 steps. Filter → aggregate → compare.
- HARD: 4-6 steps. Multiple aggregations, derived metrics, or conditional logic.
- VERY_HARD: 6-10 steps. Complex chains with ratios, nested comparisons, or multi-group analysis.

CRITICAL: difficulty and expected_steps MUST align:
- If expected_steps = 3-4 → difficulty = MEDIUM
- If expected_steps = 4-6 → difficulty = HARD
- If expected_steps = 6-10 → difficulty = VERY_HARD

WRONG: difficulty = VERY_HARD but expected_steps = 3 (should be MEDIUM)
WRONG: difficulty = HARD but expected_steps = 3 (should be MEDIUM)

═══════════════════════════════════════════════════════════════════════════════
REASONING PATH FORMAT
═══════════════════════════════════════════════════════════════════════════════

The reasoning_path describes HOW to solve the question in numbered steps.
Use natural language. Do NOT include tool names or code.

Example:
```
"reasoning_path": "1) For each treatment, compute mean and std of TL. 2) Derive CV = std/mean for each treatment. 3) Identify the treatment with minimum CV. 4) Report that treatment and its CV value."
```

Each step should be:
- Concrete (what computation, what columns, what filter)
- Sequential (later steps build on earlier ones)
- Verifiable (produces an intermediate or final value)

═══════════════════════════════════════════════════════════════════════════════
OUTPUT SCHEMA
═══════════════════════════════════════════════════════════════════════════════

Output a JSON array with {target_questions} objects:
```json
{QUESTION_PLAN_SCHEMA}
```

WORKED EXAMPLE:
```json
{{
  "question_text": "Which treatment produces the most consistent branch lengths (lowest CV of TL), and by what percentage is its CV lower than control's CV?",
  "difficulty": "HARD",
  "reasoning_path": "1) For each treatment, compute mean TL and std TL. 2) Derive CV = std/mean for each. 3) Find treatment with minimum CV. 4) Compute control's CV. 5) Calculate percentage difference: (control_CV - min_CV) / control_CV * 100.",
  "key_columns": ["TR", "TL"],
  "expected_steps": 5
}}
```

═══════════════════════════════════════════════════════════════════════════════
TURN STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

Turn 1-2: Broad exploration. Map distributions, relationships, treatment effects.
          State what you're looking for. Call tools. Stop.
Turn 3-4: Interpret prior results. What patterns emerged? What's surprising?
          If any tool calls failed, acknowledge errors and adjust syntax.
          State follow-up questions. Call tools. Stop.
Turn 5+:  Synthesize findings. Draft question ideas based on discovered patterns.
          Verify feasibility with targeted tool calls. Stop.
Final:    Write DONE on its own line, then output the JSON array.
          (No tool calls in this turn.)

BEFORE OUTPUTTING QUESTIONS - VALIDATION CHECKLIST:
For each question, verify:
1. ✓ Single answer: Does it produce ONE result? (No "and" clauses asking for separate answers)
2. ✓ Explicit comparisons: If comparing, does it specify "ratio", "difference", or "percentage difference"?
3. ✓ Difficulty match: Does difficulty match expected_steps?
   - expected_steps 3-4 → MEDIUM
   - expected_steps 4-6 → HARD
   - expected_steps 6-10 → VERY_HARD
4. ✓ Concrete answer: Can the answer be a specific number, treatment name, or single comparison result?
5. ✓ Grounded: Does the question reflect patterns you actually discovered in the data?

Your questions will be evaluated on:
- Grounding: Do questions reflect patterns you actually found in the data?
- Specificity: Does each question have ONE concrete, verifiable answer?
- Reasoning clarity: Can someone follow the numbered steps to reproduce the answer?
- Format compliance: Single answer, explicit metrics, difficulty matches steps

Begin by stating what you want to learn about this dataset, then call tools.
Do NOT output DONE until you have completed at least 3 turns with tool results."""


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
