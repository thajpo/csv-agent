"""
Prompts and templates for the data analysis pipeline.

Architecture:
- FRAGMENTS: Reusable prompt pieces (tool docs, schemas, guidance)
- COMPOSERS: Functions that assemble fragments for specific modes
- Teacher prompts: Full guidance + examples (API models with large context)
- Student prompts: Minimal contracts only (local models with limited context)
"""

import pandas as pd
from src.tools import format_tool_docs, inspect, describe, value_counts, TOOL_SPECS
from dataclasses import dataclass


# ============================================================================
# SCHEMAS - Output format definitions
# ============================================================================

EPISODE_SCHEMA = """
{
  "question_text": "string - multi-step exam question requiring reasoning",
  "difficulty": "MEDIUM | HARD | VERY_HARD",
  "hooks": [
    {
      "id": "string - unique identifier (e.g. 'ctrl_mean')",
      "tool": "group_stat|group_extremum|correlation|count_filter|quantile|derive_stat|combine|lookup|aggregate_hooks",
      "params": { ... },
      "depends_on": ["list of hook ids this depends on"]
    }
  ],
  "teacher_answers": { "hook_id": "computed value" },
  "solution_trace": "string - step-by-step explanation"
}
""".strip()

QUESTION_PLAN_SCHEMA = """
{
  "question_text": "string - multi-step exam question requiring reasoning",
  "difficulty": "MEDIUM | HARD | VERY_HARD",
  "reasoning_path": "string - numbered steps to solve (no code)",
  "key_columns": ["string"],
  "expected_steps": 4
}
""".strip()


# ============================================================================
# TOOL DOCUMENTATION - Generated from TOOL_SPECS
# ============================================================================

def format_hook_tools_doc() -> str:
    """
    Generate documentation for hook tools (scalar-producing tools for episode chains).
    
    Hook tools are:
    - Data query tools that return scalars (with group_val specified)
    - Chain tools that operate on previous hook results
    """
    # Tools that can be used as hooks (produce scalar outputs for chaining)
    hook_tools = [
        "group_stat", "group_extremum", "correlation", "count_filter",
        "filter_stat", "quantile", "derive_stat", "group_condition_rate",
        "missing_rate",
        # Chain tools
        "combine", "lookup", "aggregate_hooks"
    ]
    
    lines = ["HOOK TOOLS (for building multi-step reasoning chains):", ""]
    
    # Data query tools
    lines.append("═══ DATA QUERY TOOLS (return scalars for chaining) ═══")
    lines.append("")
    for name in hook_tools:
        if name not in TOOL_SPECS or TOOL_SPECS[name].get("chain"):
            continue
        spec = TOOL_SPECS[name]
        desc = spec["description"].split(" | ")[0]  # Key description only
        lines.append(f"**{name}**: {desc}")
        
        # Format params
        params = spec.get("parameters", {}).get("properties", {})
        required = spec.get("parameters", {}).get("required", [])
        param_strs = []
        for pname, pinfo in params.items():
            req = "*" if pname in required else ""
            param_strs.append(f"{pname}{req}")
        if param_strs:
            lines.append(f"  params: {', '.join(param_strs)}")
        
        # Add examples if present
        examples = spec.get("examples", [])
        for ex in examples[:2]:  # Max 2 examples per tool
            # Format example nicely (remove outer quotes if it's a string)
            if isinstance(ex, str):
                lines.append(f"  Example: {ex}")
            else:
                import json
                lines.append(f"  Example: {json.dumps(ex, separators=(',', ':'))}")
        lines.append("")
    
    # Chain tools
    lines.append("═══ CHAIN TOOLS (operate on previous hook results) ═══")
    lines.append("")
    for name in ["combine", "lookup", "aggregate_hooks"]:
        if name not in TOOL_SPECS:
            continue
        spec = TOOL_SPECS[name]
        desc = spec["description"].split(" | ")[0]
        lines.append(f"**{name}**: {desc}")
        
        params = spec.get("parameters", {}).get("properties", {})
        required = spec.get("parameters", {}).get("required", [])
        param_strs = []
        for pname, pinfo in params.items():
            req = "*" if pname in required else ""
            param_strs.append(f"{pname}{req}")
        if param_strs:
            lines.append(f"  params: {', '.join(param_strs)}")
        
        examples = spec.get("examples", [])
        for ex in examples[:2]:
            if isinstance(ex, str):
                lines.append(f"  Example: {ex}")
            else:
                import json
                lines.append(f"  Example: {json.dumps(ex, separators=(',', ':'))}")
        lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# GUIDANCE FRAGMENTS - Reusable instruction pieces
# ============================================================================

TOOL_CALL_SYNTAX = """
Call tools by placing JSON in <code>...</code> tags:
<code>{{"tool": "tool_name", "param1": "value", ...}}</code>

Multiple <code> blocks execute in parallel.
""".strip()

EXPLORATION_RULES = """
HOW TO EXPLORE:
1. Call multiple tools in parallel using <code>...</code> blocks
2. WAIT for results before interpreting
3. Use actual results to inform next steps
4. Do NOT predict tool outputs

Make 3-8 tool calls per turn. After <code> blocks, STOP.
""".strip()

QUESTION_QUALITY_GUIDANCE = """
WHAT MAKES A GOOD QUESTION:

BAD (simple lookups):
- "What is the mean TL for control?" (1 step)
- "How many rows have IN > 10?" (1 step)

GOOD (multi-step reasoning chains):
- Questions requiring 3+ steps: filter → aggregate → compare → derive
- Questions that chain results: find best group → query its stats → compare

Each question must have ONE CONCRETE, VERIFIABLE ANSWER.
Avoid "and" clauses requiring separate answers.

DIFFICULTY (must match hook count):
- MEDIUM: 3-4 hooks
- HARD: 4-6 hooks  
- VERY_HARD: 5-10 hooks
""".strip()

DIFFICULTY_GUIDANCE = """
DIFFICULTY LEVELS:
- MEDIUM (3-4 hooks): Linear chains. Filter→aggregate→compare.
- HARD (4-6 hooks): Branching logic or derived metrics.
- VERY_HARD (5-10 hooks): Complex chains with multiple derivations.

Hooks MUST build on each other via depends_on—form a DAG, not parallel lookups.
""".strip()

TURN_STRUCTURE_EPISODES = """
TURN STRUCTURE:
- Turn 1-3: Explore dataset, observe patterns
- Turn 4-6: Brainstorm questions based on discoveries
- Turn 7-9: Refine questions, verify hook chains work
- Final: Write DONE, output episodes as JSON array
""".strip()

TURN_STRUCTURE_QUESTION_GEN = """
TURN STRUCTURE:
- Turn 1-2: Broad exploration (distributions, relationships)
- Turn 3-4: Interpret results, identify patterns
- Turn 5+: Draft questions, verify feasibility
- Final: Write DONE, output question plans as JSON array
""".strip()


# ============================================================================
# WORKED EXAMPLES - For teacher prompts
# ============================================================================

WORKED_EPISODE_EXAMPLE = """
WORKED EXAMPLE (VERY_HARD, 10 hooks):

Question: "Compare CV (std/mean of TL) between control and the treatment with highest mean TL. Which has lower CV, and by what percentage?"

```json
{
  "question_text": "Compare CV of TL between control and treatment with highest mean TL...",
  "difficulty": "VERY_HARD",
  "hooks": [
    {"id": "ctrl_std", "tool": "group_stat", "params": {"group_col": "TR", "target_col": "TL", "agg": "std", "group_val": "control"}, "depends_on": []},
    {"id": "ctrl_mean", "tool": "group_stat", "params": {"group_col": "TR", "target_col": "TL", "agg": "mean", "group_val": "control"}, "depends_on": []},
    {"id": "ctrl_cv", "tool": "combine", "params": {"expr": "s / m", "vars": {"s": "ctrl_std", "m": "ctrl_mean"}}, "depends_on": ["ctrl_std", "ctrl_mean"]},
    {"id": "best_tr", "tool": "group_extremum", "params": {"group_col": "TR", "target_col": "TL", "extremum": "max", "return_what": "group"}, "depends_on": []},
    {"id": "best_std", "tool": "lookup", "params": {"group_hook": "best_tr", "group_col": "TR", "target_col": "TL", "agg": "std"}, "depends_on": ["best_tr"]},
    {"id": "best_mean", "tool": "lookup", "params": {"group_hook": "best_tr", "group_col": "TR", "target_col": "TL", "agg": "mean"}, "depends_on": ["best_tr"]},
    {"id": "best_cv", "tool": "combine", "params": {"expr": "s / m", "vars": {"s": "best_std", "m": "best_mean"}}, "depends_on": ["best_std", "best_mean"]},
    {"id": "pct_lower", "tool": "combine", "params": {"expr": "(a - b) / a * 100", "vars": {"a": "ctrl_cv", "b": "best_cv"}}, "depends_on": ["ctrl_cv", "best_cv"]}
  ],
  "teacher_answers": {"ctrl_std": "18.45", "ctrl_mean": "45.23", "ctrl_cv": "0.408", ...},
  "solution_trace": "1-2) Get std/mean for control. 3) Compute CV. 4) Find best treatment. 5-6) Lookup its stats. 7) Compute its CV. 8) Percentage difference."
}
```

DAG structure:
ctrl_std ──┬──→ ctrl_cv ──┬──→ pct_lower
ctrl_mean ─┘              │
best_tr ──┬──→ best_std ──┬──→ best_cv ──┘
          └──→ best_mean ─┘
""".strip()

HOOK_CHAIN_PATTERNS = """
HOOK CHAIN PATTERNS:

**combine**: Arithmetic/boolean on hook results
  {{"tool": "combine", "params": {{"expr": "a / b", "vars": {{"a": "std_hook", "b": "mean_hook"}}}}}}
  {{"tool": "combine", "params": {{"expr": "a > b", "vars": {{"a": "cv1", "b": "cv2"}}}}}}

**lookup**: Query stats for a dynamically-found group
  {{"tool": "lookup", "params": {{"group_hook": "best_treatment", "group_col": "TR", "target_col": "TL", "agg": "std"}}}}

**aggregate_hooks**: Find min/max across computed values
  {{"tool": "aggregate_hooks", "params": {{"hooks": ["cv1", "cv2", "cv3"], "agg": "min"}}}}
""".strip()


# ============================================================================
# COMPOSER FUNCTIONS - Assemble prompts for specific modes
# ============================================================================

def build_episodes_prompt(
    dataset_description: str,
    data_overview: str,
    include_worked_example: bool = True,
) -> str:
    """
    Build system prompt for episode generation (teacher model).
    
    Args:
        dataset_description: Description of the dataset
        data_overview: Pre-computed data exploration output
        include_worked_example: Include the 50-line worked example (default True)
    """
    tool_docs = format_tool_docs(verbosity="compact")
    hook_docs = format_hook_tools_doc()
    
    sections = [
        "You are a senior data scientist designing multi-step exam questions.",
        "",
        f"DATASET:\n{dataset_description}",
        "",
        f"DATA OVERVIEW:\n```\n{data_overview}\n```",
        "",
        tool_docs,
        "",
        EXPLORATION_RULES,
        "",
        QUESTION_QUALITY_GUIDANCE,
        "",
        DIFFICULTY_GUIDANCE,
        "",
        hook_docs,
        "",
        f"EPISODE SCHEMA:\n```json\n{EPISODE_SCHEMA}\n```",
    ]
    
    if include_worked_example:
        sections.extend([
            "",
            WORKED_EPISODE_EXAMPLE,
            "",
            HOOK_CHAIN_PATTERNS,
        ])
    
    sections.extend([
        "",
        TURN_STRUCTURE_EPISODES,
        "",
        "Begin exploring. Write DONE when ready to output episodes.",
    ])
    
    return "\n".join(sections)


def build_question_generation_prompt(
    dataset_description: str,
    data_overview: str,
    target_questions: int = 10,
) -> str:
    """Build system prompt for question planning phase (teacher model)."""
    tool_docs = format_tool_docs(verbosity="full")
    
    sections = [
        "You are a senior data scientist designing exam questions.",
        "This is PHASE 1: Propose question blueprints (no hooks, no computed answers).",
        "",
        f"DATASET:\n{dataset_description}",
        "",
        f"DATA OVERVIEW:\n```\n{data_overview}\n```",
        "",
        tool_docs,
        "",
        EXPLORATION_RULES,
        "",
        QUESTION_QUALITY_GUIDANCE,
        "",
        """TOOL SYNTAX:
- filter_expr uses Python: "TL > 50 and TR == 'control'" (not SQL)
- derive_stat formula uses column names: "TL / IN" (not agg names)""",
        "",
        f"OUTPUT: JSON array with {target_questions} question plans:",
        f"```json\n{QUESTION_PLAN_SCHEMA}\n```",
        "",
        TURN_STRUCTURE_QUESTION_GEN,
        "",
        "Begin exploring. Write DONE when ready to output questions.",
    ]
    
    return "\n".join(sections)


def build_tool_feedback_prompt(dataset_description: str, data_overview: str) -> str:
    """Build prompt for tool feedback mode."""
    tool_docs = format_tool_docs(verbosity="compact")
    
    return f"""You are evaluating a tool library for data analysis. Identify gaps and friction.

DATASET:
{dataset_description}

DATA OVERVIEW:
```
{data_overview}
```

{tool_docs}

{TOOL_CALL_SYNTAX}

YOUR TASK:
1. Explore the dataset
2. Note friction with <TOOL_WISH>...</TOOL_WISH> tags
3. Write DONE, output JSON recommendations:

```json
[{{"name": "tool_name", "priority": "high|medium|low", "why": "...", "example_call": {{...}}, "returns": "..."}}]
```"""


def build_student_prompt(
    dataset_description: str,
    data_overview: str,
    verbosity: str = "minimal",
    include_data_overview: bool = False,
    filter_tools: set[str] | None = None,
) -> str:
    """
    Build minimal prompt for student model (local, limited context).

    Optimized for RL training where teacher traces are in context.

    Args:
        dataset_description: Description of the dataset
        data_overview: Pre-computed data exploration output
        verbosity: "minimal" for function signatures (default),
                   "compact" for key descriptions + required params
        include_data_overview: Include full data overview (default False,
                               assumes it's already in teacher trace context)
        filter_tools: Optional set of tool names to include (lazy loading)
    """
    tool_docs = format_tool_docs(verbosity=verbosity, filter_tools=filter_tools)

    # Minimal data reference if overview not included
    if include_data_overview:
        data_section = f"""DATA:
```
{data_overview}
```"""
    else:
        data_section = "DATA: See data overview in context above."

    return f"""Answer the question using the tools and data provided.

DATASET:
{dataset_description}

{data_section}

{tool_docs}

{TOOL_CALL_SYNTAX}

Output your answer as JSON (see examples in teacher trace for format)."""


# ============================================================================
# ROLLOUT CONFIG - For multi-turn execution
# ============================================================================

@dataclass
class RolloutConfig:
    """Configuration for a rollout (system prompt, intermediate messages)."""
    system_prompt: str
    mode: str
    continue_msg: str
    final_msg: str


def build_rollout_config(
    mode: str,
    dataset_description: str,
    data_overview: str,
    target_questions: int = 10,
) -> RolloutConfig:
    """Build rollout config for the given pipeline mode."""
    
    if mode == "question-gen":
        return RolloutConfig(
            system_prompt=build_question_generation_prompt(
                dataset_description, data_overview, target_questions
            ),
            mode=mode,
            continue_msg="\n\nContinue exploring. Write DONE when ready to output question plans.",
            final_msg="Turn limit reached. Output question plans now as JSON. Write DONE then JSON.",
        )
    
    elif mode == "question-answer":
        return RolloutConfig(
            system_prompt=build_episodes_prompt(dataset_description, data_overview),
            mode=mode,
            continue_msg="\n\nTool results above. Continue exploring (3-8 parallel calls). Write DONE when ready for episodes.",
            final_msg="Turn limit reached. Output episodes now as JSON. Write DONE then JSON.",
        )
    
    elif mode == "tool-feedback":
        return RolloutConfig(
            system_prompt=build_tool_feedback_prompt(dataset_description, data_overview),
            mode=mode,
            continue_msg="\n\nContinue. Note friction with <TOOL_WISH> tags. Write DONE when ready.",
            final_msg="Turn limit reached. Output tool recommendations as JSON.",
        )
    
    else:
        raise ValueError(f"Unknown mode '{mode}' (expected: question-gen, question-answer, tool-feedback)")


# ============================================================================
# UTILITIES
# ============================================================================

def extract_tools_from_trace(turns: list) -> set[str]:
    """
    Extract unique tool names from a teacher trace (list of Turn objects).

    Used for lazy tool loading: only include tool docs for tools the teacher used.

    Args:
        turns: List of Turn objects from teacher trace

    Returns:
        Set of tool names used in the trace
    """
    tools = set()
    for turn in turns:
        if hasattr(turn, 'tool_calls'):
            for tool_call in turn.tool_calls:
                if hasattr(tool_call, 'tool_name'):
                    tools.add(tool_call.tool_name)
    return tools


def generate_data_overview(csv_path: str = "data.csv") -> str:
    """Generate bootstrap exploration output for initial data inspection."""
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
