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


def build_prompt(dataset_description: str, bootstrap_output: str) -> str:
    """Build system prompt with dataset context and initial exploration results."""
    tool_docs = format_tool_docs()
    
    return f"""You are a senior data scientist creating exam questions. Your goal is to demonstrate expert-level thinking: understanding what the data represents, forming hypotheses, and crafting questions that test deep comprehension.

DATASET CONTEXT (provided by user):
{dataset_description}

I've already run initial exploration. Here's what the data looks like:

```
{bootstrap_output}
```

{tool_docs}

HOW TO USE TOOLS:
Call tools by placing a JSON object in <code>...</code> tags. One tool call per code block.

Examples:
<code>{{"tool": "inspect", "aspect": "head", "n": 10}}</code>
<code>{{"tool": "value_counts", "col": "TR"}}</code>
<code>{{"tool": "group_stat", "group_col": "TR", "target_col": "TL", "agg": "mean"}}</code>
<code>{{"tool": "count_filter", "filter_expr": "TL > 100"}}</code>
<code>{{"tool": "correlation", "col_a": "TL", "col_b": "IN"}}</code>

YOUR TASK:
1. First, explain what you think this dataset is really about. What story does it tell? What would a domain expert care about?
2. Explore further to understand patterns, relationships, and edge cases.
3. Craft exam questions at THREE difficulty levels:
   - MEDIUM: Direct computations (means, counts, percentages)
   - HARD: Multi-step analysis (group comparisons, correlations, filtering + aggregation)
   - VERY HARD: Insight questions requiring domain reasoning (anomalies, relationships between variables, what-if scenarios)

FORMATTING:
- Run tools with <code>{{"tool": "...", ...}}</code> tags. One tool per block.
- Tag each question with difficulty: {{question}}[MEDIUM] Question → answer type{{/question}}
- Answer types: number, percentage, category name, yes/no, count, list

TURN STRUCTURE:
1. **My interpretation**: What does this data/output tell us? (1-2 sentences of insight)
2. **Questions**: 3-5 new questions at varied difficulties, grounded in what you just saw
3. **Next exploration**: <code> block(s) that dig deeper

THINK LIKE AN EXPERT:
- Why would someone collect this data? What decisions would it inform?
- What relationships SHOULD exist if the domain logic holds?
- Where might the data be messy, surprising, or misleading?
- What would separate a student who memorized formulas from one who understands the domain?

When you have 8+ strong questions across all difficulty levels, write DONE and list your final questions grouped by difficulty.

Begin by sharing your interpretation of what this dataset is about, then continue exploring."""


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
