"""
Prompts and templates for the data analysis pipeline.

Architecture:
- Teacher prompts: Guide free-form Python code generation (tutor mode and consistency mode)
- Student prompts: Minimal guidance for RL training
- Utilities: Data overview generation and dataset description
"""

import pandas as pd
from dataclasses import dataclass


# ============================================================================
# TEACHER PROMPTS - Free-form Python code generation
# ============================================================================

TEACHER_TUTOR_MODE_PROMPT = """You are a data analysis tutor solving pandas problems step-by-step.

DATASET:
{dataset_description}

DATA OVERVIEW:
```
{data_overview}
```

QUESTION:
{question_text}

HINT:
{hint}

RULES:
1. Write verbose, educational Python code
2. Use meaningful intermediate variable names (df_filtered, df_grouped, etc.)
3. Avoid complex one-liners - break down into steps
4. Print intermediate results to verify your work
5. Call submit(final_answer) when done

Your code will execute in a stateful Jupyter kernel. You can:
- Inspect data: df.head(), df.info(), df.describe()
- Debug errors: try different approaches across turns
- Print intermediate results: print(df_filtered.shape)
- Build incrementally: define variables across multiple cells

Write ONE code cell per turn in ```python blocks. End with submit(final_answer) when you have the answer.
""".strip()

TEACHER_CONSISTENCY_PROMPT = """You are a data analyst solving pandas problems efficiently.

DATASET:
{dataset_description}

DATA OVERVIEW:
```
{data_overview}
```

QUESTION:
{question_text}

Your code will execute in a stateful Jupyter kernel. The dataframe 'df' is already loaded.

Write Python code in ```python blocks to solve this question. Call submit(final_answer) when done.
""".strip()

STUDENT_PROMPT = """Solve this data analysis question using Python and pandas.

DATASET:
{dataset_description}

DATA OVERVIEW:
```
{data_overview}
```

QUESTION:
{question_text}

The dataframe 'df' is already loaded. Write Python code in ```python blocks.
Call submit(final_answer) when you have the answer.
""".strip()


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
    question_text: str = "",
    hint: str = "",
    target_questions: int = 10,
) -> RolloutConfig:
    """
    Build rollout config for the given pipeline mode.

    Args:
        mode: Pipeline mode (teacher-tutor, teacher-consistency, student, etc.)
        dataset_description: Description of the dataset
        data_overview: Pre-computed data exploration
        question_text: The question to solve (for teacher/student modes)
        hint: Optional hint for teacher tutor mode
        target_questions: Number of questions to generate (for question-gen mode)
    """
    if mode == "teacher-tutor":
        return RolloutConfig(
            system_prompt=TEACHER_TUTOR_MODE_PROMPT.format(
                dataset_description=dataset_description,
                data_overview=data_overview,
                question_text=question_text,
                hint=hint
            ),
            mode=mode,
            continue_msg="\n\nExecution result above. Continue if needed, or call submit() if done.",
            final_msg="Turn limit reached. Please call submit() with your final answer.",
        )

    elif mode == "teacher-consistency":
        return RolloutConfig(
            system_prompt=TEACHER_CONSISTENCY_PROMPT.format(
                dataset_description=dataset_description,
                data_overview=data_overview,
                question_text=question_text
            ),
            mode=mode,
            continue_msg="\n\nExecution result above. Continue if needed, or call submit() if done.",
            final_msg="Turn limit reached. Please call submit() with your final answer.",
        )

    elif mode == "student":
        return RolloutConfig(
            system_prompt=STUDENT_PROMPT.format(
                dataset_description=dataset_description,
                data_overview=data_overview,
                question_text=question_text
            ),
            mode=mode,
            continue_msg="\n\nExecution result above. Continue if needed, or call submit() if done.",
            final_msg="Turn limit reached. Please call submit() with your final answer.",
        )

    else:
        raise ValueError(f"Unknown mode '{mode}' (expected: teacher-tutor, teacher-consistency, student)")


# ============================================================================
# UTILITIES
# ============================================================================

def generate_data_overview(csv_path: str = "data.csv") -> str:
    """
    Generate bootstrap exploration output for initial data inspection.

    TODO: Currently uses old tool functions. Will be simplified in Phase 2
    to use direct pandas operations.
    """
    df = pd.read_csv(csv_path)
    lines = []

    # Basic shape info
    lines.append(f"=== SHAPE ===")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append("")

    # Column names and types
    lines.append(f"=== COLUMNS ===")
    for col, dtype in df.dtypes.items():
        lines.append(f"{col}: {dtype}")
    lines.append("")

    # Head
    lines.append(f"=== HEAD (first 5 rows) ===")
    lines.append(df.head().to_string())
    lines.append("")

    # Numeric summary
    lines.append(f"=== NUMERIC SUMMARY ===")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        lines.append(df[numeric_cols].describe().to_string())
    else:
        lines.append("No numeric columns")
    lines.append("")

    # Missing values
    lines.append(f"=== MISSING VALUES ===")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        lines.append(missing[missing > 0].to_string())
    else:
        lines.append("No missing values")
    lines.append("")

    # Categorical value counts
    lines.append("=== CATEGORICAL VALUE COUNTS ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        lines.append(f"\n{col}:")
        value_counts = df[col].value_counts().head(10)
        lines.append(value_counts.to_string())

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
