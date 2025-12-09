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
    """
    Build rollout config for the given pipeline mode.

    TODO: Will be implemented in Phase 2 with new prompt templates.
    """
    # Stub for now
    return RolloutConfig(
        system_prompt=f"Dataset: {dataset_description}\n\n{data_overview}\n\nTODO: Implement prompts in Phase 2",
        mode=mode,
        continue_msg="\n\nContinue.",
        final_msg="Complete your work.",
    )


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
