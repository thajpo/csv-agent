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

TURN STRUCTURE (IMPORTANT):
Each turn must follow this exact pattern:
1. Write your reasoning: Explain what you'll do and why (1-3 sentences)
2. Write exactly ONE ```python code block
3. STOP - Do not write analysis or next steps after the code

The execution result will be shown at the start of your next turn.

Example:
"I'll filter to the control group and calculate the mean TL.
```python
df_control = df[df['TR'] == 'control']
mean_tl = df_control['TL'].mean()
print(f"Mean TL: {mean_tl}")
```"
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

TURN STRUCTURE:
1. Explain what you'll do (1-2 sentences)
2. Write exactly ONE ```python code block
3. Stop after the code

Execution results appear at the start of your next turn. Call submit(final_answer) when done.
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

The dataframe 'df' is already loaded.

TURN STRUCTURE:
Write your reasoning, then ONE ```python code block, then stop.
Execution results appear next turn. Call submit(final_answer) when done.
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
            continue_msg="\n\nWhat will you do next?",
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
            continue_msg="\n\nWhat will you do next?",
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
            continue_msg="\n\nWhat will you do next?",
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


# ============================================================================
# QUESTION GENERATION PROMPTS
# ============================================================================

EXPLORATION_SYSTEM_PROMPT = """You are a data analyst exploring a CSV dataset to generate interesting analytical questions.

Your task:
1. EXPLORE the dataset thoroughly using Python/pandas
2. DOCUMENT your exploration by writing notes about patterns, insights, and hypotheses.
3. GENERATE exactly 13 questions with the following distribution:
   - 3 EASY questions (1-3 logical steps)
   - 4 MEDIUM questions (4-6 logical steps)
   - 4 HARD questions (7-8 logical steps)
   - 2 VERY_HARD questions (9+ logical steps)

STATISTICAL/ANALYTICAL COMPLEXITY:
Questions involving statistical modeling and analytical operations are ENCOURAGED, but you must follow these determinism rules to ensure answers are reproducible and verifiable:

🟢 **GREEN LIGHT (Strongly Encouraged)**:
   - Simple Linear Regression (OLS): `np.polyfit`, `scipy.stats.linregress`
   - Correlations: Pearson, Spearman (`df.corr()`, `scipy.stats.pearsonr`)
   - Statistical Tests: T-tests, Chi-square (`scipy.stats.ttest_ind`, `scipy.stats.chisquare`)
   - Deterministic aggregations and mathematical transformations
   - *Why*: These are purely analytical (matrix math) and always yield identical results

🟡 **YELLOW LIGHT (Use with Explicit Seed)**:
   - Train/Test Splits: `train_test_split(random_state=42)`
   - K-Nearest Neighbors: `KNeighborsClassifier(n_neighbors=k)` (deterministic if no ties)
   - PCA: `PCA(random_state=42)` (watch for sign flipping across versions)
   - *Requirements*: If your question involves these, you MUST specify in the question text: "Use random_state=42 for all random operations"

🔴 **RED LIGHT (Avoid)**:
   - K-Means, DBSCAN, or other clustering (highly sensitive to initialization)
   - Random Forests, Gradient Boosting (too many hyperparameters, non-deterministic even with seeds across library versions)
   - Hyperparameter tuning (GridSearchCV, etc.) - too slow for execution environment
   - Deep learning models

**Seed Requirement**: For ANY question involving randomness (data splitting, sampling, etc.), you MUST include the phrase "Use seed=42 for all random operations" in the question text.

CRITICAL Guidelines:
- **Exploration First**: Use Python code in ```python blocks to explore the data. You MUST use print() to see outputs (e.g., `print(df.head())`).
- **Question Phrasing (CRITICAL)**:
   * **AVOID** recipe-style instructions like "Do X, then do Y, then report Z."
   * **USE** natural analytical phrasing where the steps are implied prerequisites.
   * *Bad Example:* "Group by Treatment, find the max Mean, then filter to that group and count rows."
   * *Good Example:* "How many observations belong to the Treatment group that exhibits the highest average Mean?"
   * The question should state the *goal*, not the *procedure*.
- **Logical Chaining**:
   * Ensure the answer requires multiple dependencies (Step B depends on the result of Step A).
   * **EASY**: Direct lookups or single aggregations.
   * **MEDIUM**: Comparing groups, simple filtering + aggregation.
   * **HARD**: Nested aggregations, derived metrics (e.g., ratios), or multi-stage filtering.
   * **VERY_HARD**: Complex logical trees, regressions on subsets, or optimization logic.
- **Verifiable Answers**: Questions must have a single, objective answer (Number, Boolean, specific String).

Output format (at the end):
```json
{
  "questions": [
    {
      "question": "What is the average TL for the 'control' group?",
      "hint": "Filter for the control group, then aggregate TL.",
      "n_steps": 2,
      "difficulty": "EASY"
    },
    {
      "question": "Among the treatment groups with at least 10 observations, which one displays the highest variance in TL?",
      "hint": "First, count observations per group and filter out small groups. Then, calculate variance for the remaining groups and identify the maximum.",
      "n_steps": 5,
      "difficulty": "MEDIUM"
    }
  ]
} """.strip()

# Minimum turns before allowing question generation
MIN_EXPLORATION_TURNS = 3


def get_exploration_continue_msg(turn_number: int, min_turns: int = 3) -> str:
    """Get context-appropriate continue message based on turn number."""
    if turn_number < min_turns:
        return f"\n\nContinue exploring the dataset. You must explore for at least {min_turns} turns before generating questions. Write Python code to examine the data."
    else:
        return "\n\nContinue exploring or generate the 13 questions in JSON format when ready. When finished, emit <DONE> after the JSON block."

EXPLORATION_CONTINUE_MSG = "\n\nContinue exploring the dataset. Write Python code to examine the data."
