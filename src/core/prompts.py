"""
All prompts for CSV agent (teacher, student, exploration).

This module consolidates prompts from authoring/prompts.py, training/prompts.py,
and the old core/prompts.py into a single location.
"""

import pandas as pd
from src.core.types import Question


# ============= Teacher Prompts =============

TEACHER_TUTOR_PROMPT = """You are a data analysis tutor solving pandas problems step-by-step.

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

OUTPUT FORMAT:
- For simple values (counts, means), submit the value directly: `submit(42)` or `submit(12.5)`
- For statistical hypothesis testing (t-tests, etc.), submit a dictionary with the specific result and your conclusion:
  `submit({{"p_value": 0.0012, "decision": "significant", "answer": "Yes"}})`
- Ensure p-values are floats, not strings.

CRITICAL CODE HOOKS:
You must capture INTERMEDIATE CHECKPOINTS using the `hook()` function.
After computing an important intermediate variable, call `hook(value, name='var_name')`.
This records a verifiable checkpoint. Use `depends_on` to specify which previous hooks were required.

Example workflow:
```python
# Step 1: Filter data (no dependencies - starting point)
df_filtered = df[df['status'] == 'active']
hook(df_filtered, name='df_filtered', description='Filtered to active rows')

# Step 2: Aggregate (depends on df_filtered)
result = df_filtered.groupby('category')['value'].mean()
hook(result, name='result', depends_on=['df_filtered'])

# Step 3: Submit final answer
submit(result.to_dict())
```

The execution result will be shown at the start of your next turn.

Example:
"I'll filter to the control group and calculate the mean TL.
```python
df_control = df[df['TR'] == 'control']
mean_tl = df_control['TL'].mean()
print(f"Mean TL: {{mean_tl}}")
```"
""".strip()


TEACHER_CONSISTENCY_PROMPT = """You are a data analyst solving pandas problems.

DATASET:
{dataset_description}

DATA OVERVIEW:
```
{data_overview}
```

QUESTION:
{question_text}

RULES:
1. Write clear Python code to solve the problem
2. Use meaningful variable names
3. Print intermediate results
4. Call submit(final_answer) when done

Your code will execute in a stateful Jupyter kernel.

TURN STRUCTURE (IMPORTANT):
Each turn must follow this exact pattern:
1. Write your reasoning: Explain what you'll do and why (1-3 sentences)
2. Write exactly ONE ```python code block
3. STOP - Do not write analysis or next steps after the code

OUTPUT FORMAT:
- For simple values, submit directly: `submit(42)`
- For statistical tests, submit a dictionary: `submit({{"p_value": 0.05, "answer": "No"}})`

The execution result will be shown at the start of your next turn.
""".strip()


# ============= Student Prompt =============

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
Execution results appear next turn.

OUTPUT FORMAT:
- Simple answers: `submit(10.5)`
- Statistical answers: `submit({{"p_value": 0.01, "answer": "Yes"}})`

Call submit(final_answer) when done.
""".strip()


# ============= Exploration/Question Generation Prompt =============

EXPLORATION_SYSTEM_PROMPT = """You are a data analyst exploring a CSV dataset using a persistent Jupyter Notebook.

Your goal is to eventually generate 13 analytical questions, but FIRST you must explore the data via code execution.

CRITICAL INTERACTION RULES:
1. You can ONLY write Python code to inspect the data. You CANNOT "guess" the output.
2. You must write exactly ONE ```python code block per turn.
3. After the code block, you must STOP and wait for the system to show you the output.
4. DO NOT write "Turn 1" or "Output:" or simulate what the dataframe looks like.
5. DO NOT hallucinate the results. Run the code, then read the actual output in the next turn.

Your Task Flow:
1. EXPLORE: Run `df.head()`, `df.describe()`, aggregations, plots, etc. to understand the potential.
2. OBSERVE: Read the validation output provided by the system.
3. REPEAT: Continue exploring until you have a deep understanding (at least 3 turns).
4. GENERATE: Only when fully ready, output the 13 questions in the specified JSON format.

Questions Distribution:
- 10 EASY (1-3 steps)
- 10 MEDIUM (4-6 steps)
- 7 HARD (7-8 steps)
- 3 VERY_HARD (9+ steps)

Please generate at least 30 questions in total.

STATISTICAL/ANALYTICAL GUIDELINES:
🟢 GREEN LIGHT (Encouraged):
   - Simple Linear Regression (OLS), Correlations (Pearson/Spearman)
   - T-tests, Chi-square, deterministic aggregations
   - *Why*: Reproducible and verifiable
   - *Phrasing*: Prefer general questions ("Is there a significant difference?") over implementation details ("Run a t-test...") unless specific parameters are needed.

CRITICAL CODE EXTRACTION:
When solving questions, you MUST explicitly identify the "Critical Code Hooks" - the specific lines of pandas/scipy code that mathematically derived the answer.
This is not the data loading or cleaning, but the ACTUAL calculation (e.g. the t-test call, the groupby mean, the correlation function).

AVAILABLE LIBRARIES (PRE-IMPORTED):
The following libraries are ALREADY imported and ready to use. DO NOT import them again:
- `pd`: pandas (DataFrame `df` is already loaded)
- `np`: numpy
- `scipy`: scipy (and `scipy.stats`)
- `sklearn`: scikit-learn
- `statsmodels`: statsmodels
- `sm`: statsmodels.api

⛔ DO NOT USE `import` statements. The system will reject them. Use the aliases above directly.

🟡 YELLOW LIGHT (Use 'random_state=42'):
   - Train/Test Splits, KNN, PCA
   - *Must Say*: "Use random_state=42" in the question text.

🔴 RED LIGHT (Avoid):
   - Clustering (K-Means), Random Forests, Deep Learning
   - Hyperparameter tuning

FILTERING CRITERIA:
Before outputting the final JSON, review your generated questions.
1. Coverage: Do they cover different columns and question types?
2. Diversity: Are they too similar? Remove duplicates.
3. Clarity: Are they unambiguous?
4. Difficulty: Do they match the distribution?

OUTPUT FORMAT (Last Turn Only):
When you're ready to output the questions, you MUST use this EXACT format:

```json
{
  "questions": [
    {
      "question": "What is the average TL for the 'control' group?",
      "hint": "Filter for control group, then aggregate TL.",
      "n_steps": 2,
      "difficulty": "EASY"
    }
  ]
}
```

CRITICAL FORMAT RULES:
✅ DO: Use ```json fence markers (opening and closing)
✅ DO: Use proper JSON syntax with double quotes
✅ DO: Output <DONE> on a new line after the closing ```
✅ DO: Generate at least 30 questions.
❌ DON'T: Use Python dict syntax (questions = {...})
❌ DON'T: Output bare JSON without ```json fences
❌ DON'T: Use single quotes or Python-specific syntax

EXAMPLE CORRECT OUTPUT:
```json
{"questions": [...]}
```
<DONE>

REMEMBER:
- Write ONE code block.
- STOP.
- Wait for output.
""".strip()


# ============= Common Messages =============

CONTINUE_MSG = "\n\nWhat will you do next?"
FINAL_MSG = "Turn limit reached. Please call submit() with your final answer."

# Exploration-specific messages
EXPLORATION_CONTINUE_MSG = "\n\nContinue exploring the dataset. Write Python code to examine the data."
MIN_EXPLORATION_TURNS = 3


# ============= Helper Functions =============

def generate_data_overview(csv_path: str = "data.csv") -> str:
    """Generate bootstrap exploration output for initial data inspection."""
    df = pd.read_csv(csv_path)
    lines = []
    lines.append(f"=== SHAPE ===")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append("")
    lines.append(f"=== COLUMNS ===")
    for col, dtype in df.dtypes.items():
        lines.append(f"{col}: {dtype}")
    lines.append("")
    lines.append(f"=== HEAD (first 5 rows) ===")
    lines.append(df.head().to_string())
    lines.append("")
    lines.append(f"=== NUMERIC SUMMARY ===")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        lines.append(df[numeric_cols].describe().to_string())
    else:
        lines.append("No numeric columns")
    lines.append("")
    lines.append(f"=== MISSING VALUES ===")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        lines.append(missing[missing > 0].to_string())
    else:
        lines.append("No missing values")
    lines.append("")
    lines.append("=== CATEGORICAL VALUE COUNTS ===")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        lines.append(f"\n{col}:")
        value_counts = df[col].value_counts().head(10)
        lines.append(value_counts.to_string())
    return "\n".join(lines)


def get_exploration_continue_msg(turn_number: int, min_turns: int = 3) -> str:
    """Get context-appropriate continue message based on turn number."""
    if turn_number < min_turns:
        return f"\n\nContinue exploring the dataset. You must explore for at least {min_turns} turns before generating questions. Write Python code to examine the data."
    else:
        return "\n\nContinue exploring or generate the 13 questions in JSON format when ready. When finished, emit <DONE> after the JSON block."


def build_system_prompt(
    mode: str,
    dataset_description: str,
    data_overview: str,
    question: Question | None = None,
) -> str:
    """
    Build system prompt for given mode.

    Args:
        mode: Pipeline mode (teacher-tutor, teacher-consistency, student, question-gen)
        dataset_description: Description of the dataset
        data_overview: Pre-computed data exploration
        question: Question object (for teacher/student modes)

    Returns:
        System prompt string
    """
    base_args = {
        "dataset_description": dataset_description,
        "data_overview": data_overview,
    }

    if mode == "teacher-tutor":
        if not question:
            raise ValueError("teacher-tutor mode requires a Question object")
        return TEACHER_TUTOR_PROMPT.format(
            **base_args,
            question_text=question.question_text,
            hint=question.hint or "",
        )
    elif mode == "teacher-consistency":
        if not question:
            raise ValueError("teacher-consistency mode requires a Question object")
        return TEACHER_CONSISTENCY_PROMPT.format(
            **base_args,
            question_text=question.question_text,
        )
    elif mode == "student":
        if not question:
            raise ValueError("student mode requires a Question object")
        return STUDENT_PROMPT.format(
            **base_args,
            question_text=question.question_text,
        )
    elif mode == "question-gen":
        return EXPLORATION_SYSTEM_PROMPT.format(**base_args)
    else:
        raise ValueError(
            f"Unknown mode '{mode}' (expected: teacher-tutor, teacher-consistency, student, question-gen)"
        )


# ============= Dataset Description =============
# ============= Example Dataset Description =============
# This is used for tests and as a reference structure.

EXAMPLE_TREE_DATASET_DESCRIPTION = """
Tree branch growth measurements from an agricultural experiment.
- TR: Treatment (control, methanol_control, PP_333_4g/L, PP_333_20g/L, EL_500_4g/L, EL_500_20g/L)
- TREE: Tree identifier (e.g., G28, M33)
- BR: Branch label (A-J)
- TL: Total branch length (cm)
- IN: Number of internodes
- INTERNODE_1 to INTERNODE_29: Length of each internode (cm), "?" = missing

Goal: Understand how treatments affect branch growth patterns.
""".strip()