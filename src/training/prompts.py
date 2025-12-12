"""
Prompts for the training pipeline (student model).

This module contains the student prompt used during RL training.
"""

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
