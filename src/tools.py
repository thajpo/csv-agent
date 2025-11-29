"""
The LLM can only interact with the environment with the set of tools provided.
This is constrained to pandas operations for now.

Tool categories:
1. Inspection, visualisation, descriptive statistics:
- 
- group_stat: Calculate the group statistic of the target column.
- correlation: Calculate the correlation between two columns.
- count_filter: Count the number of rows that match a filter.
- model_eval: Evaluate a model on the data.
- python_code: Execute arbitrary Python code.
"""

import pandas as pd

def group_stat(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    agg: str,
) -> float:
    """
    Calculate the group statistic of the target column.
    """
    return df.groupby(group_col)[target_col].agg(agg).iloc[0]

