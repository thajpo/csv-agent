"""
Value normalization utilities.

Standardizes answer formats for comparison between teacher traces.
This module is used both:
1. Host-side in teacher.py for answer comparison
2. Injected into Docker containers via inspect.getsource() for submission normalization
"""

import pandas as pd
import numpy as np
from typing import Any


def normalize_value(val: Any) -> Any:
    """
    Standardize answer formats for better comparison.
    Converts DataFrames/Series to Dictionaries or Scalars where appropriate.
    
    NOTE: This function is injected into Docker containers at boot time.
    Keep it self-contained (only uses pd, np which are pre-imported in container).
    """
    if val is None:
        return None

    # Handle Pandas/Numpy types
    if isinstance(val, pd.DataFrame):
        if val.empty:
            return {}
        
        # 1x1 -> scalar
        if val.shape == (1, 1):
            return val.iloc[0, 0]
        
        # 1 column -> dict (if index meaningful) or list
        if val.shape[1] == 1:
            series = val.iloc[:, 0]
            if not isinstance(val.index, pd.RangeIndex):
                return series.to_dict()
            return series.tolist()

        # 2 columns -> dict {col0: col1}
        if val.shape[1] == 2:
            return dict(zip(val.iloc[:, 0], val.iloc[:, 1]))
        
        # Default: list of records
        return val.to_dict('records')
    
    if isinstance(val, pd.Series):
        if val.size == 1:
            return val.iloc[0]
        return val.to_dict()

    if isinstance(val, np.generic):
        return val.item()
    
    return val
