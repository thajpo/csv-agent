"""
Artifact hashing utilities for state matching.

This module provides utilities for hashing DataFrames and scalar values
to compare intermediate states between teacher and student executions.
"""

from typing import Any
from hashlib import sha256
import pickle
import pandas as pd


def hash_artifact(obj: Any) -> str:
    """
    Hash a DataFrame or scalar value for state matching.

    Args:
        obj: A pandas DataFrame or scalar (int, float, str, bool, None)

    Returns:
        16-character hex hash of the object's state
    """
    if isinstance(obj, pd.DataFrame):
        # Hash structure (columns, dtypes) and values
        try:
            content = pickle.dumps({
                'columns': obj.columns.tolist(),
                'dtypes': obj.dtypes.tolist(),
                'values': obj.values.tobytes(),
                'shape': obj.shape
            })
        except Exception:
            # Fallback for non-byteable dtypes
            content = pickle.dumps({
                'columns': obj.columns.tolist(),
                'dtypes': [str(dt) for dt in obj.dtypes],
                'values': obj.to_dict(),
                'shape': obj.shape
            })
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        # Scalar: just pickle it
        content = pickle.dumps(obj)
    else:
        # Try to pickle anything else
        content = pickle.dumps(obj)

    return sha256(content).hexdigest()[:16]  # 16 chars is enough
