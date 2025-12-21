"""
Artifact hashing utilities for state matching.

This module provides utilities for hashing DataFrames and scalar values
to compare intermediate states between teacher and student executions.

IMPORTANT: Uses same algorithm as container's hook() function:
  1. Normalize value (DataFrame → dict, etc.)
  2. JSON serialize with sort_keys=True
  3. SHA256 hash, truncated to 16 chars
"""

from typing import Any
from hashlib import sha256
import json

from src.utils.normalization import normalize_value


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy/pandas types."""
    import numpy as np
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def hash_artifact(obj: Any) -> str:
    """
    Hash a value for state matching.
    
    Uses normalize → JSON → SHA256 approach, matching the container's
    hook() function for consistent hashes across host and container.

    Args:
        obj: Any value (DataFrame, scalar, dict, etc.)

    Returns:
        16-character hex hash of the normalized object
    """
    normalized = normalize_value(obj)
    json_str = json.dumps(normalized, sort_keys=True, default=_json_default)
    return sha256(json_str.encode()).hexdigest()[:16]

