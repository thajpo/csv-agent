"""
Artifact hashing utilities for state matching.

This module provides utilities for hashing DataFrames and scalar values
to compare intermediate states between teacher and student executions.

IMPORTANT: Uses same algorithm as container's hook() function:
  1. Normalize value (DataFrame → dict, etc.)
  2. Round floats to consistent precision (default: 2 decimals)
  3. JSON serialize with sort_keys=True
  4. SHA256 hash, truncated to 16 chars

IMPORTANT: This is a CONTRACT file. Changes here affect both environment and trainer.
"""

from typing import Any
from hashlib import sha256
import json

from csv_spec.normalization import normalize_value

# Default precision for float rounding before hashing.
# This ensures small floating-point differences don't produce different hashes.
# Matches the precision used in synthetic templates (2 decimals for stats, 4 for p-values).
DEFAULT_HASH_PRECISION = 2


def _round_floats(obj: Any, precision: int) -> Any:
    """Recursively round all floats in a structure to consistent precision."""
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        rounded = [_round_floats(item, precision) for item in obj]
        return tuple(rounded) if isinstance(obj, tuple) else rounded
    return obj


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy/pandas types."""
    import numpy as np

    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)  # Convert numpy bool to Python bool for proper JSON serialization
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def hash_artifact(obj: Any, precision: int = DEFAULT_HASH_PRECISION) -> str:
    """
    Hash a value for state matching.

    Uses normalize → round → JSON → SHA256 approach, matching the container's
    hook() function for consistent hashes across host and container.

    Args:
        obj: Any value (DataFrame, scalar, dict, etc.)
        precision: Decimal places to round floats to (default: 2)

    Returns:
        16-character hex hash of the normalized object
    """
    normalized = normalize_value(obj)
    rounded = _round_floats(normalized, precision)
    json_str = json.dumps(rounded, sort_keys=True, default=_json_default)
    return sha256(json_str.encode()).hexdigest()[:16]
