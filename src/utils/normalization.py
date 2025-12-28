"""
Value normalization utilities.

DEPRECATED: This module re-exports from csv_spec for backward compatibility.
New code should import directly from csv_spec.

Example:
    # Old (still works):
    from src.utils.normalization import normalize_value

    # New (preferred):
    from csv_spec import normalize_value
"""

# Re-export from csv_spec for backward compatibility
from csv_spec import normalize_value

__all__ = ["normalize_value"]
