"""
Artifact hashing utilities for state matching.

DEPRECATED: This module re-exports from csv_spec for backward compatibility.
New code should import directly from csv_spec.

Example:
    # Old (still works):
    from src.utils.hashing import hash_artifact

    # New (preferred):
    from csv_spec import hash_artifact
"""

# Re-export from csv_spec for backward compatibility
from csv_spec import hash_artifact

__all__ = ["hash_artifact"]
