"""Type definitions for compositional program generation.

This module defines the type system used by the grammar search to ensure
operator chains are valid. The key design decision is to track TYPE COUNTS
(not just presence) to support multi-input operators.

Why counts matter:
- correlation needs 2 NumCol inputs (not 1)
- ratio needs 2 NumCol inputs
- regression needs multiple columns
- Without counts, the grammar can't distinguish "has 1" from "has 2"

Previous approach (frozenset[str]) only tracked presence, which blocked
multi-input composition. Using Counter allows us to say "needs NumCol × 2".
"""

from collections import Counter
from typing import Final

# Type system for operator composition
# Using Counter to track MULTIPLICITY, not just presence
# Example: Counter({"NumCol": 2, "Table": 1}) means "2 NumCols and 1 Table"
Types = Counter[str]

# Predefined type constants
# These are the base types that operators consume and produce
TYPE_TABLE: Final = "Table"  # The dataframe itself
TYPE_NUM_COLS: Final = "NumCols"  # Collection of numeric columns (for selection)
TYPE_CAT_COLS: Final = "CatCols"  # Collection of categorical columns
TYPE_NUM_COL: Final = "NumCol"  # Single bound numeric column
TYPE_CAT_COL: Final = "CatCol"  # Single bound categorical column
TYPE_GROUPS: Final = "Groups"  # Grouped data (from groupby)
TYPE_NUM_SERIES: Final = "NumSeries"  # Transformed numeric series
TYPE_SCALAR: Final = "Scalar"  # Single scalar value
TYPE_BOOL: Final = "Bool"  # Boolean value
TYPE_DICT: Final = "Dict"  # Dictionary result (final answer)


def is_subtype(actual: Types, required: Types) -> bool:
    """Check if actual types satisfy all required type counts.

    This is the key compatibility check for the grammar search.

    Args:
        actual: Counter of types currently available in the chain
        required: Counter of types needed by the operator

    Returns:
        True if actual has AT LEAST as many of each type as required

    Example:
        >>> actual = Counter({"NumCol": 2, "Table": 1})
        >>> required = Counter({"NumCol": 2})
        >>> is_subtype(actual, required)
        True  # We have 2 NumCols, operator needs 2

        >>> required = Counter({"NumCol": 3})
        >>> is_subtype(actual, required)
        False  # We have 2, but operator needs 3

    Why this design:
    - Simple set inclusion (required ⊆ actual) doesn't work for counts
    - We need to check: actual[type] >= required[type] for all types
    - Counter's subset check does exactly this
    """
    # Counter.issubset checks: all counts in required <= corresponding counts in actual
    return required <= actual


def add_types(current: Types, outputs: list[str]) -> Types:
    """Add output types to current type state.

    This is used by the grammar search when extending a chain.
    Outputs are ADDED to the current types (not replacing).

    Args:
        current: Current type counts
        outputs: List of type names produced by the operator

    Returns:
        New Counter with outputs added

    Example:
        >>> current = Counter({"Table": 1})
        >>> add_types(current, ["NumCols"])
        Counter({"Table": 1, "NumCols": 1})
    """
    new_types = current.copy()
    for output in outputs:
        new_types[output] += 1
    return new_types


def types_from_list(type_list: list[str]) -> Types:
    """Create a Types Counter from a list of type names.

    Convenience function for creating operator input/output specifications.

    Args:
        type_list: List of type names (can have duplicates for multi-input)

    Returns:
        Counter mapping type names to their counts

    Example:
        >>> types_from_list(["NumCol", "NumCol"])  # correlation needs 2
        Counter({"NumCol": 2})
    """
    return Counter(type_list)
