"""Type definitions for compositional program generation."""

from typing import Final

# Type system for operator composition
Types = frozenset[str]  # Set of type names (Table, NumCol, etc.)

# Predefined types
TYPE_TABLE: Final = "Table"
TYPE_NUM_COLS: Final = "NumCols"
TYPE_CAT_COLS: Final = "CatCols"
TYPE_NUM_COL: Final = "NumCol"
TYPE_CAT_COL: Final = "CatCol"
TYPE_GROUPS: Final = "Groups"
TYPE_NUM_SERIES: Final = "NumSeries"
TYPE_SCALAR: Final = "Scalar"
TYPE_BOOL: Final = "Bool"
TYPE_DICT: Final = "Dict"


# For type compatibility checks
def is_subtype(actual: Types, required: Types) -> bool:
    """Check if actual types include all required types."""
    return required.issubset(actual)
