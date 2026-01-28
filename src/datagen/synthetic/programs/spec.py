"""Core data structures for compositional programs.

This module defines the data structures that represent compositional programs
and their execution state.

KEY DESIGN DECISION: Explicit binding requirements
--------------------------------------------------
Operators declare their binding requirements explicitly. This enables systematic
enumeration of all valid column combinations without arbitrary selection.

Example binding requirements:
- bind_num_col_1: requires_bindings={"num_col_1": True}
- bind_num_col_2: requires_bindings={"num_col_2": True}
- correlation: requires_bindings={}  # uses num_col_1 and num_col_2 internally

Why explicit bindings:
- No arbitrary "first column" selection
- All valid combinations are enumerated
- Questions are reproducible and verifiable
- Easy to track what each operator needs

Previous limitation:
Hardcoded bindings (selected_col, cat_col) couldn't scale to multi-input ops.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class State:
    """Compile-time state for program compilation.

    Stores bindings and metadata only (not runtime data):
    - selected columns (names)
    - group sizes / counts
    - evidence values computed earlier (p-values, outlier scores)
    - variable names for emitted code

    The bindings dict is the key mechanism for passing column names
    and other parameters between operators in a chain.

    Example bindings after a correlation chain:
    {
        "num_col_1": "sales",
        "num_col_2": "profit",
        "profile": {...}  # dataset profile
    }
    """

    variable_counter: int = 0
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    id_like_cols: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpInstance:
    """A specific instance of an operator with bound parameters.

    The params dict contains concrete values for the operator's
    binding requirements. These are filled during enumeration.

    Example:
        OpInstance(
            op_name="bind_num_col_1",
            params={"num_col_1": "sales"}
        )
    """

    op_name: str
    params: dict[str, Any]


@dataclass
class ProgramSpec:
    """Compositional program specification.

    A program is a sequence of operators with specific column bindings.
    Programs are serializable and stable for caching.

    Example program for correlation:
        name: "program_correlation_sales_profit"
        ops: [
            OpInstance("select_numeric_cols", {}),
            OpInstance("bind_num_col_1", {"num_col_1": "sales"}),
            OpInstance("bind_num_col_2", {"num_col_2": "profit"}),
            OpInstance("correlation", {})
        ]
    """

    name: str
    ops: list[OpInstance]
    output_type: str
    output_schema: str
    difficulty: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Op:
    """Operator definition with types, attributes, and binding requirements.

    This is the core building block of compositional programs. Each operator
    declares:

    1. Type signature (inputs/outputs) - for grammar validation
    2. Binding requirements - for enumeration
    3. Code emission - for execution
    4. Preconditions - for validity checking

    The inputs list uses COUNTS to support multi-input operators:
    - ["NumCol"] means needs 1 NumCol
    - ["NumCol", "NumCol"] means needs 2 NumCols (distinct)

    The requires_bindings dict specifies which params must be provided:
    - {"num_col_1": True} - this op needs num_col_1 bound
    - {} - no bindings needed (uses previously bound values)

    Example operator (correlation):
        name="correlation"
        inputs=["NumCol", "NumCol"]  # needs 2 NumCols
        outputs=["Dict"]
        attributes=["analysis"]
        requires_bindings={}  # uses num_col_1 and num_col_2 from previous binds
    """

    name: str
    inputs: list[str]  # type names (with duplicates for multi-input)
    outputs: list[str]
    attributes: list[str]  # semantic tags: selector, analysis, evidence, decision, test
    emit: Callable[[State], str]
    update: Callable[[State], None]
    precondition: Callable[[dict, State], bool]
    reads: set[str] = field(default_factory=set)  # which bindings this op reads
    writes: set[str] = field(default_factory=set)  # which bindings this op writes
    emits_answer: bool = False  # whether this op produces the final answer
    requires_bindings: dict[str, bool] = field(
        default_factory=dict
    )  # binding keys this op needs filled
