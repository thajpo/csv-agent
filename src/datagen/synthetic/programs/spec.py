"""Core data structures for compositional programs."""

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
    """

    variable_counter: int = 0
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    id_like_cols: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpInstance:
    """A specific instance of an operator with parameters."""

    op_name: str
    params: dict[str, Any]


@dataclass
class ProgramSpec:
    """Compositional program specification.

    Minimal fields for Phase-1. Programs must be serializable and stable for caching.
    """

    name: str
    ops: list[OpInstance]
    output_type: str
    output_schema: str
    difficulty: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Op:
    """Operator definition with types, attributes, and preconditions."""

    name: str
    inputs: list[str]  # type names
    outputs: list[str]
    attributes: list[str]  # semantic tags: selector, analysis, evidence, decision, test
    emit: Callable[[State], str]
    update: Callable[[State], None]
    precondition: Callable[[dict, State], bool]
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)
    emits_answer: bool = False
    requires_bindings: dict[str, bool] = field(
        default_factory=dict
    )  # which columns this op needs bindings for
