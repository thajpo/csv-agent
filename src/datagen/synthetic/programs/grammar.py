"""Grammar search for compositional program generation.

This module implements the BFS search for valid operator chains.

KEY DESIGN DECISION: Type counts, not just presence
---------------------------------------------------
The grammar tracks how many of each type are available, not just whether
a type exists. This enables multi-input operators like correlation that
need two distinct columns.

Example chain evolution:
1. Start: Counter({"Table": 1})
2. After select_numeric_cols: Counter({"Table": 1, "NumCols": 1})
3. After bind_num_col_1: Counter({"Table": 1, "NumCols": 1, "NumCol": 1})
4. After bind_num_col_2: Counter({"Table": 1, "NumCols": 1, "NumCol": 2})
5. After correlation: Counter({"Table": 1, "NumCols": 1, "NumCol": 2, "Dict": 1})

Why this matters:
- correlation needs NumCol × 2 (two different columns)
- Without counts, we can't distinguish step 3 from step 4
- The grammar would think correlation is valid at step 3 (only 1 NumCol)

Previous limitation:
The old system used set[str] which only tracked presence. This blocked
multi-input operators and forced awkward workarounds (internal selection).
"""

from typing import List
from collections import Counter

from src.datagen.synthetic.programs.types import (
    Types,
    TYPE_TABLE,
    is_subtype,
    add_types,
    types_from_list,
)
from src.datagen.synthetic.programs.operators import get_operator, OPERATORS
from src.datagen.synthetic.programs.spec import State, OpInstance


def list_operator_names() -> List[str]:
    """Get all operator names."""
    return list(OPERATORS.keys())


def search_programs(profile: dict, max_depth: int = 15) -> List[List[OpInstance]]:
    """Search for valid operator chains using BFS from Table state.

    The search tracks type COUNTS to support multi-input operators.
    Each operator declares its input requirements as a Counter, and the
    grammar checks if the current chain has enough of each type.

    Args:
        profile: Dataset profile with column info
        max_depth: Maximum chain length

    Returns:
        List of valid operator chains

    Example valid chain:
        [select_numeric_cols, bind_num_col_1, bind_num_col_2, correlation]

    Why BFS:
    - Explores shorter chains first (better for finding simple questions)
    - Guarantees we find all chains up to max_depth
    - Easy to limit total iterations for performance
    """
    # Start with empty chain and just the Table
    # Using Counter to track type counts from the start
    initial_types = Counter({TYPE_TABLE: 1})

    queue: List[tuple[List[OpInstance], Types, State]] = [([], initial_types, State())]
    valid_chains: List[List[OpInstance]] = []

    iteration = 0
    max_iterations = 10000  # Increased for 10-15 step chains

    while queue and iteration < max_iterations:
        iteration += 1
        chain, current_types, state = queue.pop(0)

        # Skip if depth limit reached
        if len(chain) >= max_depth:
            continue

        # Try to extend chain with each operator
        for op_name in list_operator_names():
            op_def = get_operator(op_name)
            if op_def is None:
                continue

            # Convert operator inputs to Counter for count-aware checking
            # This is the key change: we now check COUNTS, not just presence
            required_types = types_from_list(op_def.inputs)

            # Check type compatibility with counts
            # Example: if op needs 2 NumCol but we only have 1, this fails
            if not is_subtype(current_types, required_types):
                continue

            # Simulate state update BEFORE checking precondition
            test_state = _clone_state(state, profile)
            op_def.update(test_state)

            # Now check precondition (may depend on updated state)
            if not op_def.precondition(profile, test_state):
                continue

            # Create OpInstance
            op_instance = OpInstance(op_name=op_name, params={})

            # Compute new types: ADD outputs to current types
            # We don't consume inputs - types accumulate through the chain
            # This allows later operators to use any previously produced type
            new_types = add_types(current_types, op_def.outputs)

            # Build new chain
            new_chain = chain + [op_instance]

            # Update state for next iteration
            next_state = _clone_state(state, profile, increment_counter=True)
            op_def.update(next_state)

            # Accept chain if it meets structural requirements
            if is_valid_chain(new_chain):
                valid_chains.append(new_chain)

            # Always continue extending chains until depth limit
            queue.append((new_chain, new_types, next_state))

    return valid_chains


def _clone_state(state: State, profile: dict, increment_counter: bool = False) -> State:
    """Clone compile-time state for grammar expansion."""
    new_state = State()
    new_state.bindings = state.bindings.copy()
    new_state.bindings["profile"] = profile
    new_state.variable_counter = state.variable_counter + (
        1 if increment_counter else 0
    )
    new_state.numeric_cols = list(state.numeric_cols)
    new_state.categorical_cols = list(state.categorical_cols)
    new_state.id_like_cols = list(state.id_like_cols)
    return new_state


def list_operators() -> List[tuple[str, str]]:
    """Get all (op_name, docstring) pairs from registry."""
    from src.datagen.synthetic.programs.operators import OPERATORS

    return [(name, "") for name in OPERATORS.keys()]


def get_operator_def(name: str):
    """Get operator definition by name."""
    return get_operator(name)


def is_valid_chain(chain: List[OpInstance]) -> bool:
    """Check if chain meets structural requirements.

    Requirements:
    - Must have >=1 selector (column selection/binding)
    - Must have >=1 analysis (computation)
    - If decision op present, must have evidence ops before it

    Why these requirements:
    - Selector ensures we're working with specific columns (not arbitrary)
    - Analysis ensures there's actual computation (not just selection)
    - Evidence-before-decision enforces proper statistical workflow
    """
    if not chain:
        return False

    selectors = sum(1 for op in chain if is_selector(op.op_name))
    analyses = sum(1 for op in chain if is_analysis(op.op_name))

    if selectors == 0:
        return False

    if analyses == 0:
        return False

    # Check decision constraint
    has_decision = any(is_decision(op.op_name) for op in chain)
    if has_decision:
        decision_idx = next(i for i, op in enumerate(chain) if is_decision(op.op_name))
        evidence_before = any(is_evidence(op.op_name) for op in chain[:decision_idx])
        if not evidence_before:
            return False

    return True


def is_selector(op_name: str) -> bool:
    """Check if operator is a selector (chooses/binds columns)."""
    op = get_operator(op_name)
    return op is not None and "selector" in op.attributes


def is_analysis(op_name: str) -> bool:
    """Check if operator performs analysis (computation)."""
    op = get_operator(op_name)
    return op is not None and "analysis" in op.attributes


def is_evidence(op_name: str) -> bool:
    """Check if operator provides evidence (for decision ops)."""
    op = get_operator(op_name)
    return op is not None and "evidence" in op.attributes


def is_decision(op_name: str) -> bool:
    """Check if operator is a decision (chooses between options)."""
    op = get_operator(op_name)
    return op is not None and "decision" in op.attributes
