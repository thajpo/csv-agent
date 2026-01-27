"""Grammar search for compositional program generation (Option B)."""

from typing import List, Set
from src.datagen.synthetic.programs.types import Types, TYPE_TABLE, is_subtype
from src.datagen.synthetic.programs.operators import get_operator, OPERATORS
from src.datagen.synthetic.programs.spec import State, OpInstance


def list_operator_names() -> List[str]:
    """Get all operator names."""
    return list(OPERATORS.keys())


def search_programs(profile: dict, max_depth: int = 6) -> List[List[OpInstance]]:
    """
    Search for valid operator chains using BFS from Table state.
    """
    # Start with empty chain and Table type
    queue: List[tuple[List[OpInstance], Types, State]] = [
        ([], Types([TYPE_TABLE]), State())
    ]
    valid_chains: List[List[OpInstance]] = []

    iteration = 0

    while queue and iteration < 2000:  # Safety limit
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

            required_types = Types(op_def.inputs)

            # Check type compatibility
            if not is_subtype(current_types, required_types):
                continue

            # Simulate state update BEFORE checking precondition
            test_state = _clone_state(state, profile)

            op_def.update(test_state)

            # Now check precondition
            if not op_def.precondition(profile, test_state):
                continue

            # Create OpInstance
            op_instance = OpInstance(op_name=op_name, params={})

            # Compute new types (add outputs, do not consume inputs)
            new_type_set = set(current_types)
            new_type_set.update(op_def.outputs)
            new_types = Types(new_type_set)

            # Build new chain
            new_chain = chain + [op_instance]

            # Update state for next iteration
            next_state = _clone_state(state, profile, increment_counter=True)
            op_def.update(next_state)

            # Accept any chain that has at least one selector or analysis
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
    """
    Check if chain meets structural requirements:
    - Must have >=1 selector
    - Must have >=1 analysis
    - If decision op present, must have evidence ops before it
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
    op = get_operator(op_name)
    return op is not None and "selector" in op.attributes


def is_analysis(op_name: str) -> bool:
    op = get_operator(op_name)
    return op is not None and "analysis" in op.attributes


def is_evidence(op_name: str) -> bool:
    op = get_operator(op_name)
    return op is not None and "evidence" in op.attributes


def is_decision(op_name: str) -> bool:
    op = get_operator(op_name)
    return op is not None and "decision" in op.attributes
