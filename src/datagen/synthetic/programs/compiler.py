"""Compiles ProgramSpec into executable code with hooks."""

from typing import Any

from src.datagen.synthetic.programs.spec import ProgramSpec, State
from src.datagen.synthetic.programs.operators import get_operator


def compile_program(spec: ProgramSpec, profile: dict) -> str:
    """Compile a ProgramSpec into executable code.

    Rules:
    - Use deterministic variable naming (v0, v1, v2, ...)
    - Exclude ID-like columns using profile-derived list
    - Apply deterministic tie-breaks (sort col names; pick first on ties)
    - Emit hook() for evidence + decision values
    - Final line must call submit(answer) with required schema
    """
    state = State()
    state.bindings["profile"] = profile

    # Build code with preamble
    code = _preamble(profile, state)

    # Emit each operator
    for instance in spec.ops:
        op = get_operator(instance.op_name)
        if op is None:
            raise ValueError(f"Unknown operator: {instance.op_name}")

        # Update params into state
        for k, v in instance.params.items():
            state.bindings[k] = v

        # Check precondition
        if not op.precondition(profile, state):
            raise ValueError(f"Precondition failed for {op.name}")

        # Update state BEFORE emit (emit needs updated state)
        op.update(state)

        # Emit code
        code += op.emit(state) + "\n"

        # Increment counter
        state.variable_counter += 1

    return code


def _preamble(profile: dict, state: State) -> str:
    """Generate code preamble with helper functions.

    df is already available in the sandbox environment.
    """
    id_cols = _get_id_like_columns(profile)
    return f"""# Helper to exclude ID-like columns
_ID_LIKE_COLS = {id_cols}

def _is_id_like(col_name):
    return col_name in _ID_LIKE_COLS or col_name in _ID_LIKE_COLS

# Load numeric columns (excluding IDs)
numeric_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64'] and not _is_id_like(c)]
categorical_cols = [c for c in df.columns if df[c].dtype in ['object', 'category'] and not _is_id_like(c)]

"""


def _get_id_like_columns(profile: dict) -> list[str]:
    """Extract ID-like columns from profile."""
    id_patterns = [
        r"^unnamed:\s*\d+$",
        r"^index$",
        r"^id$",
        r"^_id$",
        r"^row_?id$",
        r"^person_?id$",
    ]
    import re

    id_cols = [
        col
        for col in profile.get("columns", {}).keys()
        if any(re.match(p, col.lower(), re.IGNORECASE) for p in id_patterns)
        or col.lower().startswith("unnamed:")
    ]
    return id_cols
