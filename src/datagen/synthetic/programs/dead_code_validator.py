"""Dead code validator for compositional programs.

Policy: REJECT chains with dead code immediately.

Dead code is defined as any operator that produces a variable that is never
consumed by a subsequent operator (except for the final 'answer' variable).
"""

from src.datagen.synthetic.programs.spec import OpInstance
from src.datagen.synthetic.programs.operators import OPERATORS


def validate_no_dead_code(chain: list[OpInstance]) -> bool:
    """Validate that a chain has no dead code.

    Dead code detection algorithm:
    1. Build a dependency graph tracking what each op produces and consumes
    2. Track all variables that are produced
    3. Track all variables that are consumed (including by subsequent ops)
    4. Any produced variable that is never consumed (except 'answer') is dead code

    Special handling for 'df':
    - 'df' represents the DataFrame/table that flows through the chain
    - Chainable operators (filter, sort, etc.) produce and consume 'df' implicitly
    - We track the last op that produces 'df' and consider earlier 'df' productions
      as consumed by subsequent chainable operators

    Args:
        chain: List of OpInstance representing the program chain

    Returns:
        True if no dead code, False otherwise

    Example:
        >>> chain = [
        ...     OpInstance("select_numeric_cols", {}),
        ...     OpInstance("bind_numeric_col", {"selected_col": "col1"}),
        ...     OpInstance("mean", {}),
        ... ]
        >>> validate_no_dead_code(chain)
        True

        >>> chain_with_dead_code = [
        ...     OpInstance("select_numeric_cols", {}),
        ...     OpInstance("bind_numeric_col", {"selected_col": "col1"}),
        ...     OpInstance("zscore", {}),  # num_series never used
        ...     OpInstance("mean", {}),
        ... ]
        >>> validate_no_dead_code(chain_with_dead_code)
        False
    """
    if not chain:
        return True

    # Track all variables produced and consumed
    # produced: variable_name -> list of op indices that produced it
    produced: dict[str, list[int]] = {}
    # consumed: set of variable names that are consumed
    consumed: set[str] = set()

    for i, op_instance in enumerate(chain):
        op_name = op_instance.op_name
        op_def = OPERATORS.get(op_name)

        if op_def is None:
            # Unknown operator - skip dead code analysis for it
            continue

        # Record what this op produces
        for var in op_def.produces:
            if var not in produced:
                produced[var] = []
            produced[var].append(i)

        # Record what this op consumes
        for var in op_def.consumes:
            consumed.add(var)

    # Special handling for 'df': chainable operators flow df through the chain
    # Only the last df producer is potentially dead (if nothing after consumes it)
    if "df" in produced:
        df_producers = produced["df"]
        # All but the last df producer are consumed by subsequent chainable ops
        # The last one may or may not be dead depending on what follows
        for i in range(len(df_producers) - 1):
            consumed.add("df")
        # If there's a chainable operator after the last df producer, it's consumed
        last_df_idx = df_producers[-1]
        if last_df_idx < len(chain) - 1:
            consumed.add("df")

    # Check for dead code: any produced variable that is never consumed
    # The 'answer' variable is always considered used (it's the final output)
    for var, op_indices in produced.items():
        if var == "answer":
            continue
        if var not in consumed:
            # This variable was produced but never consumed = dead code
            return False

    return True
