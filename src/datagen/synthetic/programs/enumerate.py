"""Column enumeration for compositional programs (no arbitrary selection).

KEY DESIGN DECISION: General binding enumeration
-------------------------------------------------
Instead of hardcoded bindings (selected_col, cat_col), we enumerate ALL
required bindings for each operator chain. This supports multi-input operators
like correlation that need two different columns.

How enumeration works:
1. Inspect the operator chain to find all required binding keys
2. For each binding key, determine which column pool to use
3. Generate the cartesian product of all valid combinations
4. Create a ProgramSpec for each combination

Example for correlation chain:
    Chain: [select_numeric_cols, bind_num_col_1, bind_num_col_2, correlation]
    Required bindings: {"num_col_1", "num_col_2"}
    Column pool: numeric_cols (from profile)
    Enumerations: all ordered pairs (col1, col2) where col1 != col2

Why this design:
- Supports any number of binding keys per chain
- No arbitrary selection (all valid combos enumerated)
- Extensible: adding a new binding type only requires adding its pool
- Clear separation: grammar finds valid chains, enumeration fills bindings

Previous limitation:
Only supported selected_col and cat_col. Couldn't handle correlation
or other multi-input operators without hacks.
"""

from typing import List, Dict, Any, Set
from src.datagen.synthetic.programs.spec import ProgramSpec, OpInstance
from src.datagen.synthetic.programs.operators import (
    get_eligible_numeric_cols,
    get_eligible_categorical_cols,
    get_eligible_binary_categorical_cols,
    get_operator,
)


def enumerate_bindings(
    chains: List[List[OpInstance]], profile: Dict[str, Any]
) -> List[ProgramSpec]:
    """Enumerate all valid column bindings for each operator chain.

    This is where we avoid "first column" selection - we generate
    a separate ProgramSpec for each eligible combination of bindings.

    Args:
        chains: List of operator chains from grammar search
        profile: Dataset profile

    Returns:
        List of ProgramSpecs with specific bindings filled in

    Example:
        Input chain: [select_numeric_cols, bind_num_col_1, mean]
        Output: One ProgramSpec per numeric column, each with
                params={"num_col_1": "column_name"}
    """
    # Build eligible column pools from profile
    numeric_cols = get_eligible_numeric_cols(profile)
    categorical_cols = get_eligible_categorical_cols(profile)
    binary_cat_cols = get_eligible_binary_categorical_cols(profile)

    programs = []

    for chain_idx, chain in enumerate(chains):
        # Discover all binding requirements in this chain
        binding_keys = _discover_binding_keys(chain)

        # Skip chains with no bindings (shouldn't happen for valid chains)
        if not binding_keys:
            programs.append(_create_program_spec(chain, {}, chain_idx))
            continue

        # Generate all valid binding combinations
        binding_combinations = _generate_binding_combinations(
            binding_keys, numeric_cols, categorical_cols, binary_cat_cols
        )

        # Create a ProgramSpec for each combination
        for bindings in binding_combinations:
            programs.append(_create_program_spec(chain, bindings, chain_idx))

    return programs


def _discover_binding_keys(chain: List[OpInstance]) -> Set[str]:
    """Find all binding keys required by operators in the chain.

    Each operator declares its required bindings in requires_bindings.
    We collect all unique keys across the chain.

    Args:
        chain: Operator chain

    Returns:
        Set of binding key names

    Example:
        Chain with bind_num_col_1 and bind_num_col_2
        Returns: {"num_col_1", "num_col_2"}
    """
    keys = set()
    for op_instance in chain:
        op = get_operator(op_instance.op_name)
        if op is not None:
            for key, required in op.requires_bindings.items():
                if required:
                    keys.add(key)
    return keys


def _generate_binding_combinations(
    binding_keys: Set[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    binary_cat_cols: List[str],
) -> List[Dict[str, str]]:
    """Generate all valid binding combinations.

    Maps each binding key to its appropriate column pool and generates
    the cartesian product, excluding invalid combinations (e.g., same
    column bound to two different keys).

    Args:
        binding_keys: Set of required binding keys
        numeric_cols: Available numeric columns
        categorical_cols: Available categorical columns
        binary_cat_cols: Available binary categorical columns

    Returns:
        List of binding dicts, each mapping keys to column names

    Example:
        binding_keys = {"num_col_1", "num_col_2"}
        numeric_cols = ["A", "B"]
        Returns: [
            {"num_col_1": "A", "num_col_2": "B"},
            {"num_col_1": "B", "num_col_2": "A"}
        ]
    """
    if not binding_keys:
        return [{}]

    # Map each binding key to its column pool
    # This is extensible: add new patterns here for new binding types
    key_pools = {}
    for key in binding_keys:
        if "num_col" in key or "numeric" in key:
            key_pools[key] = numeric_cols
        elif "binary_cat" in key or key == "cat_col":
            key_pools[key] = binary_cat_cols
        elif "cat" in key:
            key_pools[key] = categorical_cols
        else:
            # Default to numeric for unknown keys
            key_pools[key] = numeric_cols

    # Generate cartesian product
    import itertools

    keys = list(binding_keys)
    pools = [key_pools[key] for key in keys]

    combinations = []
    for values in itertools.product(*pools):
        # Skip if any two keys got the same column (unless allowed)
        # This prevents binding the same column to num_col_1 and num_col_2
        if len(set(values)) == len(values):
            combinations.append(dict(zip(keys, values)))

    return combinations


def _create_program_spec(
    chain: List[OpInstance], bindings: Dict[str, str], chain_idx: int
) -> ProgramSpec:
    """Create ProgramSpec with specific bindings.

    Args:
        chain: Operator chain
        bindings: Binding dict mapping keys to column names
        chain_idx: Index for naming

    Returns:
        ProgramSpec with bindings filled in
    """
    # Build name from chain index and bound columns
    name_parts = [f"program_{chain_idx}"]
    for key, value in sorted(bindings.items()):
        # Sanitize column name for use in program name
        safe_name = value.replace(" ", "_").replace("-", "_")[:20]
        name_parts.append(f"{key}_{safe_name}")
    name = "_".join(name_parts)

    return ProgramSpec(
        name=name,
        ops=[_bind_instance(op, bindings) for op in chain],
        output_type="dict",
        output_schema="Generated by compositional generator",
        difficulty="EASY",
        tags=["generated"],
    )


def _bind_instance(instance: OpInstance, bindings: Dict[str, str]) -> OpInstance:
    """Bind column params to OpInstance.

    Only binds keys that the operator explicitly requires.
    Other bindings are ignored (they're for other operators in the chain).

    Args:
        instance: OpInstance to bind
        bindings: Available bindings

    Returns:
        New OpInstance with relevant bindings filled
    """
    params = instance.params.copy()

    op = get_operator(instance.op_name)
    if op is not None:
        for key in op.requires_bindings:
            if key in bindings:
                params[key] = bindings[key]

    return OpInstance(op_name=instance.op_name, params=params)
