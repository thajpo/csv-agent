"""Sampler for compositional program generation (Option B).

Pipeline:
1) Grammar search (typed BFS/DFS, depth 6)
2) Dead code validation (reject chains with dead code)
3) Enumerate all eligible column bindings
4) Return ProgramSpecs (no arbitrary selection)
"""

from typing import List, Dict, Any

from src.datagen.synthetic.programs.spec import ProgramSpec
from src.datagen.synthetic.programs.grammar import search_programs
from src.datagen.synthetic.programs.enumerate import enumerate_bindings
from src.datagen.synthetic.programs.reduction import reduce_chains
from src.datagen.synthetic.programs.semantic_long_chains import (
    generate_semantic_long_programs,
)
from src.datagen.synthetic.programs.dead_code_validator import validate_no_dead_code


def sample_programs(
    profile: Dict[str, Any], target_length: int = 10
) -> List[ProgramSpec]:
    """Generate programs via grammar search + enumeration.

    This is true compositional generation. No hardcoded program catalogs.
    Dead code is rejected immediately.

    Args:
        profile: Dataset profile
        target_length: Target chain length (10-15 for complexity)
    """
    chains = search_programs(profile, max_depth=15)

    # For long chains, skip reduction to preserve complexity
    # Only reduce chains shorter than target to remove true dead code
    short_chains = [c for c in chains if len(c) < target_length]
    long_chains = [c for c in chains if len(c) >= target_length]

    # Reduce short chains only
    reduced_short = reduce_chains(
        short_chains, min_length=3, observation=("answer", "chosen_test")
    )

    # Keep long chains as-is (they're already complex)
    # Just deduplicate them
    seen = set()
    unique_long = []
    for chain in long_chains:
        sig = tuple(op.op_name for op in chain)
        if sig not in seen:
            seen.add(sig)
            unique_long.append(chain)

    # Combine: prefer long chains, fill with short if needed
    all_chains = unique_long + reduced_short

    # Validate chains for dead code - reject any with dead code
    valid_chains = [c for c in all_chains if validate_no_dead_code(c)]

    programs = enumerate_bindings(valid_chains, profile)

    # Add semantic long-chain templates (10-15 steps)
    # These are meaningful workflows where every step affects the answer
    semantic_long_programs = generate_semantic_long_programs(profile)

    # Validate semantic programs too - check the ops chain inside each ProgramSpec
    valid_semantic = [p for p in semantic_long_programs if validate_no_dead_code(p.ops)]

    # Combine: semantic long chains first, then discovered chains
    return valid_semantic + programs
