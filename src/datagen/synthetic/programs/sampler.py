"""Sampler for compositional program generation (Option B).

Pipeline:
1) Grammar search (typed BFS/DFS, depth 6)
2) Enumerate all eligible column bindings
3) Return ProgramSpecs (no arbitrary selection)
"""

from typing import List, Dict, Any

from src.datagen.synthetic.programs.spec import ProgramSpec
from src.datagen.synthetic.programs.grammar import search_programs
from src.datagen.synthetic.programs.enumerate import enumerate_bindings
from src.datagen.synthetic.programs.reduction import reduce_chains


def sample_programs(profile: Dict[str, Any]) -> List[ProgramSpec]:
    """Generate programs via grammar search + enumeration.

    This is true compositional generation. No hardcoded program catalogs.
    """
    chains = search_programs(profile, max_depth=6)
    chains = reduce_chains(chains, min_length=3, observation=("answer",))
    programs = enumerate_bindings(chains, profile)
    return programs
