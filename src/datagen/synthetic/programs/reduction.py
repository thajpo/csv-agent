"""Search-space reduction for compositional programs.

Reductions are semantics-preserving under the observation definition:
observation = final submitted answer.

Includes:
- Dead-code elimination (backward slicing)
- Commutativity canonicalization (partial-order reduction)
"""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Iterable

from src.datagen.synthetic.programs.spec import OpInstance
from src.datagen.synthetic.programs.operators import get_operator


_WILDCARD = "*"


def reduce_chains(
    chains: Iterable[list[OpInstance]],
    *,
    min_length: int = 3,
    observation: Iterable[str] = ("answer",),
) -> list[list[OpInstance]]:
    """Reduce operator chains with DCE + canonicalization.

    Args:
        chains: Raw operator chains from grammar search.
        min_length: Minimum chain length after DCE.
        observation: Observable outputs (default: final answer).

    Returns:
        Deduplicated list of reduced chains.
    """
    reduced: list[list[OpInstance]] = []
    seen: set[tuple] = set()
    observation_set = set(observation)

    for chain in chains:
        pruned = dead_code_eliminate(chain, observation_set)
        if len(pruned) < min_length:
            continue

        canonical = canonicalize_chain(pruned)
        signature = chain_signature(canonical)
        if signature in seen:
            continue

        seen.add(signature)
        reduced.append(canonical)

    return reduced


def dead_code_eliminate(
    chain: list[OpInstance], observation: set[str]
) -> list[OpInstance]:
    """Remove ops that do not affect the observed answer.

    Backward slice from the observation set (default: {"answer"}).
    """
    needed = set(observation)
    kept_reversed: list[OpInstance] = []

    for op in reversed(chain):
        reads, writes = _get_effects(op)

        if _WILDCARD in needed or _WILDCARD in writes:
            kept_reversed.append(op)
            needed = {_WILDCARD}
            continue

        if writes & needed:
            kept_reversed.append(op)
            needed = (needed - writes) | reads

    kept_reversed.reverse()
    return kept_reversed


def canonicalize_chain(chain: list[OpInstance]) -> list[OpInstance]:
    """Canonicalize chain by reordering independent ops.

    Uses a dependency DAG built from read/write conflicts, then
    returns the lexicographically smallest topological ordering
    using a stable operator key.
    """
    if len(chain) <= 1:
        return list(chain)

    reads_list: list[set[str]] = []
    writes_list: list[set[str]] = []
    for op in chain:
        reads, writes = _get_effects(op)
        reads_list.append(reads)
        writes_list.append(writes)

    n = len(chain)
    edges: list[list[int]] = [[] for _ in range(n)]
    indegree = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            if _dependent(reads_list[i], writes_list[i], reads_list[j], writes_list[j]):
                edges[i].append(j)
                indegree[j] += 1

    heap: list[tuple[tuple, int]] = []
    for i in range(n):
        if indegree[i] == 0:
            heappush(heap, (_op_sort_key(chain[i]), i))

    ordered: list[OpInstance] = []
    while heap:
        _, idx = heappop(heap)
        ordered.append(chain[idx])
        for nxt in edges[idx]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heappush(heap, (_op_sort_key(chain[nxt]), nxt))

    return ordered


def chain_signature(chain: list[OpInstance]) -> tuple:
    """Stable signature for deduplication."""
    return tuple(_op_signature(op) for op in chain)


def _op_signature(op: OpInstance) -> tuple:
    return (op.op_name, tuple(sorted(op.params.items())))


def _dependent(
    reads_a: set[str], writes_a: set[str], reads_b: set[str], writes_b: set[str]
) -> bool:
    if _WILDCARD in reads_a or _WILDCARD in writes_a:
        return True
    if _WILDCARD in reads_b or _WILDCARD in writes_b:
        return True
    if writes_a & (reads_b | writes_b):
        return True
    if writes_b & (reads_a | writes_a):
        return True
    return False


def _op_sort_key(op: OpInstance) -> tuple:
    op_def = get_operator(op.op_name)
    stage = _op_stage(op_def)
    params = tuple(sorted(op.params.items()))
    return (stage, op.op_name, params)


def _op_stage(op_def) -> int:
    if op_def is None:
        return 99
    attrs = op_def.attributes or []
    if "selector" in attrs:
        return 0
    if "transform" in attrs:
        return 1
    if "evidence" in attrs:
        return 2
    if "decision" in attrs:
        return 3
    if "analysis" in attrs:
        return 4
    if "test" in attrs:
        return 5
    return 6


def _get_effects(op: OpInstance) -> tuple[set[str], set[str]]:
    op_def = get_operator(op.op_name)
    if op_def is None:
        return {_WILDCARD}, {_WILDCARD}
    return set(op_def.reads), set(op_def.writes)
