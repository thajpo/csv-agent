# Search Space Reduction (Programs)

This document explains how we reduce the compositional program search space while
preserving semantic diversity under a clear observation definition.

## Observation Definition

**Observation = the final value submitted via `submit(...)`.**

This means we preserve semantic diversity with respect to the final answer only.
Hooks and intermediate values are not part of the observation for pruning.

## Why Reduction Is Needed

Grammar search produces many operator chains that are equivalent up to:

- reordering of independent operators
- extra operations that do not affect the final submitted answer

Reducing these redundancies keeps program diversity while lowering cost.

## Reductions Implemented

### 1) Dead-Code Elimination (Backward Slicing)

**Idea:** Drop any operator that does not contribute to the final submitted answer.

Algorithm:

- Start with `needed = {"answer"}`
- Traverse the chain backward
- Keep an op if it writes something in `needed`
- Update `needed = (needed - writes(op)) ∪ reads(op)`

This removes steps that do not affect the final output.

### 2) Commutativity Canonicalization

**Idea:** If two operators are independent, their order does not change the
final answer. We keep only a canonical ordering.

Two ops are independent when their read/write sets do not conflict:

```
writes(A) & (reads(B) | writes(B)) = empty
writes(B) & (reads(A) | writes(A)) = empty
```

We build a dependency DAG from conflicts and return the lexicographically
smallest topological ordering. This deduplicates permutations without losing
semantic diversity.

### 3) Minimum Chain Length

After dead-code elimination, we drop chains shorter than 3 operators. This
avoids "nothing burger" programs while still allowing simple, valid programs
like `select -> pick -> mean`.

## Required Metadata for Operators

Each operator must define its abstract **reads** and **writes** sets.
These sets are used for both DCE and commutativity.

Example:

```
pick_numeric_by_variance:
  reads:  {"numeric_cols"}
  writes: {"selected_col"}

mean:
  reads:  {"selected_col"}
  writes: {"answer"}
```

If new operators are added, their read/write sets must be filled in. If they
are missing, the reduction becomes conservative and will keep more chains.

## Integration Point

The reduction runs in `sample_programs()` after grammar search and before
column enumeration:

```
chains = search_programs(profile, max_depth=6)
chains = reduce_chains(chains, min_length=3, observation=("answer",))
programs = enumerate_bindings(chains, profile)
```

## Scope Notes

- Reductions are semantics-preserving for the final submitted answer only.
- Hooks are not treated as observable for pruning.
- If we later decide hooks are part of observation, DCE must be restricted.
