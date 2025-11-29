# Ongoing

## Where we are
- Bootstrap EDA + teacher exploration loop work.
- Docs aligned to multi-hook design.
- Oracle and tools: not yet implemented.

## V1 scope
- 2-4 hooks per episode, DAG with explicit deps
- Tools: `group_stat`, `correlation`, `count_filter`, `model_eval`, `python_code`
- `python_code` operates on hook results only, not df
- Oracle = ground truth, ~5% float tolerance, partial credit per hook

## Deferred
- Hidden eval hooks, corruption variants
- Behavioral regularizers, coverage metrics
- Tool evolution pipeline

## Next
1. `HookSpec`, `OracleResult`, `Episode` dataclasses
2. Built-in tools (group_stat, correlation, count_filter, model_eval w/ seeds)
3. `run_hooks` with topo sort + validation
4. Teacher prompt update + JSON parsing
5. Hand-author a few episodes to sanity-check oracle
