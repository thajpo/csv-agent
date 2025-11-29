# Tool Design

## Why tools?

Tools constrain the action space. That's the point.

- **Verifiable**: oracle can say correct/incorrect definitively
- **Analyzable**: we can see what the model does (tool distribution, filter usage)
- **Bounded**: no arbitrary code, no I/O, no side effects

The tradeoff is flexibility. The model won't learn general Python. It learns to use *this* tool vocabulary well. For disciplined DS behavior, that's fine.

## Design rules

1. **Inputs**: `(df, params)` only. No question text, no other hooks' answers.
2. **Outputs**: one measurement (stat, metric, count). No tables, no free-form blobs.
3. **Deterministic**: stateless, seed randomness, no caches.
4. **Sandboxed**: `python_code` gets prior hook results only, not df.

## V1 tools

| Tool | What it does | Returns |
|------|--------------|---------|
| `group_stat` | filter → group → aggregate | stat + n |
| `correlation` | Pearson/Spearman between cols | r, p, n |
| `count_filter` | count rows matching filter | count |
| `model_eval` | train/test with fixed seed | metric + split sizes |
| `python_code` | compose over hook results | value |

## Anti-hacking

- Tools can't see question text or emit arbitrary payloads
- Reject degenerate hooks (empty groups, `python_code` ignoring inputs)
- Vary episode design (effect directions, null effects) so "always answer X" fails

## Future

Tool set will evolve. But offline—log traces, find gaps, add tools in batches. Not mid-training.
