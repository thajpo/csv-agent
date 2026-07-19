Experimental and ad-hoc analysis scripts.

These are intentionally not part of the automated `tests/` suite.
Run them manually when needed.

- `investigate_failures.py`: ad-hoc failure diagnostics for triangulation output.
- `collect_prefix_values.py`: bounded real-model canary that replays one selected
  turn boundary and estimates its value from terminally verified continuations.
  It defaults to one episode and four continuations; pass the immutable
  Hugging Face dataset revision used for the input snapshot.

## Prefix-value canary

The canary requires an episode JSONL file, access to each episode's local CSV
(or one `--csv` override), the normal sandbox runtime, and an
`OPENROUTER_API_KEY`. Run it from the repository root:

```bash
uv run python scripts/experiments/collect_prefix_values.py \
  --episodes data/episodes/template.jsonl \
  --model openai/gpt-oss-120b \
  --dataset-revision <immutable-hugging-face-commit> \
  --output artifacts/prefix-values.jsonl
```

Before starting, the command reports one source rollout plus the requested
continuation rollouts per episode as its planned call count. Each rollout can
make multiple turn-level provider requests. The defaults select the boundary
after one completed turn, sample four continuations, and process one episode.
`--turn-count 0` selects the initial state; every selected boundary must leave
at least one of `--max-turns` available. Continuations are capped at 16;
`--max-episodes` has no hard cap, so keep its default of one for the intended
canary. The base seed is offset for the source rollout and each continuation.
The provider receives the seed when its OpenAI-compatible API supports that
field.

Each output line is a validated `PrefixValueRecord`. It includes the exact
public prefix, actor policy, seeds, continuation traces or errors, terminal
verdicts, current Git commit, and supplied dataset revision. The value is
successful continuations divided by all attempts, so replay, rollout, and
verifier errors count as unsuccessful attempts while remaining visibly
unlabeled. Hooks may appear in traces as diagnostics, but they neither reject a
submission nor define a value label.
