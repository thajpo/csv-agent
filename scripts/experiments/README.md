Experimental and ad-hoc analysis scripts.

These are intentionally not part of the automated `tests/` suite.
Run them manually when needed.

- `investigate_failures.py`: ad-hoc failure diagnostics for triangulation output.
- `collect_prefix_values.py`: bounded real-model canary that replays one selected
  turn boundary and estimates its value from terminally verified continuations.
  It defaults to one episode and four continuations; pass the immutable
  Hugging Face dataset revision used for the input snapshot.

## Prefix-value canary

The canary requires an episode JSONL file whose questions contain a nonempty
`ground_truth_hash` or `ground_truth_hashes`, access to each episode's local CSV
(or one `--csv` override), the normal sandbox runtime, an
`OPENROUTER_API_KEY`, and a clean Git worktree. The clean-worktree check ensures
that the recorded commit identifies the exact code used for collection. Run it
from the repository root:

```bash
uv run python scripts/experiments/collect_prefix_values.py \
  --episodes data/episodes/template.jsonl \
  --model openai/gpt-oss-120b \
  --dataset-revision <immutable-hugging-face-commit> \
  --output artifacts/prefix-values.jsonl
```

Before starting, the command reports one source rollout plus the requested
continuation rollouts and their worst-case provider-request count, including
retries. The defaults select the boundary after one completed turn, sample four
continuations, and process one episode. `--turn-count 0` selects the initial
state; every selected boundary must leave at least one of `--max-turns`
available. The hard caps are 16 continuations, 8 episodes, 20 turns, and 300
worst-case provider requests, with the last bound calculated as episodes times
source-plus-continuation rollouts times turns times 3 API attempts. The base
seed is offset for the source rollout and each continuation. The provider
receives the seed when its OpenAI-compatible API supports that field.

Each output line is a validated `PrefixValueRecord`. It includes the exact
public prefix, actor policy, continuation seeds, continuation traces or errors,
terminal verdicts, current Git commit, and supplied dataset revision. The value
is successful continuations divided by all attempts only when every attempt is
labeled. Missing verifier-hash provenance and replay, rollout, or verifier
errors make the estimate unavailable while remaining auditable. Hooks may
appear in traces as diagnostics, but they neither reject a submission nor
define a value label.
