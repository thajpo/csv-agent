Experimental and ad-hoc analysis scripts.

These are intentionally not part of the automated `tests/` suite.
Run them manually when needed.

- `investigate_failures.py`: ad-hoc failure diagnostics for triangulation output.
- `prepare_value_episodes.py`: makes CSV-disjoint episode files from a pinned
  Hugging Face snapshot and locally downloaded Kaggle inputs.
- `collect_prefix_values.py`: samples distinct first actions and estimates each
  saved partial attempt from terminally verified continuations.
- `train_value_model.py`: trains the local text value model and compares it with
  average-success and simple-execution baselines.
- `evaluate_value_selection.py`: compares random and value-guided candidate
  selection on one reserved continuation per candidate.

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
  --model qwen/qwen3-30b-a3b-instruct-2507 \
  --dataset-revision <immutable-hugging-face-commit> \
  --output artifacts/prefix-values.jsonl
```

Before starting, the command reports one source rollout plus the requested
continuation rollouts and their worst-case provider-request count, including
retries. The defaults select the boundary after one completed turn, sample four
continuations, and process one episode. `--turn-count 0` selects the initial
state; every selected boundary must leave at least one of `--max-turns`
available. The hard caps are 16 continuations, 8 candidate actions per episode,
64 episodes, 20 turns, 16 concurrent rollouts, and 10,000 worst-case provider
requests. The request bound includes source resampling, continuation turns, and
three API attempts per request. Each completed record is appended immediately
so an interrupted batch retains its finished evidence. The base seed is offset
for the source rollout and each continuation. The provider receives the seed
when its OpenAI-compatible API supports that field.

Each output line is a validated `PrefixValueRecord`. It includes the exact
public prefix, actor policy, continuation seeds, continuation traces or errors,
terminal verdicts, current Git commit, and supplied dataset revision. The value
is successful continuations divided by all attempts only when every attempt is
labeled. Missing verifier-hash provenance and replay, rollout, or verifier
errors make the estimate unavailable while remaining auditable. Hooks may
appear in traces as diagnostics, but they neither reject a submission nor
define a value label.

## Train and test selection

Reserve the final continuation from every record, fit on the others, and then
use that unseen outcome for the equal-call selection comparison:

```bash
uv run python scripts/experiments/train_value_model.py \
  --train data/experiments/value-canary/train-values.jsonl \
  --validation data/experiments/value-canary/validation-values.jsonl \
  --test data/experiments/value-canary/test-values.jsonl \
  --output-dir data/experiments/value-canary/model

uv run python scripts/experiments/evaluate_value_selection.py \
  --test data/experiments/value-canary/test-values.jsonl \
  --checkpoint data/experiments/value-canary/model/value_model.joblib \
  --output data/experiments/value-canary/model/selection.json
```

The trainer refuses overlapping CSV datasets. Its input renderer uses the
system prompt, visible conversation, tool feedback, and turns left; it never
reads expected answers, continuation traces, or verifier outcomes as features.
