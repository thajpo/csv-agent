# Research experiments

These scripts are manual research entry points, not stable `csvagent`
subcommands. Start with the repository-level [README](../../README.md) and keep
generated artifacts under the ignored `data/experiments/` path.

## What the value canary tests

For each CSV question, an OpenRouter actor samples several distinct first
actions. Each resulting nonterminal state is replayed in a fresh Docker
sandbox, then continued multiple times. The terminal verifier labels those
continuations. A local value model is trained to rank the first actions and is
compared with expected random selection under the same deployed actor-call
budget.

This is a test of bounded action selection, not evidence that the agent is
self-improving. Current labels also inherit known semantic problems from the
terminal verifier.

## Reproduce the frozen DeepSeek result

Requirements: the normal `uv sync --dev` setup, Hugging Face authentication,
and access to the private dataset named in
`configs/value/deepseek-canary.toml`. No OpenRouter calls are made during this
reproduction.

```bash
uv run hf auth login

uv run python scripts/experiments/download_value_snapshot.py \
  --dataset-config configs/value/deepseek-canary.toml \
  --output-dir data/experiments/deepseek-canary

uv run python scripts/experiments/train_value_model.py \
  --train data/experiments/deepseek-canary/train-values.jsonl \
  --validation data/experiments/deepseek-canary/validation-values.jsonl \
  --output-dir data/experiments/deepseek-canary/model

uv run python scripts/experiments/evaluate_value_selection.py \
  --test data/experiments/deepseek-canary/test-values.jsonl \
  --model-dir data/experiments/deepseek-canary/model \
  --output data/experiments/deepseek-canary/model/test-selection.json
```

The pinned snapshot contains 192 train, 48 validation, and 96 test candidate
records. The expected primary result is:

```text
expected random success: 70.31%
pairwise-guided success: 73.44%
difference:              +3.12 percentage points
95% hierarchical CI:     [-6.25, 11.98]
positive datasets:       2/4
```

The trainer saves three models and a `model-freeze.json` containing input and
checkpoint hashes. The evaluator verifies that freeze before opening the
sealed test records. It uses the two reserved continuations per candidate;
the six training continuations are never reused as the selection outcome.

## Collect a new bounded OpenRouter dataset

Do this only after defining a new hypothesis and resolving the verifier-label
problem described in [research.md](../../research.md). Collection costs money,
requires Docker, and deliberately refuses a dirty Git worktree so every record
identifies the code that produced it.

Inputs must be JSONL episodes with terminal-verifier hashes and locally
available CSVs. The command below is illustrative; choose the episode snapshot,
immutable revision, request cap, and output path before running it.

```bash
export OPENROUTER_API_KEY="your-key"
docker info >/dev/null

uv run python scripts/experiments/collect_prefix_values.py \
  --episodes data/experiments/tasks/train.jsonl \
  --model deepseek/deepseek-v4-flash \
  --dataset-revision <immutable-source-snapshot-commit> \
  --output data/experiments/new-canary/train-values.jsonl \
  --max-episodes 1 \
  --candidates-per-episode 3 \
  --continuations 2 \
  --max-turns 3 \
  --concurrency 1 \
  --max-provider-requests 100
```

Before making a request, the collector prints and validates its worst-case
provider-call bound. It appends each completed record immediately and supports
`--resume`. Hard caps prevent accidentally unbounded collection. Run
`uv run python scripts/experiments/collect_prefix_values.py --help` for every
contract field and cap.

Each `PrefixValueRecord` preserves the public state shown to the critic, actor
policy, continuation seeds and outcomes, Git commit, source dataset revision,
and collection contract. Expected answers remain private verifier inputs.
Hooks are diagnostic only and do not supply value labels.

## Other scripts

- `prepare_value_episodes.py`: create CSV-disjoint episode splits from a pinned
  source snapshot and locally restored Kaggle inputs.
- `investigate_failures.py`: ad hoc triangulation diagnostics.
- `test_pass_rates*.py` and `test_verbalization*.py`: historical manual probes;
  they are not part of pytest or the frozen value result.
