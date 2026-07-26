# csv-agent

`csv-agent` is a research repository for execution-grounded CSV agents. It can
generate verified CSV-analysis episodes, run models against those episodes in a
Docker sandbox, and test whether a learned value model can choose promising
partial attempts.

The current value-function result is **negative/inconclusive**, not a claimed
improvement. The useful output is a reproducible experimental path and a clear
measurement problem to solve next.

## Current result

| Actor | Expected random selection | Value-guided selection | Conclusion |
| --- | ---: | ---: | --- |
| Qwen canary | 11/16 | 10/16 | No benefit observed |
| DeepSeek V4 Flash replication | 70.31% | 73.44% | +3.12 pp; 95% CI [-6.25, 11.98] |

The DeepSeek improvement was below the preregistered five-point threshold, its
interval crossed zero, and only two of four test datasets improved. More
importantly, an audit found that many rejected model answers were reasonable
equivalents or depended on conventions hidden from the prompt. The present
critic therefore predicts the existing verifier contract, not clean semantic
correctness. See [research.md](research.md) for the short handoff and
[the full report](docs/research/value-canary-2026-07-20.md) for the evidence.

## Start from a fresh clone

Prerequisites:

- Git
- [uv](https://docs.astral.sh/uv/) for Python and dependency management
- Docker, running locally, for model-executed Python code
- an OpenRouter API key only for live model calls

```bash
git clone https://github.com/thajpo/csv-agent.git
cd csv-agent
uv python install 3.13
uv sync --dev
uv run csvagent --help
```

Run the same local checks as CI. They make no paid model calls:

```bash
uv run --with ruff ruff check .
uv run pytest -q
```

The default test configuration excludes tests marked `live`. Run a focused
test while iterating with `uv run pytest -q tests/test_value_trainer.py`.

## Make one OpenRouter-backed agent run

The checked-in smoke episode asks the model to inspect a checked-in CSV and
submit its row count. It is intentionally tiny, but it exercises the real
OpenRouter client, conversation loop, Docker sandbox, Python execution, final
submission, verifier, and report writer.

```bash
export OPENROUTER_API_KEY="your-key"
docker info >/dev/null

uv run python scripts/evaluate_model.py \
  --model deepseek/deepseek-v4-flash \
  --episodes data/fixtures/openrouter-smoke.jsonl \
  --output /tmp/csv-agent-openrouter-smoke.md \
  --max-turns 3 \
  --temperature 0 \
  --concurrency 1
```

The first run builds the `csv-analysis-env` Docker image and may take a few
minutes. The command evaluates one episode, so the provider receives at most
three actor turns (plus automatic retries on transient failures). A correct run
reports `Accuracy: 100.0% (1/1)`; a model failure is still a valid smoke result
if the report is written and the failure is recorded.

Evaluation episodes must use the canonical `question.ground_truth` value and
provide `question.ground_truth_hash` or a non-empty
`question.ground_truth_hashes` list. The evaluator checks the submitted answer
against every accepted hash, with the existing tolerant comparison as a
fallback, and rejects missing hash provenance before starting a rollout.

Use any OpenRouter model slug with `--model`. Keep the model ID, sampling
settings, dataset revision, and Git commit in experiment notes; changing any of
them changes the policy being evaluated.

## Reproduce the frozen value-selector result

The DeepSeek prefix records are stored in a private, immutable Hugging Face
snapshot. With access to `ThaJpo/csv-agent-prefix-values-deepseek-canary`:

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

These commands do not call OpenRouter; they retrain and evaluate the local
scikit-learn selectors from already collected records. Exact collection and
interpretation details live in
[scripts/experiments/README.md](scripts/experiments/README.md).

## Repository map

- `src/core/`: model client, prompts, conversation loop, and environment
  orchestration.
- `src/envs/`: Docker-backed Python execution for CSV analysis.
- `src/datagen/`: question, trace, and terminal-verification pipelines.
- `src/value/`: prefix construction, replay, continuation collection, and value
  training inputs.
- `packages/csv-spec/`: shared episode and prefix-value contracts.
- `scripts/experiments/`: bounded research commands kept outside the main CLI.
- `configs/`: pinned dataset and experiment inputs.
- `data/fixtures/`: small deterministic inputs tracked by Git. Generated and
  downloaded data under `data/` is ignored.

See [docs/architecture.md](docs/architecture.md) for the data flow.

## Data and credentials

Generated datasets belong on Hugging Face at an immutable commit revision;
source Kaggle datasets are restored through the Kaggle API. Git tracks only
small deterministic fixtures under `data/fixtures/`. Never commit provider,
Hugging Face, or Kaggle credentials.

The frozen dataset references used by current research are:

- source episodes: `ThaJpo/csv-agent-template-episodes` at
  `e19fadf8d713c5afb7fe1476e2160b9bece1233a`
- DeepSeek prefix values: `ThaJpo/csv-agent-prefix-values-deepseek-canary` at
  `890eb10b775a224035807d9a29db3e52743d1c18`

## Generation CLI

The older data-generation path remains available, but it is not the current
research priority:

```bash
uv run csvagent generate questions --template --dry-run
uv run csvagent generate episodes --template --dry-run
uv run csvagent run --template --test --dry-run
```

Run `uv run csvagent <command> --help` before a paid or large operation. New
value research should not resume until verifier labels are valid enough to
distinguish mathematically equivalent answers from real failures.
