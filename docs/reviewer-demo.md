# Reviewer Demo

This path is meant to prove the repo shape without a paid model, GPU, or large
dataset generation run.

## Local Checks

Install dependencies:

```bash
uv sync --dev
```

Run the non-Docker regression subset:

```bash
uv run pytest -q \
  tests/test_manifest.py \
  tests/test_golden_artifact_regression_gate.py \
  tests/test_episode_contract.py \
  tests/test_artifact_path_resolver.py \
  tests/test_strict_answer_contract.py \
  tests/test_robust_matching.py \
  tests/test_shared_filters.py \
  tests/test_cli_entrypoint_contract.py
```

Run lint:

```bash
uv run --with ruff ruff check .
```

## Inspect Existing Data

Show the data inventory:

```bash
uv run csvagent status
```

Show generation progress and gaps:

```bash
uv run csvagent progress
uv run csvagent stats --gaps
```

Preview generated questions without doing a large run:

```bash
uv run csvagent inspect questions --source template
uv run csvagent generate questions --template --dry-run
```

Run the quick end-to-end smoke path:

```bash
uv run csvagent run --test
```

## Heavier Or External Work To Skip In A Review

- LLM generation/teacher triangulation can require API/model access.
- Full dataset generation can be long-running.
- Kaggle download requires account credentials.
- Full `uv run pytest -q` includes Docker-backed sandbox/container tests and
  requires Docker to be available locally.

## What This Demonstrates

- The CLI is the primary control surface.
- Questions and episodes have explicit contracts.
- Golden artifacts catch unintended output drift.
- Manifest caching gives resumable generation instead of one-shot scripts.
- The same raw episodes can feed multiple downstream training formats.
