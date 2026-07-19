# csv-agent Architecture

`csv-agent` builds verified CSV-analysis training data by turning real tabular
datasets into questions, executing candidate analyses, and saving structured
episode traces that can later be converted into SFT, DPO, PRM, or evaluation
formats.

## System Flow

```text
Kaggle/local CSV datasets
-> source/spec resolution
-> question generation
   -> deterministic templates
   -> procedural generators
   -> optional LLM exploration
-> episode execution in a constrained analysis runtime
-> teacher/consistency verification
-> manifest cache and golden artifact checks
-> reusable raw trace episodes
-> downstream training/eval exports
```

The experimental prefix-value path branches from a verified episode and runs
separately from training-data generation:

```text
episode question + private terminal-verifier inputs
-> fresh actor rollout
-> exact nonterminal completed-turn boundary
-> replay code cells in a fresh sandbox
-> reject the continuation if replayed execution diverges
-> seeded continuations under one recorded actor policy
-> terminal-verifier verdicts
-> auditable prefix-value JSONL record
```

The serialized prefix contains only public agent state: the CSV source,
question, system prompt, exact model responses, execution results, conversation
feedback, and remaining turn budget. Expected answers and ground-truth hashes
remain private verifier inputs. Hook records are retained as diagnostics but do
not control terminal acceptance or supply value labels.

## Main Boundaries

- `src/cli.py`: single `csvagent` command surface for status, progress,
  generation, validation, inspection, and stats.
- `src/core/config.py`: Pydantic configuration for generation, verification,
  triangulation, and runtime settings.
- `src/datagen/`: question and episode generation workflows.
- `src/value/`: experimental prefix construction, replayed continuations, and
  terminal-verifier value collection.
- `src/contracts/` and `packages/csv-spec/`: shared schemas/contracts for CSV
  source specs, generated artifacts, and prefix-value records.
- `scripts/experiments/collect_prefix_values.py`: bounded manual real-model
  canary; it is intentionally outside the main `csvagent` CLI.
- `tests/fixtures/golden_artifacts/`: regression fixtures that make artifact
  drift explicit.
- `data/datagen_manifest.jsonl`: resumable cache of processed questions and
  template/content fingerprints.

## Engineering Claims

- Generation is resumable: interrupted runs can restart from the manifest.
- Training formats are derived from raw traces instead of baked into generation.
- Artifact drift is testable through golden fixtures.
- Path/source resolution and sandbox behavior are covered by tests.
- Live LLM work is separated from local smoke/regression tests.
- Prefix replay fails fast when code execution, outputs, hooks, or submissions
  differ from the recorded boundary.
- Prefix values are tied to a recorded actor policy, horizon, seeds, code
  commit, and dataset revision.

## What Reviewers Should Inspect

- `README.md` for the CLI and happy path.
- `docs/reviewer-demo.md` for a low-cost local demo.
- `tests/test_golden_artifact_regression_gate.py` for drift detection.
- `tests/test_manifest.py` for resumable generation behavior.
- `tests/test_sandbox_security.py` for runtime safety boundaries.
- `packages/csv-spec/README.md` for the shared source-spec contract.
- `research.md` for the selected execution-aware value-model direction.
- `scripts/experiments/README.md` for the bounded prefix-value canary.

## Current Limits

- Some commands require Docker or real model/API access and are not part of the
  default local demo path.
- Prefix-value collection is an initial feasibility instrument, not critic
  training or a production data pipeline; provider seed support may also vary.
- The repo still needs a small checked-in sample trace specifically curated for
  external reviewers.
- Public release hygiene should include a changelog and versioned package
  metadata once the project is treated as reusable tooling.
