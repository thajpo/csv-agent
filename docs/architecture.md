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

## Main Boundaries

- `src/cli.py`: single `csvagent` command surface for status, progress,
  generation, validation, inspection, and stats.
- `src/core/config.py`: Pydantic configuration for generation, verification,
  triangulation, and runtime settings.
- `src/datagen/`: question and episode generation workflows.
- `src/contracts/` and `packages/csv-spec/`: shared schemas/contracts for CSV
  source specs and generated artifacts.
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

## What Reviewers Should Inspect

- `README.md` for the CLI and happy path.
- `docs/reviewer-demo.md` for a low-cost local demo.
- `tests/test_golden_artifact_regression_gate.py` for drift detection.
- `tests/test_manifest.py` for resumable generation behavior.
- `tests/test_sandbox_security.py` for runtime safety boundaries.
- `packages/csv-spec/README.md` for the shared source-spec contract.

## Current Limits

- Some commands require Docker or real model/API access and are not part of the
  default local demo path.
- The repo still needs a small checked-in sample trace specifically curated for
  external reviewers.
- Public release hygiene should include a changelog and versioned package
  metadata once the project is treated as reusable tooling.
