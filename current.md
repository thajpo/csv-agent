# current.md

Lean Flow canonical planning file.
Last updated: 2026-02-16

## Institutional Knowledge
- [2026-02-08] Repo default: fail-fast contracts over backward-compat shims.
- [2026-02-08] Planning is single-file (`current.md`).
- [2026-02-08] Git history/PRs are canonical for shipped work; this file is canonical for unimplemented work.
- [2026-02-16] MERGED: PR #11 (issue #6) unified source contract to `llm_gen|template|procedural`.
- [2026-02-16] MERGED: PR #12 (issue #7) removed legacy `_ground_truth*` compatibility paths in targeted surfaces.
- [2026-02-16] MERGED: PR #13 (issue #5) removed a large part of `skip_existing` legacy behavior, but residual paths remained.
- [2026-02-16] MERGED: strict hook grounding (issue #10) now uses exact line matching.

## Beliefs
- Prefer one canonical contract path; avoid dual/legacy branches.
- Cleanup should be test-driven and finish fully, not partially.
- Pipeline reliability gates training scale-up.

## Brainstormed

### Residual Contract Purge (active)
status: in_progress
readiness: executing

description:
- Finish the remaining contract cleanup after merged issues #5/#6/#7.

current code evidence:
- residual `skip_existing` code paths remained in CLI + datagen runtime.
- residual legacy `question` fallback reads remained in contract-facing code.
- source mismatch existed in some inspect/preflight filters (`llm` vs `llm_gen`).

files being touched:
- `src/cli.py`
- `src/datagen/validate_synthetic.py`
- `src/datagen/episode_gen.py`
- `src/datagen/pipeline.py`
- `src/datagen/validate_question.py`
- `src/datagen/shared/episode_factory.py`
- `src/datagen/shared/questions_io.py`
- `src/datagen/shared/filters.py`
- `src/gui/panels/explorer.py`
- `src/gui/panels/trace.py`
- `src/utils/inspect.py`
- `src/datagen/question_gen.py`

acceptance:
- no `skip_existing` references remain under `src/`.
- no contract-facing `q.get("question", ...)` fallbacks remain in touched runtime surfaces.
- source filters consistently use `llm_gen` for LLM-generated questions.

### Test Hardening (active)
status: in_progress
readiness: executing

description:
- Add regression tests proving residual purge invariants stay enforced.

planned tests:
- new contract test: no `skip_existing` in runtime signatures/calls.
- source vocabulary tests for inspect/preflight (`llm_gen` only).
- schema checks for removed parallel metadata fields.

### Pipeline Validation + Small Batch Eval
status: pending
readiness: near-spec

description:
- Run end-to-end validation matrix and evaluate small batches per source.

notes:
- depends on residual purge + test hardening finishing first.
- open issue #9 (Dataset-Quality Snapshot) should be used in this phase.

## Specd
- no active specd items (direct execution requested by user on `main`)

## Recovery Notes
- Prior planning content was heavily compressed to reduce noise and keep active work scannable.
