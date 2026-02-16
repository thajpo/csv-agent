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
status: done
readiness: completed

description:
- Finish the remaining contract cleanup after merged issues #5/#6/#7.

current code evidence:
- completed in commit `3057545` on `main`.

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
- met: no `skip_existing` references remain under `src/`.
- met: contract-facing fallback cleanup completed for targeted runtime surfaces.
- met: source filters now consistently use `llm_gen`.

### Test Hardening (active)
status: done
readiness: completed

description:
- Add regression tests proving residual purge invariants stay enforced.

planned tests:
- completed in commit `3057545`:
  - `tests/test_residual_contract_purge.py`
  - full lint pass (`ruff check src tests`)
  - full test pass (`257 passed, 6 skipped`)

### Dead Code + Duplicate Functionality Sweep
status: in_progress
readiness: executing

description:
- delete as much dead/duplicate code as possible without behavior regressions.

current code evidence:
- broad lint/dead-code cleanup already merged in `3057545`.
- likely duplicate functionality still exists in analysis/debug helper surfaces.

execution approach:
- run file-by-file duplicate detection + usage mapping.
- remove dead wrappers/helpers only when covered by regression tests.
- keep behavior-neutral mechanical removals isolated from feature changes.

acceptance:
- no dead wrappers in targeted modules.
- no duplicate active code paths for the same contract behavior.
- tests + lint stay green.

### Pipeline Validation + Small Batch Eval
status: pending
readiness: near-spec

description:
- Run end-to-end validation matrix and evaluate small batches per source.

notes:
- depends on residual purge + test hardening finishing first.
- open issue #9 (Dataset-Quality Snapshot) should be used in this phase.

## Specd

- Unified Episode/Question Contract + No-Data-Loss Regression Gate | status: issued | issue: #14

## Recovery Notes
- Prior planning content was heavily compressed to reduce noise and keep active work scannable.
