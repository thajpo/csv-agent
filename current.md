# current.md

Lean Flow canonical planning file.
Last updated: 2026-02-17

## Institutional Knowledge
- [2026-02-08] Repo default: fail-fast contracts over backward-compat shims.
- [2026-02-08] Planning is single-file (`current.md`).
- [2026-02-08] Git history/PRs are canonical for shipped work; this file is canonical for unimplemented work.
- [2026-02-16] MERGED: PR #11 (issue #6) unified source contract to `llm_gen|template|procedural`.
- [2026-02-16] MERGED: PR #12 (issue #7) removed legacy `_ground_truth*` compatibility paths in targeted surfaces.
- [2026-02-16] MERGED: PR #13 (issue #5) removed a large part of `skip_existing` legacy behavior, but residual paths remained.
- [2026-02-16] MERGED: strict hook grounding (issue #10) now uses exact line matching.
- [2026-02-16] MERGED: PR #15 (issue #14) locked no-data-loss fail-fast write semantics and unified contract keys.
- [2026-02-16] MERGED: PR #17 (issue #16) migrated artifact storage to source-scoped layout and removed mixed runtime paths.

## Beliefs
- Prefer one canonical contract path; avoid dual/legacy branches.
- Cleanup should be test-driven and finish fully, not partially.
- Pipeline reliability gates training scale-up.

## Brainstormed

### Dead Code + Duplicate Functionality Sweep
status: in_progress
readiness: executing

description:
- delete as much dead/duplicate code as possible without behavior regressions.

current code evidence:
- broad lint/dead-code cleanup merged in `3057545`.
- source-split migration merged in PR #17 reduced overlap in path handling.
- root-level experimental test scripts still clutter repo root and should be relocated.

execution approach:
- run file-by-file duplicate detection + usage mapping.
- remove dead wrappers/helpers only when covered by regression tests.
- keep behavior-neutral mechanical removals isolated from feature changes.

acceptance:
- no dead wrappers in targeted modules.
- no duplicate active code paths for the same contract behavior.
- tests + lint stay green.

### Root Test Script Cleanup
status: in_progress
readiness: executing

description:
- move root-level `test_*.py` experiment scripts out of repo root into a dedicated scripts area.

acceptance:
- no root-level `test_*.py` files remain.
- test discovery remains constrained to `tests/` only.
- commands/docs for running experiment scripts are explicit.

### Pipeline Validation + Small Batch Eval
status: pending
readiness: near-spec

description:
- Run end-to-end validation matrix and evaluate small batches per source.

notes:
- depends on residual purge + test hardening finishing first.
- open issue #9 (Dataset-Quality Snapshot) should be used in this phase.

## Specd
- no active specd items

## Recovery Notes
- Prior planning content was heavily compressed to reduce noise and keep active work scannable.
