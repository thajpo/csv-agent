# current.md

Lean Flow canonical planning file.
Last updated: 2026-02-14

## Institutional Knowledge
- Repo default: fail-fast contracts over backward-compat shims.
- Planning is single-file (`current.md`).
- Implementation is allowed only from `Specd: ready`.
- `Ready` requires manual approval evidence in this markdown file (chat approval alone is insufficient).
- Overlap analysis is required before promotion to `ready`.
- Repo scope: dataset generation + Hugging Face upload. Training/fine-tune is out of scope.

## Beliefs
- Short plans improve execution speed.
- Entrypoints must be unambiguous and fail-fast.
- Reliability and schema contracts should be explicit before long generation runs.
- Keep only active, execution-relevant items visible.

## Brainstormed
- no active brainstorm items

## Specd

### S1) Pipeline Contract Cleanup -> strict-answer-contract purge
status: ready

Behavior change:
- Reject legacy `_ground_truth` / `_ground_truths`; accept only unified answer keys.

Surfaces touched:
- `src/datagen/shared/verification.py`
- `src/datagen/shared/questions_io.py`
- `src/datagen/validate_synthetic.py`

Non-goals:
- No migration adapters.
- No procedural metadata policy change.

Tests:
- Add fail-first legacy-key rejection test.
- Run regression on synthetic validation + question-io.

Risk and rollback:
- Risk: old cached files with underscored keys fail by design.
- Rollback trigger: unified-key payload regressions in validation/load path.

Overlap analysis:
- Overlaps with `S3` only at runtime generation validation surfaces.
- Merge/split decision: keep split (schema contract cleanup is independent of entrypoint command contract).

Manual approval evidence:
- approver: user (manual markdown approval recorded)
- approval_date: 2026-02-14
- approved_scope: implement exactly the defined behavior change/surfaces/tests/non-goals in this S1 spec only
- unresolved_blockers: none

### S2) Reliability Hardening -> container-stop observability
status: ready

Behavior change:
- Failed `ContainerPool.stop()` calls log container id + error while continuing shutdown.

Surfaces touched:
- `src/envs/container_pool.py`
- `tests/test_container_pool.py`

Non-goals:
- No restart/retry policy.

Tests:
- Add fail-first test expecting error log when one container stop fails.

Risk and rollback:
- Risk: noisier shutdown logs in unstable local Docker environments.
- Rollback trigger: if shutdown semantics change beyond observability-only scope.

Overlap analysis:
- No direct overlap with `S1` or `S3` file surfaces.
- Merge/split decision: keep separate and execute first as reliability unblocker.

Manual approval evidence:
- approver: user (manual markdown approval recorded)
- approval_date: 2026-02-14
- approved_scope: implement observability-only shutdown logging for failed stop calls as defined in S2
- unresolved_blockers: none

### S3) Entrypoint Contract Cleanup
status: ready

Behavior change:
- Remove ambiguous `synth` naming from user-facing generation commands.
- Introduce explicit generation modes: `llm_gen`, `template`, `procedural`.
- Standardize canonical commands for smoke/full/upload and remove duplicate entry scripts/docs drift.
- Add fail-fast validation for conflicting/legacy modes and invalid combinations.

Naming decisions (locked):
- Canonical generation modes:
  - `template`
  - `procedural`
  - `llm_gen`
- CLI flag style:
  - expose as kebab-case flags: `--template`, `--procedural`, `--llm-gen`
  - internal mode value for LLM generation remains `llm_gen` (underscore; matches codebase style)
- Canonical run commands (target contract):
  - `csvagent run --template`
  - `csvagent run --procedural`
  - `csvagent run --llm-gen`
  - `csvagent run --all`
  - `csvagent run --test` (smoke)
- Stage-level parity:
  - keep stage-level commands (`generate questions`, `generate episodes`)
  - keep mode flags available for all stages, including `--all`
- Episodes-only contract:
  - use `generate episodes --<mode>` (or `--all`) as the single episodes-only path
  - remove `--triangulate` and do not introduce `--verify`
- `--all` semantics:
  - run question generation + episode verification for `template`, `procedural`, and `llm_gen`
- Inspect contract:
  - require explicit source/mode selection (no implicit defaults)
  - use `all|template|procedural|llm_gen` vocabulary consistently
- Legacy policy:
  - no legacy aliases or compatibility shims
  - legacy `--synth` and old mode names hard-fail with no migration messaging
- Error UX:
  - fail fast, but print valid options for required mode/source flags

Canonical command enumeration (post-refactor target):
- Status/progress/meta:
  - `csvagent status`
  - `csvagent progress`
  - `csvagent manifest`
  - `csvagent stats`
  - `csvagent stats --gaps`
- Full pipeline:
  - `csvagent run --template`
  - `csvagent run --procedural`
  - `csvagent run --llm-gen`
  - `csvagent run --all`
  - `csvagent run --test`
- Stage: question generation:
  - `csvagent generate questions --template`
  - `csvagent generate questions --procedural`
  - `csvagent generate questions --llm-gen`
  - `csvagent generate questions --all`
- Stage: episode generation:
  - `csvagent generate episodes --template`
  - `csvagent generate episodes --procedural`
  - `csvagent generate episodes --llm-gen`
  - `csvagent generate episodes --all`
- Inspect/debug:
  - `csvagent inspect questions --source template`
  - `csvagent inspect questions --source procedural`
  - `csvagent inspect questions --source llm_gen`
  - `csvagent inspect questions --source all`
  - `csvagent inspect episodes --verified`
  - `csvagent inspect trace <episode_id_prefix>`
  - `csvagent validate --csv <csv> --questions-file <questions.json> --index 0 --show-code`
- Upload:
  - `huggingface-cli login`
  - `uv run python scripts/upload_hf.py --repo <org>/<dataset>`
  - `uv run python scripts/upload_hf.py --repo <org>/<dataset> --private`
- Tests:
  - `uv run pytest tests/ -v`
- Explicitly rejected:
  - any `--synth`
  - `run --triangulate`
  - duplicate HF upload path (`scripts/push_to_hf.py`)

Surfaces touched (initial target):
- `src/cli.py` (argument model + mode routing + help text)
- `src/datagen/pipeline.py` (mode handling)
- `entrypoints/`
- `README.md`
- workflow scripts under `scripts/` that duplicate canonical paths or conflict with canonical CLI
- tests for entrypoint contract behavior

Non-goals:
- No dataset schema redesign.
- No generation algorithm changes.

Tests:
- Fail-first tests for legacy `--synth` invocations (must hard-fail).
- Fail-first tests for conflicting mode selections.
- Fail-first tests for missing required mode/source selection in CLI surfaces (including inspect).
- Regression tests for canonical commands:
  - smoke run
  - full `template`
  - full `procedural`
  - full `llm_gen`
  - full `all`
  - episodes-only runs via `generate episodes --template|--procedural|--llm-gen|--all`
  - upload path

Risk and rollback:
- Risk: existing scripts/docs using `--synth` break immediately.
- Rollback trigger: inability to run canonical `template/procedural/llm_gen` workflows end-to-end.

Overlap analysis:
- Overlaps with `S1` only indirectly via runtime invocation of validation code.
- Overlaps with `S4` if training entrypoints are removed in same sweep.
- Merge/split decision: split from `S4`; execute `S3` first, then prune training surfaces.
- Additional overlap: touches any generation docs/tests currently keyed on `synth`; bundle those updates inside `S3`.
- Downstream rename scope: move persisted/source labels to `template|procedural|llm_gen` across pipeline, artifacts, inspect, and analytics.

Manual approval evidence:
- approver: user (manual markdown approval recorded)
- approval_date: 2026-02-14
- approved_scope: implement full S3 command-surface refactor exactly as specified, including mode rename, command contract, downstream naming updates, and rejected command removals
- unresolved_blockers: none

### S4) Repo Scope Cleanup -> training-surface removal
status: ready

Behavior change:
- Remove training/fine-tune code paths and training-focused docs from this repo.
- Keep generation + Hugging Face upload workflows intact.

Surfaces touched (initial target):
- `training/`
- `src/training/`
- training references in `README.md`
- tests/docs/scripts that are training-only

Non-goals:
- No changes to generation semantics.
- No changes to upload behavior except dependency cleanup tied to training removal.

Tests:
- Fail-first check: references to removed training entrypoints should fail cleanly.
- Regression: generation smoke/full + HF upload smoke continue to work.

Risk and rollback:
- Risk: hidden dependencies on training modules may break imports.
- Rollback trigger: generation/upload workflows fail due to training removal.

Overlap analysis:
- Overlaps with `S3` on entrypoint/docs surfaces.
- Merge/split decision: keep split; run `S3` first to define canonical command surface, then remove training surfaces.

Manual approval evidence:
- approver: user (manual markdown approval recorded)
- approval_date: 2026-02-14
- approved_scope: remove training/fine-tune surfaces from this repo while preserving generation and HF upload workflows per S4 contract
- unresolved_blockers: none

### S5) Reliability/Validation -> strict-hook-grounding
status: ready

Behavior change:
- Hook `code_line` grounding must use normalized full-line equality (not substring).
- False-positive example to prevent: `x = 1` matching `x = 10`.

Surfaces touched (initial target):
- `src/core/environment.py`
- tests covering hook grounding behavior

Non-goals:
- No prompt redesign.
- No generation template rewrite in this slice.

Tests:
- Fail-first: substring-only matches are rejected.
- Regression: valid exact/normalized line matches still pass.

Risk and rollback:
- Risk: stricter grounding may increase retries if hooks are noisy.
- Rollback trigger: grounding rejects valid hooks at high rates in smoke runs.

Overlap analysis:
- Low overlap with `S1`; moderate runtime overlap with generation validation flow.
- Merge/split decision: keep split from schema and entrypoint work.

Manual approval evidence:
- approver: user (manual markdown approval recorded)
- approval_date: 2026-02-14
- approved_scope: implement strict full-line hook grounding as defined in S5, including fail-first and regression tests
- unresolved_blockers: none

## Active Sequence
1. Finish `S2` (container-stop observability).
2. Dispatch `S3` for implementation in separate agent tab.
3. Dispatch `S1` for implementation in separate agent tab.
4. Dispatch `S4` for implementation in separate agent tab.
5. Dispatch `S5` for implementation in separate agent tab.

## Recovery Notes
- Historical planning detail remains in git history.
- This file is intentionally compact and execution-focused.
