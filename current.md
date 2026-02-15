# current.md

Lean Flow canonical planning file.
Last rebuilt with recovered historical context: 2026-02-08

This file intentionally carries more detail now. The previous version was too sparse and hid what was near-spec.

## Institutional Knowledge
- [2026-02-08] Repo default: fail-fast contracts over backward-compat shims.
- [2026-02-08] Planning is single-file (`current.md`), but depth matters; compression should not erase decision-critical context.
- [2026-02-08] Implementation is allowed only from `Specd`.
- [2026-02-08] Git history/PRs are canonical for shipped work; this file is canonical for unimplemented work.
- [2026-02-08] Synthetic/program generation and verification are first-class, not sidecar features.
- [2026-02-08] Question quality target is data-aware + difficult + non-procedural wording when verbalized.
- [2026-02-08] If planning context is pruned, recover from Git history rather than guessing.
- [2026-02-08] Communication-first gate: before implementation, confirm intent with the user in natural conversation even if a spec appears complete.
- [2026-02-08] Spec-gate is evidence-based but not rigidly questionnaire-based; do not require a fixed q-count when user confirmation is clear.

## Beliefs
- [2026-02-08] Minimal file count helps only if each remaining file is information-dense and reviewable.
- [2026-02-08] Several brainstorm tracks were already close to spec and only needed explicit file/test contracts.
- [2026-02-08] Reliability should gate training scale-up to avoid training on corrupted traces.
- [2026-02-08] Pipeline contract cleanup and synthetic ambiguity cleanup should share one metadata abstraction pass where possible.
- [2026-02-08] PM-style review improves when each candidate includes concrete touch points and expected diff shape.
- [2026-02-08] Honest, straightforward dialogue is required; inferred confirmation is not acceptable evidence for readiness.

## Brainstormed

### Pipeline Contract Cleanup
status: open
readiness: near-spec

recovered context:
- Original direction: unify question schema, keep synthetic mechanical canonical, separate verification from generation, and remove hidden legacy fallbacks.
- Metadata intent was explicit `source/subtype` plus procedural tracking without brittle naming hacks.

current code evidence:
- Unified loader exists: `src/datagen/shared/questions_io.py`.
- Centralized shared modules exist: `src/datagen/shared/{verification.py,submission.py,dataset_meta.py,episode_factory.py}`.
- Mixed legacy still present:
  - `src/datagen/shared/verification.py` still reads `_ground_truth` and `_ground_truths`.
  - `src/datagen/episode_gen.py` still carries `skip_existing` legacy path.
  - `question` fallback reads still appear in multiple paths.

missing:
- Canonical triad schema contract across question/episode/inspect surfaces: `template | procedural | llm`.
- Removal of parallel discriminator fields/labels that duplicate the triad concept.
- Cleanup of fallback reads that silently normalize old shapes.

spec candidates (not yet promoted):
- candidate: procedural-metadata normalization
  - behavior change: enforce one triad schema end-to-end (`template|procedural|llm`) and remove parallel procedural flags/aliases from contract-facing paths.
  - user intent: unify schema to explicit triad only; no split between boolean flags and subtype aliases.
  - files to touch:
    - `src/datagen/shared/questions_io.py`
    - `src/datagen/synthetic/programs/program_generator.py`
    - `src/datagen/shared/episode_factory.py`
    - `src/datagen/validate_synthetic.py`
    - `src/utils/inspect.py`
    - `tests/test_program_generator_schema.py`
    - `tests/test_episode_factory.py`
  - fail-first tests:
    - assert triad vocabulary is the only accepted contract in touched CLI/inspect/validator surfaces.
    - assert procedural records are represented via triad schema without auxiliary flag dependencies.
  - non-goals:
    - no downstream analytics redesign.
  - risks:
    - breaks artifacts or scripts expecting legacy subtype/boolean distinctions.
  - touch points:
    - `QuestionRecord` type/validation contract
    - episode source labeling + inspect filters
  - expected diff shape:
    - modify, ~120-220 LOC
  - review checks:
    - all touched surfaces accept/emit only `template|procedural|llm` vocabulary.

### Synthetic Question Quality Improvements
status: open
readiness: near-spec

recovered context:
- Prior pass-rate analysis identified ambiguity clusters:
  - column choice ambiguity
  - include/exclude target ambiguity
  - implicit parameter ambiguity
  - method-choice ambiguity.
- Older improvement spec called out concrete templates for fixes.

current code evidence:
- Many fixes already landed in `src/datagen/synthetic/templates.py`:
  - explicit parameters exist in several outlier descriptions.
  - exclude-target alternatives exist for some correlation templates.
  - `ks_normality_test` and `adaptive_two_sample_test` have more explicit method framing.
- Remaining gaps are uneven template-by-template and not reconciled with a current pass-rate report.

missing:
- Verified completion pass for the original pending template set:
  - `ttest_discovered_groups`
  - `spearman_rank_correlation`
  - `regression_most_predictive`
  - `correlation_change_analysis`
- Multi-path column selection policy for templates still using "first matching" logic.
- Fresh benchmark showing whether ambiguity classes actually dropped.

spec candidates (not yet promoted):
- candidate: ambiguity-template completion pack
  - behavior change: complete missing exclude-target/method-explicit alternatives for the unresolved templates above.
  - files to touch:
    - `src/datagen/synthetic/templates.py`
    - `tests/test_synthetic.py`
  - fail-first tests:
    - per-template acceptance tests expecting at least one alternative execution path.
  - non-goals:
    - no verbalizer redesign in this slice.
  - risks:
    - template explosion increases generation cost.
  - touch points:
    - template definitions + `alternative_code_templates`
  - expected diff shape:
    - modify/add template blocks, ~150-350 LOC
  - review checks:
    - each targeted template has deterministic alternatives and stable output schema.

- candidate: multi-path selection in generator
  - behavior change: enumerate valid column choices for selected templates and accept any valid output.
  - files to touch:
    - `src/datagen/synthetic/generator.py`
    - `src/datagen/synthetic/templates.py`
    - `tests/test_synthetic.py`
  - fail-first tests:
    - generator test fails when only first-match path is accepted.
  - non-goals:
    - no global probabilistic scoring redesign.
  - risks:
    - runtime increase if path count is unbounded.
  - touch points:
    - candidate expansion + answer-hash acceptance
  - expected diff shape:
    - modify, ~120-220 LOC
  - review checks:
    - logs show multiple valid paths evaluated.

### Compositional Generator Reconciliation
status: open
readiness: mid-spec

recovered context:
- Intended target was typed grammar search, full binding enumeration, dead-code rejection, and system-2 decision coverage.
- Historical spec had checklist language; current code/test state appears ahead of older narrative docs.

current code evidence:
- Strong infra present:
  - grammar/enumeration/reduction stack in `src/datagen/synthetic/programs/`.
  - dead-code validator integrated in sampler.
  - tests exist:
    - `tests/test_programs_smoke.py`
    - `tests/test_dead_code_validator.py`
    - `tests/test_program_generator_schema.py`
- One explicit gap is still acknowledged in tests:
  - `test_program_count_floor` does not enforce target counts yet.

missing:
- A clear evidence matrix mapping intended requirements -> exact tests/code.
- Final target policy on minimum accepted program counts by dataset class.
- Confidence report on operator-family coverage and failure reasons.

spec candidates (not yet promoted):
- candidate: checklist-evidence reconciler
  - behavior change: add machine-checkable requirement matrix and enforce missing invariants.
  - files to touch:
    - `tests/test_programs_smoke.py`
    - `src/datagen/synthetic/programs/sampler.py`
    - `src/datagen/synthetic/programs/filter.py`
  - fail-first tests:
    - enforce non-trivial floor policy (for example fail when zero decision chains where eligible).
  - non-goals:
    - no new operator families in this slice.
  - risks:
    - brittle tests on tiny datasets.
  - touch points:
    - floor/coverage assertions
  - expected diff shape:
    - modify, ~80-180 LOC
  - review checks:
    - test suite clearly signals unmet compositional guarantees.

### Training and E2E Readiness
status: open
readiness: near-spec

recovered context:
- Training pipeline existed but ownership/pathing was inconsistent (`training/train_sft.py` vs `src/training` docs).
- Validation intent was: real episode generation -> format conversion -> trainability sanity checks.

current code evidence:
- `training/train_sft.py` exists.
- `src/training/prepare_finetune_data.py` exists and is active.
- Path split remains, so ownership is still ambiguous.

missing:
- Single canonical training entrypoint and location policy.
- One reproducible E2E smoke command set with expected outputs.
- Clear "minimum ready" gate before running larger training jobs.

spec candidates (not yet promoted):
- candidate: training-path normalization
  - behavior change: define canonical training script location and remove duplicate ambiguity.
  - files to touch:
    - `training/train_sft.py`
    - `src/training/__init__.py`
    - `README.md`
  - fail-first tests:
    - invocation test for canonical command path.
  - non-goals:
    - no model-quality claims.
  - risks:
    - user scripts pointing at old path break.
  - touch points:
    - module import path and docs commands
  - expected diff shape:
    - modify, ~60-140 LOC
  - review checks:
    - one documented command path works end-to-end.

- candidate: E2E readiness smoke suite
  - behavior change: add deterministic smoke workflow that validates generation + formatting + eval harness wiring.
  - files to touch:
    - `tests/` (new smoke test module)
    - `README.md`
  - fail-first tests:
    - smoke test asserts output file count/shape and non-empty formatted examples.
  - non-goals:
    - no full real-API load test in CI.
  - risks:
    - flaky if external dependencies leak into the smoke path.
  - touch points:
    - fixture-based episodes and converter output
  - expected diff shape:
    - add tests/docs, ~120-220 LOC
  - review checks:
    - smoke suite is local/offline reproducible.

### Reliability Hardening
status: open
readiness: near-spec

recovered context:
- Historical bug investigation prioritized:
  - hook parse failures
  - container stop failures
  - trace alignment mismatches
  - worker/container health checks.

current code evidence:
- Hook parser now logs malformed entries (`src/datagen/teacher.py`).
- Trace turn mismatch now logs warnings (`src/datagen/teacher.py`).
- Container pool stop still swallows stop exceptions without logging (`src/envs/container_pool.py`).
- Hook grounding still uses substring matching (`src/core/environment.py`) and can false-positive.

missing:
- Explicit logging/reporting for container stop failures.
- Worker health/recovery checks in long-running pool usage.
- Strict hook grounding that avoids substring false positives.

spec candidates (not yet promoted):
- candidate: container-stop observability
  - behavior change: failed container stop calls are logged with container id and error.
  - files to touch:
    - `src/envs/container_pool.py`
    - `tests/` (new unit tests with mocked stop failures)
  - fail-first tests:
    - test expects logged errors when one `stop()` coroutine fails.
  - non-goals:
    - no automatic restart in this slice.
  - risks:
    - noisy logs if shutdown is frequently interrupted.
  - touch points:
    - `ContainerPool.stop()`
  - expected diff shape:
    - modify, ~30-90 LOC
  - review checks:
    - no swallowed exception path during shutdown.

- candidate: strict-hook-grounding
  - behavior change: hook `code_line` grounding requires normalized line-level exact match, not substring.
  - files to touch:
    - `src/core/environment.py`
    - `tests/` (hook grounding tests)
  - fail-first tests:
    - `"x = 1"` must not ground against only `"x = 10"`.
  - non-goals:
    - no model prompt rewrite.
  - risks:
    - stricter grounding may increase reprompts.
  - touch points:
    - `validate_hooks_grounded`
  - expected diff shape:
    - modify, ~60-120 LOC
  - review checks:
    - known false-positive cases no longer pass.

### Research Program Consolidation
status: open
readiness: early

recovered context:
- Prior research docs covered PRM vs ORM, DAG-aware credit assignment, curriculum, holdout generalization, source comparison, hint ablations, scaling, and Kaggle-solution ingestion.
- These were informative but not compressed into executable experiment cards.

missing:
- A single experiment matrix with explicit run protocol and success criteria.
- Priority order tied to current engineering constraints.

spec candidates (not yet promoted):
- candidate: experiment-matrix v1
  - behavior change: add one concise research matrix section in this file with 3 immediate experiments only.
  - files to touch:
    - `current.md`
  - fail-first tests:
    - n/a (planning artifact)
  - non-goals:
    - no code changes.
  - risks:
    - drift if not maintained.
  - touch points:
    - research section only
  - expected diff shape:
    - docs-only, ~50-100 LOC
  - review checks:
    - each experiment has dataset/model/eval/success criteria.

### Context Compression Policy
status: open
readiness: early

recovered context:
- Prior notes repeatedly flagged context blow-up from tool outputs.
- Desired behavior: compact outputs while preserving verifiability.

missing:
- Explicit policy for what gets truncated, summarized, or stored verbatim.
- Consistent instrumentation for summary quality vs information loss.

spec candidates (not yet promoted):
- candidate: tool-output compression policy v1
  - behavior change: define strict summarization rules for stdout/log artifacts.
  - files to touch:
    - `current.md`
    - `src/core/environment.py` (if enforcing output budget at runtime)
  - fail-first tests:
    - summarize long output while preserving key markers/hashes.
  - non-goals:
    - no model-level long-context optimization.
  - risks:
    - excessive truncation can hide failures.
  - touch points:
    - output feedback generation paths
  - expected diff shape:
    - docs-first, optional code follow-up
  - review checks:
    - summaries retain actionable error context.

### Reliability-to-Training Gate
status: open
readiness: mid-spec

recovered context:
- Repeated intent: do not scale training before reliability smoke checks pass.

missing:
- Concrete gate definition (which tests, thresholds, and blockers).

spec candidates (not yet promoted):
- candidate: pre-training reliability gate
  - behavior change: require smoke suite pass + zero critical trace integrity failures before training runs.
  - files to touch:
    - `current.md`
    - `tests/` (gate suite)
  - fail-first tests:
    - one simulated trace-integrity failure blocks gate.
  - non-goals:
    - no performance benchmarking.
  - risks:
    - gate too strict can slow iteration.
  - touch points:
    - smoke tests + launch checklist
  - expected diff shape:
    - tests/docs, ~100-180 LOC
  - review checks:
    - gate failure clearly explains why training is blocked.

### Research Execution Template
status: open
readiness: mid-spec

recovered context:
- Needed structure was repeatedly: dataset/model/eval/success criteria.

missing:
- Standard experiment card format in this file.
- Link from each research idea to one card.

spec candidates (not yet promoted):
- candidate: experiment-card template
  - behavior change: add reusable card format and instantiate 2 immediate cards.
  - files to touch:
    - `current.md`
  - fail-first tests:
    - n/a (planning artifact)
  - non-goals:
    - no infra build.
  - risks:
    - template used inconsistently.
  - touch points:
    - research planning section
  - expected diff shape:
    - docs-only, ~40-90 LOC
  - review checks:
    - every card includes measurable success criteria.

### Merge Heuristics
status: open
readiness: early

recovered context:
- You explicitly want merged abstractions instead of parallel fragmented ideas.

missing:
- Hard criteria for when two brainstorms merge vs stay separate.

spec candidates (not yet promoted):
- candidate: merge-decision rubric
  - behavior change: add a 5-rule rubric to decide merge/split actions.
  - files to touch:
    - `current.md`
  - fail-first tests:
    - n/a (planning artifact)
  - non-goals:
    - no automated clustering.
  - risks:
    - over-merging can hide priority differences.
  - touch points:
    - brainstorming process only
  - expected diff shape:
    - docs-only, ~30-70 LOC
  - review checks:
    - rubric explains at least one concrete merge candidate.

### Pipeline Orchestrator Refactor Follow-up
status: open
readiness: early

discussion:
- PR feedback indicates `src/datagen/pipeline.py` orchestration flow is hard to read after S3 fixes. // user note in PR: "this function does feel kind of mangled together."
- Keep behavior unchanged; focus this item on structure/decomposition only.

spec candidates (not yet promoted):
- candidate: pipeline-orchestrator decomposition
  - behavior change: split orchestration helpers for stage config, synthetic append/skip handling, and summary reporting.
  - files to touch:
    - `src/datagen/pipeline.py`
    - tests that assert stage ordering/flags
  - fail-first tests:
    - verify `run --all` preserves stage 2a + 2b outputs and does not regress explicit mode behavior.
  - non-goals:
    - no command contract changes.
    - no new generation logic.
  - risks:
    - accidental stage ordering regressions during extraction.
  - touch points:
    - stage builder helpers
    - skip-existing ID loading and append behavior
  - expected diff shape:
    - refactor-only, ~80-180 LOC
  - review checks:
    - `pipeline.main` reads as orchestration only (minimal inline logic).

- candidate: source-split storage layout
  - behavior change: split question/episode artifact storage by source (`template`, `procedural`, `llm`) instead of mixed synthetic files.
  - user intent: make ownership and overwrite semantics explicit by directory/file layout.
  - files to touch:
    - `src/datagen/pipeline.py`
    - `src/datagen/validate_synthetic.py`
    - `src/datagen/synthetic/programs/program_generator.py`
    - `src/cli.py`
    - inspect/readers that assume mixed synthetic paths
  - fail-first tests:
    - assert `generate --all` and `run --all` keep each source artifact isolated with no cross-source truncation.
  - non-goals:
    - no schema redesign beyond storage layout.
  - risks:
    - migration complexity for existing mixed artifacts and scripts.
  - touch points:
    - output path resolution and default artifact names
  - expected diff shape:
    - multi-file refactor, ~120-260 LOC
  - review checks:
    - source-specific commands read/write only their source-scoped artifacts by default.

## Specd
- title: Pipeline Contract Cleanup -> strict-answer-contract purge
  status: ready
  approval:
    approved_by: user
    approval_signal: "i accept"
    approved_on: 2026-02-08
    expires_on: 2026-02-13
    approval_status: valid
    approval_basis: strict no-backward-compat cleanup after Q&A; implement strict-answer-contract purge only.
  - readiness evidence:
    - [2026-02-08] user confirmed strict no-backward-compat plan in natural dialogue.
    - [2026-02-08] boundary confirmed: remove `_ground_truth` / `_ground_truths` fallback in runtime verification/load paths only.
    - [2026-02-08] non-goals confirmed: no migration adapters and no procedural metadata policy changes.
    - [2026-02-08] tests confirmed: fail-first legacy-key rejection test plus synthetic validation/question-io regression checks.
    - [2026-02-08] risk accepted: old cached question files using underscored keys fail by design.
    - [2026-02-08] user approval signal captured: "i accept".
    - [2026-02-09] promotion output record: ready; blockers none; rationale: contract complete and approved.
  - behavior change: verification/load paths reject legacy `_ground_truth` and `_ground_truths` fields and accept only unified answer keys.
  - must stay unchanged: canonical unified question schema, synthetic verification semantics, and existing non-legacy question loading behavior.
  - files to touch:
    - `src/datagen/shared/verification.py`
    - `src/datagen/shared/questions_io.py`
    - `src/datagen/validate_synthetic.py`
  - fail-first tests:
    - add test proving legacy `_ground_truth` input fails schema/verification.
  - regression tests:
    - run synthetic validation and question-io tests to confirm unified-key payloads continue to pass.
  - non-goals:
    - no migration adapters for old JSON files.
    - no procedural metadata policy changes in this slice.
  - risks:
    - old cached question files using underscored keys stop loading by design.
  - rollback trigger:
    - if canonical unified-key payloads regress, pause and revert strict-key rejection until parser/tests are fixed.
  - overlap decision:
    - split from `procedural-metadata normalization`; this slice only removes answer-key legacy fallback paths.
  - dependency snapshot:
    - depends_on: none
    - blocked_by: none
    - parallelizable_with:
      - `Reliability Hardening -> container-stop observability`
    - invalidation_watch:
      - any upstream spec or merged PR changing answer schema/validation assumptions for these touch points
  - touch points:
    - `src/datagen/shared/verification.py` -> `verify_synthetic` fallback branch removal
    - `src/datagen/shared/questions_io.py` -> `validate_question` required-field checks
    - `src/datagen/validate_synthetic.py` -> strict schema validation path
  - line anchors:
    - `src/datagen/shared/verification.py`
    - `src/datagen/shared/questions_io.py`
    - `src/datagen/validate_synthetic.py`
  - expected diff shape:
    - modify only, ~60-120 LOC
  - review checks:
    - no underscored fallback keys remain in runtime verification path.
    - unified-key payloads still validate successfully.

- title: Reliability Hardening -> container-stop observability
  status: in_progress
  - behavior change: failed `ContainerPool.stop()` container stop calls are logged with container id and error while continuing shutdown.
  - files to touch:
    - `current.md`
    - `src/envs/container_pool.py`
    - `tests/test_container_pool.py`
  - fail-first tests:
    - add test expecting an error log when one container stop raises during pool shutdown.
  - non-goals:
    - no automatic container restart or retry policy in this slice.
  - risks:
    - shutdown logs may be noisy in unstable local Docker environments.
  - touch points:
    - `src/envs/container_pool.py` -> `ContainerPool.stop`
    - `tests/test_container_pool.py` -> `TestContainerPool`
  - line anchors:
    - `src/envs/container_pool.py:1005`
  - expected diff shape:
    - modify only, ~40-90 LOC
  - review checks:
    - no stop exception is silently swallowed.
    - failing container ids appear in error logs.

ready-to-promote shortlist (based on recovered context):
- Synthetic Question Quality Improvements -> ambiguity-template completion pack
- Training and E2E Readiness -> training-path normalization

promotion checklist for each `Specd` item:
- behavior change
- files to touch
- fail-first tests
- non-goals
- risks
- touch points (path + function/class/block)
- line anchors (optional; reviewer convenience only)
- expected diff shape (add/modify/delete + rough LOC)
- review checks

## Recovery Notes
- Key planning details were recovered from `HEAD` history of previously deleted markdown docs and reattached here.
- This rebuild intentionally keeps context in-place so we do not relive the same planning loops.
