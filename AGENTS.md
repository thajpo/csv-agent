# AGENTS.md

## Repo Collaboration Defaults
- This repository is in active development.
- Backward compatibility is **not** a default goal.
- Prefer fail-fast behavior over compatibility shims.
- Prefer clean schema/contract changes and reruns over legacy branching.

## Markdown Workflow (Lean Flow Canonical)
- `current.md` is the only active planning file.
- Required sections in `current.md`:
  - `Institutional Knowledge`
  - `Beliefs`
  - `Brainstormed`
  - `Specd`

## Working Rules
- Keep discussion append-style under the active item, including inline `//` notes.
- Never implement directly from `Brainstormed`.
- Implement only from `Specd` items with explicit contract fields.
- Communication-first before implementation: confirm intent with the user directly in natural dialogue, even when a spec looks complete.
- Do not fabricate, infer, or backfill user confirmation.
- Prune stale text after promotion or merge.
- After merge, remove the `Specd` item. Git history is the canonical completion record.

## Canonical Feature Workflow
1. Capture and refine ideas in `Brainstormed`.
2. Promote to `Specd` only when the spec contract is complete.
3. Run test-first implementation via subagent from `Specd`.
4. Run regression tests for touched surfaces.
5. Merge and prune the completed `Specd` item.

## Guardrails
- No implementation without explicit `Specd` contract.
- No promotion to `Specd: ready` without direct user-confirmed readiness evidence (boundary, non-goals, tests, and risks).
- Spec-gate evidence is required, but a fixed number of scripted questions is not.
- Do not skip/disable tests to force pass.
- Do not hide behavior changes behind ambiguous defaults.
- Prefer fail-fast contracts over compatibility branches.
