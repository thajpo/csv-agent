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
- `Specd` items must include an explicit `user intent` field.
- Communication-first before implementation: confirm intent with the user directly in natural dialogue, even when a spec looks complete.
- Do not fabricate, infer, or backfill user confirmation.
- Define touched-file scope in the approved spec/issue before implementation starts.
- Touching files outside approved scope is a workflow failure unless user explicitly approves scope expansion.
- After issue creation for a ready item, compact the full `Specd` body to a tracker line:
  - `<id/title> | status: issued | issue: <url-or-#>`
- Do not track PR state in `current.md`; PR lifecycle is tracked in GitHub.
- After merge, remove the issued tracker line. Git history is the canonical completion record.

## Canonical Feature Workflow
1. Capture and refine ideas in `Brainstormed`.
2. Promote to `Specd` only when the spec contract is complete.
3. Mark `Specd` item `ready` only with explicit markdown approval evidence.
4. Create exactly one GitHub issue per `ready` item with full approved spec contract.
5. Replace the full `Specd` body with an `issued` tracker line in `current.md`.
6. Implement through branch -> commits -> PR linked to the issue.
7. Run regression tests for touched surfaces and resolve PR feedback.
8. Merge PR and prune the issued tracker line from `current.md`.

## Authority Switch
- Before issue creation: `current.md` spec contract is the planning source of truth.
- After issue creation: GitHub issue body is the execution contract source of truth.
- After PR creation: PR thread/review feedback is the iteration source of truth.
- If conflict exists, latest explicit user instruction in the PR context wins.

## PR Feedback Ingestion
- Any PR-touching agent/workflow must read all available PR feedback channels:
  - PR conversation comments
  - review summaries/states
  - inline review threads with file/line context
  - unresolved thread state
- Agent must not declare completion while unresolved blocking feedback remains.

## PR Status Cadence
- After every push to an open PR, post a status comment containing:
  - scope addressed
  - changes made
  - files touched
  - commands run
  - test results
  - remaining open items
  - risks/assumptions

## Baseline PR Checks
- All repos should enforce baseline PR checks:
  - lint
  - test

## Guardrails
- No implementation without explicit `Specd` contract.
- No `Specd: ready` without an explicit `user intent` field.
- No promotion to `Specd: ready` without direct user-confirmed readiness evidence (boundary, non-goals, tests, and risks).
- No implementation before a linked GitHub issue exists.
- One issue per ready item; do not batch multiple ready items into one issue.
- GitHub issue body must contain the full approved spec contract.
- Do not touch files outside issue-approved scope without explicit user approval.
- Do not remove a ready/spec item at issue creation without leaving an `issued` tracker line.
- Spec-gate evidence is required, but a fixed number of scripted questions is not.
- Do not skip/disable tests to force pass.
- Do not hide behavior changes behind ambiguous defaults.
- Prefer fail-fast contracts over compatibility branches.

## Skills
A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used.

### Available skills
- repo-init: Bootstrap a repo with lean planning, issue/PR workflow contracts, worktree policy, and baseline GitHub PR checks. (file: /Users/j/dotfiles/skills/repo-init/SKILL.md)
- lean-flow: Chat-first planning workflow for Brainstormed -> Specd -> ready in current.md, with strict markdown approval evidence before execution. (file: /Users/j/dotfiles/skills/lean-flow/SKILL.md)
- spec-gate: Chat-driven clarification gate for Specd readiness verification and strict contract completion. (file: /Users/j/dotfiles/skills/spec-gate/SKILL.md)
- pr-iterate: Primary execution command from ready/issued spec or PR through merge-ready feedback loops. (file: /Users/j/dotfiles/skills/pr-iterate/SKILL.md)
- issue-handoff: Internal helper that creates one issue from one ready Specd contract and compacts tracker state. (file: /Users/j/dotfiles/skills/issue-handoff/SKILL.md)
- worktree-manager: Internal helper for one-issue-per-worktree lifecycle. (file: /Users/j/dotfiles/skills/worktree-manager/SKILL.md)
- pr-scope-guard: Internal helper enforcing issue-approved file touch scope in PRs. (file: /Users/j/dotfiles/skills/pr-scope-guard/SKILL.md)
- ci-baseline: Internal helper for baseline PR checks (`lint`, `test`) and branch-protection expectations. (file: /Users/j/dotfiles/skills/ci-baseline/SKILL.md)
- partner-memory: Persistent collaboration memory for cross-repo preferences/rules. (file: /Users/j/dotfiles/skills/partner-memory/SKILL.md)
- git-cleanup: Draft commit message, confirm, then safely commit/push with a handoff packet. (file: /Users/j/dotfiles/skills/git-cleanup/SKILL.md)

### How to use skills
- Discovery: The list above is the skills available in this session (name + description + file path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, open its `SKILL.md`. Read only enough to follow the workflow.
  2) When `SKILL.md` references relative paths (e.g., `scripts/foo.py`), resolve them relative to the skill directory listed above first, and only consider other paths if needed.
  3) If `SKILL.md` points to extra folders such as `references/`, load only the specific files needed for the request; don't bulk-load everything.
  4) If `scripts/` exist, prefer running or patching them instead of retyping large code blocks.
  5) If `assets/` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue.
