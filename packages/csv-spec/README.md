# csv-spec

Type contracts for `csv-agent` question and episode artifacts.

This package contains shared schema models and normalization helpers used by
the main application and tests.

## Episode verification fields

`EpisodeJSONL.question` is the canonical source for terminal evaluation. Its
`ground_truth` field is the value used for tolerant comparisons, and
`ground_truth_hash` or the non-empty `ground_truth_hashes` list records the
terminal-verifier provenance. Evaluators compare a submitted answer with every
accepted hash and fail before rollout when no hash provenance is present.

## Prefix-value contracts

The package exports four Pydantic models for the execution-aware value
experiment:

- `TrajectoryPrefix`: exact public agent state at a nonterminal completed-turn
  boundary. Turns must be contiguous and zero-indexed, responses and completion
  flags must align with them, every completion flag must be false, at least one
  continuation turn must remain, and the conversation must end with the
  recorded assistant response plus user execution feedback. Expected answers
  and their hashes are deliberately absent. `consumed_turns` counts every actor
  response against the horizon, including format-invalid responses that did not
  execute code, so it can exceed the number of recorded execution turns.
- `ContinuationPolicy`: actor model and frozen sampling arguments that define
  what a collected value means.
- `PrefixContinuation`: one seeded continuation, including its trace and
  terminal-verifier verdict, or an error when replay, rollout, or verification
  could not produce a label.
- `PrefixValueRecord`: the prefix, policy, continuation evidence, provenance,
  and validated aggregate counts. `value` is successful continuations divided
  by attempted continuations only when every attempt received a verifier label.
  Otherwise it is `None`, and `labeled_continuations` exposes the missing
  coverage without treating infrastructure failure as policy failure.

All four models reject unknown fields. `PrefixValueRecord` also rejects counts
or values that do not exactly match its continuation records, making serialized
JSONL output self-checking on reload.
