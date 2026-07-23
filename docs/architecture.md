# Architecture

`csv-agent` has two related paths: verified episode generation and experimental
value-guided action selection.

## Data generation

```text
Kaggle or local CSV
  -> template, procedural, or LLM question generation
  -> model writes Python
  -> code executes in a Docker sandbox
  -> terminal answer is verified
  -> raw episode trace is saved
  -> SFT, DPO, PRM, or evaluation formats are derived later
```

The raw episode is the reusable artifact. Training-specific formats should not
be baked into generation.

## Prefix-value experiment

```text
verified episode + private expected answer
  -> actor samples a nonterminal first action
  -> action is replayed in a fresh sandbox
  -> actor samples several continuations from that exact state
  -> terminal verifier labels each continuation
  -> prefix and outcomes become a PrefixValueRecord
  -> local ranker scores candidate first actions
  -> sealed outcomes compare guided and expected-random selection
```

The critic sees only public state: CSV metadata, question, system prompt,
conversation, executed code and output, and turns remaining. Expected answers,
answer hashes, continuation traces, and verifier verdicts are excluded from its
input. A prefix value is tied to its actor model, sampling settings, horizon,
seeds, Git commit, and dataset revision.

Hooks remain trace diagnostics. They are not trusted process labels and do not
control the terminal value target.

## Code boundaries

- `src/core/model.py`: OpenRouter/OpenAI-compatible model client.
- `src/core/environment.py`: conversation and rollout orchestration.
- `src/envs/csv_env.py`: persistent Docker sandbox and execution protocol.
- `src/datagen/`: question generation, traces, triangulation, and verification.
- `src/value/`: prefix construction, replay, continuation labels, and critic
  examples.
- `packages/csv-spec/`: canonical episode and prefix-value schemas.
- `scripts/evaluate_model.py`: small model-evaluation entry point.
- `scripts/experiments/`: bounded value collection, training, and evaluation.
- `configs/`: immutable dataset references and experiment configuration.

## Reliability boundaries

- The default test suite never makes live model calls.
- Model code runs in Docker rather than the host Python process.
- Prefix replay fails if recorded execution cannot be reproduced.
- Training, validation, and test are split by CSV dataset.
- Test continuations remain sealed until checkpoints and hashes are frozen.
- Generated data is ignored by Git and authoritative snapshots use immutable
  Hugging Face revisions.

## Current limitation

The machinery can reproducibly estimate success under the existing terminal
verifier, but the verifier rejects some mathematically equivalent answers and
encodes some conventions that are absent from the prompt. That makes label
validity—not model scale—the next research boundary. The concise project status
is in [research.md](../research.md); the complete canary record is in
[docs/research/value-canary-2026-07-20.md](research/value-canary-2026-07-20.md).
