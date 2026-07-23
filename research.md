# Research Direction

Last reviewed: 2026-07-22

This document records the project's selected research direction and durable
rationale. It is not a roadmap or task tracker.

## Goal

Study whether execution-grounded learning can help small CSV agents make
reliable computational progress, recover from mistakes, and generalize across
unfamiliar tables and tasks.

CSV analysis is a useful controlled domain because Python provides real state
transitions, notebook states can potentially be replayed, and many tasks can
have privately executable terminal verifiers.

## Current Diagnosis

- Distillation is useful infrastructure and a baseline, not the central
  research contribution.
- Existing procedural tasks largely begin from canonical operation chains.
  Those chains do not establish that the observed data made each action useful.
- Hook matching can reward imitation or spam and exclude alternate valid paths.
- A generator-authored dependency graph is not the causal graph of every valid
  solver trajectory.
- Current tasks underrepresent exploration, dead ends, and recovery.
- Terminal correctness can often be verified exactly; whether an intermediate
  action constituted useful reasoning generally cannot.
- Issue #34 and PR #35 describe heuristic process reports, not rollout-derived
  value learning. Their future should be decided separately.

## Research Program

### Primary: Execution-Aware Value Models

> Can a model estimate which executed notebook states are promising and use
> those estimates to improve planning at a fixed execution budget?

The initial decision boundary is one complete model turn followed by Python
execution:

```text
notebook state -> proposed Python cell -> execution result -> new notebook state
```

The working model is:

- A state includes the question, interaction history, executed code, outputs or
  errors, relevant notebook artifacts, and remaining budget.
- An action is a model response that may execute one Python cell.
- Clone an intermediate state and sample several continuations.
- The fraction ending correctly estimates that state's future-success value
  under the recorded continuation policy, horizon, and budget.
- A candidate progress signal is the change in value across an action, not
  agreement with a prescribed reasoning chain.
- Exploration is judged by its effect on future success, not by whether it
  appears in a private task graph.

Terminal correctness grounds the value target; it is not the research claim
and does not commit the project to outcome-only actor training.

The first evidence gate should freeze the actor and test whether a learned
critic ranks candidate next actions on held-out tasks better than random
selection, self-consistency, direct rollout selection, and available process
heuristics at equal execution cost. Actor training comes later only if the
critic demonstrates useful calibration and branch ranking.

This direction should be reconsidered if tasks are effectively single-step,
states cannot be replayed faithfully, rollout values contain little variation,
or value-guided selection does not improve held-out success at equal compute.

### Current status: paused after two canaries

The value infrastructure now supports replayed prefixes, repeated continuation
labels, dataset-disjoint splits, frozen critics, and equal-call selection
evaluation. Neither canary demonstrated useful value guidance:

- The exploratory Qwen critic selected 10/16 successful continuations versus
  11/16 expected under uniform selection.
- The independent DeepSeek pairwise critic reached 73.44% success versus
  70.31% expected random selection. The 3.12-point difference had a 95%
  interval of -6.25 to 11.98 points and failed every preregistered evidence
  gate.

The main blocker is now label validity. In the DeepSeek train/validation audit,
50/171 rejected submissions were reasonable statistical equivalents and 89
depended on underspecified identifier or duplicate-count conventions. The
current evidence cannot isolate critic quality from verifier noise and does not
justify a larger critic, actor training, or PPO.

Resume value research only after an explicit, equivalence-aware terminal
contract passes a pre-test label audit. That work should create a new dataset
revision and fresh canary rather than reinterpret the frozen results.

Continuation sources:

- [Detailed methods and results](docs/research/value-canary-2026-07-20.md)
- [Pinned DeepSeek snapshot](configs/value/deepseek-canary.toml)
- [Implementation and review history](https://github.com/thajpo/csv-agent/pull/39)

### Parallel: Data-Conditioned Procedural Tasks

> Can we generate broad, verifiable CSV environments in which the observed
> data determines the useful analytical behavior and fixed heuristics fail?

The working direction is:

- Construct a private condition or latent problem world.
- Materialize a visible CSV and natural question about it.
- Compute or verify the terminal answer with trusted code.
- Permit any solver trajectory that satisfies the verifier.
- Use contrastive instances where changing the data changes the useful
  behavior while surface form remains as stable as practical.

This track can proceed independently and later provide broader environments for
value research. Existing tasks are sufficient for an initial value-methodology
test only if they produce meaningful multi-turn success, failure, exploration,
and recovery. They are not sufficient evidence of broad CSV reasoning.

## Shared Evaluation Principles

- Freeze evaluations before inspecting training results.
- Separate held-out datasets from held-out task structures or compositions.
- Compare interventions on identical tasks and execution budgets.
- Report multiple seeds, uncertainty, slice regressions, and negative results.
- Record the code commit, configuration, model, dataset revision, and seeds.
- Do not infer reasoning quality from agreement with one canonical trace.

## Research Context

Public critic-free reasoning recipes show that group-relative optimization can
be effective and economical on independent responses. They do not establish
that value functions are unhelpful for interactive agents. GLM-5.2, for
example, moved to critic-based PPO for variable-length, long-horizon agent
traces.

Relevant precedents and cautions include
[Outcome-supervised Value Models](https://arxiv.org/abs/2311.09724),
[Rewarding Progress](https://arxiv.org/abs/2410.08146),
[VinePPO](https://arxiv.org/abs/2410.01679),
[AgentPRM](https://arxiv.org/abs/2502.10325), the official
[GLM-5.2 report](https://z.ai/blog/glm-5.2), and
[DataPRM](https://arxiv.org/abs/2604.24198).

The repository's narrower question is whether structured Python execution
states make intermediate value sufficiently learnable, grounded, and useful to
justify its cost.

## Open Research Questions

- Can notebook states be checkpointed and replayed faithfully?
- What state representation is sufficient without exposing the private oracle?
- Are existing tasks sufficiently multi-turn for values to be nontrivial?
- How many continuations produce useful labels at affordable cost?
- When do learned values outperform direct rollout selection?
- Which task families genuinely test exploration, recovery, and
  data-conditioned choices?

## Documentation Contract

- `research.md` records the selected direction and durable conclusions.
- Chat contains unselected brainstorming.
- GitHub issues define bounded experiments and their success or stop criteria.
- PRs and code implement those experiments.
- Experiment artifacts preserve evidence and prompt updates here when the
  research direction changes.
