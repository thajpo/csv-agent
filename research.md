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

### First canary result

The first value-guided selection canary completed on 2026-07-21. Its code and
artifacts reproduce, but the study is exploratory rather than a clean held-out
test. The two test datasets were reused while candidate-generation defects were
fixed, candidate proposals came from different role prompts, the pointwise
training objective did not match the within-question deployment decision, and
two datasets cannot support useful across-dataset uncertainty.

The TF-IDF critic ranked test candidate pairs at 0.467. It selected a successful
reserved continuation for 10/16 questions, compared with 11/16 expected for
uniform random selection over the same realized candidate outcomes. A seeded
random draw happened to score 10/16. A hindsight selector that sees each
reserved outcome before choosing scored 15/16, but that realized-outcome
ceiling does not establish different latent candidate values.

The defensible conclusion is that this adaptively developed canary found no
benefit from its shallow pointwise critic. It provides no positive evidence for
value guidance and does not reject value functions generally. An independent
DeepSeek V4 Flash replication is preregistered in
`docs/research/value-canary-2026-07-20.md`; it uses fresh datasets, one candidate
policy, a within-question ranking objective, exact expected-random comparison,
and dataset-clustered uncertainty. Actor training remains premature.

The replication's train/validation label audit exposed a prior validity gate.
Of 171 submitted answers rejected by the procedural verifier, only 32 were
clearly incorrect in a full review of the 38 distinct answer clusters. Fifty
were reasonable statistical or interval-label equivalents, and 89 depended on
underspecified identifier or duplicate-count conventions. Value learning from
terminal outcomes is only as meaningful as the terminal contract: future
canaries must audit task/verifier agreement before treating rollout success as
a reasoning-quality target. The current frozen replication can still diagnose
selection under its recorded verifier, but cannot cleanly support a claim about
general CSV reasoning.

The frozen DeepSeek test was subsequently evaluated once. The pairwise critic
reached 73.44% held-out success against 70.31% expected random selection, a
3.12-point difference with a dataset-clustered 95% interval from -6.25 to 11.98
points. It improved two of four datasets and ranked unequal candidate pairs at
53.49%. This failed every preregistered evidence gate, so improvement was not
demonstrated. The current evidence does not justify actor training or a larger
critic; terminal-contract validity should be repaired and measured first.

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
