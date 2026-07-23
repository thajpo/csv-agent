# Value-guided selection canaries

This report records the exploratory first canary and the preregistered
independent replication. It intentionally keeps negative and inconclusive
results visible.

## Research question

Can a learned model rank partial CSV-agent attempts and improve which attempt
is continued, compared with expected random selection under the same deployed
actor-call allowance?

## Canary 1: exploratory Qwen study

The first canary was developed from 2026-07-20 through 2026-07-21.

### Frozen artifacts

- Source tasks: private Hugging Face snapshot
  `ThaJpo/csv-agent-template-episodes` at
  `e19fadf8d713c5afb7fe1476e2160b9bece1233a`.
- Collected records: private Hugging Face snapshot
  `ThaJpo/csv-agent-prefix-values-canary` at
  `ba6e39949798eb1918d6ec6e5a8119d74eaf8bb2`.
- Collection protocol commit:
  `b02ccf2e5bc1fc8dfe90d245860645d0bdfb64ae`.
- Actor: `qwen/qwen3-30b-a3b-instruct-2507`, temperature 0.9, top-p
  0.95, maximum 1,200 output tokens, and four total turns.
- Records: 72 train, 24 validation, and 48 test candidate states. Each
  candidate received three label continuations and one reserved continuation.
- Dataset split: four train datasets, two validation datasets, and two test
  datasets.

Candidate actions were distinct after Python AST normalization that ignores
local variable names. This is an implementation-level duplicate check, not a
claim that the actions were semantically distinct.

### Results

| Split | Model | Log loss | Brier score | Within-question ranking |
| --- | --- | ---: | ---: | ---: |
| Validation | Average train success | 0.4376 | 0.1309 | 0.500 |
| Validation | Simple execution signals | 0.4497 | 0.1370 | **1.000** |
| Validation | Text value model | **0.3409** | **0.0953** | 0.500 |
| Test | Average train success | **0.6560** | **0.2300** | **0.500** |
| Test | Simple execution signals | 0.6656 | 0.2333 | 0.333 |
| Test | Text value model | 0.6997 | 0.2431 | 0.467 |

The seeded random draw solved 10/16 questions and value-guided selection also
solved 10/16. Averaging the reserved outcomes across all candidates gives the
more relevant expected-random result of 11/16. A hindsight selector that sees
the realized reserved outcomes before choosing reaches 15/16; this is a
realized-outcome ceiling, not an oracle estimate of latent candidate values.

Results were clustered by dataset: both seeded random and guided selection
solved 3/8 water-potability questions and 7/8 diabetes questions.

### Why this is exploratory rather than confirmatory

The code and artifacts are reproducible, but the study cannot support a clean
held-out claim:

- The same two test datasets were reused while candidate-generation defects
  were diagnosed and corrected. Their outcomes therefore influenced the final
  experimental procedure.
- Two test datasets do not support meaningful across-dataset uncertainty.
- Candidate generation used three different role-specific prompts, while the
  saved state omitted those initial prompts. The tested procedure was
  strategy-conditioned proposal followed by a common continuation policy.
- The pointwise text objective could learn dataset or question difficulty even
  though the deployment decision is a within-question ranking.
- Dataset and template generalization were confounded: some test template
  names did not occur in training.
- Only seven test questions had different estimated candidate values, and only
  six had different single reserved outcomes.
- A single collection seed and one reserved Bernoulli outcome per candidate
  made the selection estimate noisy.

The defensible conclusion is narrow: this adaptively developed canary found no
benefit from its TF-IDF pointwise critic. It does not provide positive evidence
for value-guided selection, and it does not reject value functions generally.

## Canary 2: independent DeepSeek replication

Protocol frozen on 2026-07-22, before collecting any task outcomes from the
new actor.

### Hypothesis and primary endpoint

The primary hypothesis is that a within-question ranker trained on repeated
continuation outcomes will select DeepSeek V4 Flash prefixes with higher
held-out continuation success than expected random selection.

The primary endpoint is the dataset-macro-average difference:

```text
guided held-out success - mean held-out success over all candidates
```

The random term is calculated exactly from all held-out candidate outcomes;
it is not one favorable or unfavorable seeded draw. The uncertainty interval
uses a hierarchical bootstrap that resamples CSV datasets and then questions
within datasets.

This bounded canary counts as evidence of improvement only if the primary
point estimate is at least five percentage points, the 95% interval excludes
zero, and the difference is positive on at least three of four test datasets.
Otherwise improvement is not demonstrated, even if a secondary metric moves.

### Frozen data split

- Source snapshot: `ThaJpo/csv-agent-template-episodes` at
  `e19fadf8d713c5afb7fe1476e2160b9bece1233a`.
- Split seed: `20260722`.
- All eight datasets used by canary 1 are excluded.
- Train: 16 questions from each of four datasets:
  `uciml_red-wine-quality-cortez-et-al-2009`,
  `mirichoi0218_insurance`,
  `russellyates88_suicide-rates-overview-1985-to-2016`, and
  `uciml_breast-cancer-wisconsin-data`.
- Validation: eight questions from each of two datasets:
  `fedesoriano_heart-failure-prediction` and
  `pavansubhasht_ibm-hr-analytics-attrition-dataset`.
- Test: eight questions from each of four datasets:
  `neuromusic_avocado-prices`, `gregorut_videogamesales`,
  `shivamb_netflix-shows`, and
  `uciml_pima-indians-diabetes-database`.
- Every validation and test template name must occur in the selected training
  questions. This canary tests dataset transfer without simultaneously holding
  out named task templates.

Test datasets and their assignment cannot be changed after actor outcomes are
observed.

### Actor and trajectory collection

- Actor: `deepseek/deepseek-v4-flash` through OpenRouter.
- All candidates use the same system instruction, initial request, sampling
  parameters, and continuation policy. Candidate diversity comes only from
  independent sampling seeds.
- Each question receives three AST-distinct nonterminal first actions.
- Each prefix receives eight continuations. The first six form its training
  value estimate; the final two are reserved for evaluation.
- The initial boundary is after one executed Python turn.
- Temperature is 0.9, top-p is 0.95, and output is capped at 1,200 tokens.
- Terminal labels come only from the existing executable answer verifier.

The remaining-turn horizon is calibrated using train datasets only. Start at
three total turns. A calibration is usable when continuation success is
between 15% and 85% and at least 25% of questions have different candidate
labels. If success is higher, reduce the horizon by one; if lower, increase it
by one. Make at most two adjustments. If no setting is usable, stop and report
that these tasks cannot identify the value question instead of inspecting or
changing test data.

### Train/validation label audit

Before opening the test split, the six label continuations for every train and
validation candidate were audited. The two reserved continuations were excluded.
The mechanical census covered 80 questions, 240 candidate prefixes, and 1,440
label continuations:

| Outcome | Continuations | Share of all labels |
| --- | ---: | ---: |
| Accepted terminal answer | 1,072 | 74.4% |
| No terminal submission | 183 | 12.7% |
| Submitted answer rejected | 171 | 11.9% |
| Final-cell execution error | 14 | 1.0% |

All 183 no-submission outcomes ended after a successful Python execution but
without a recorded terminal answer. These are real failures under the bounded
agent protocol, although many traces had computed or printed the requested
quantity before exhausting the turn budget. The 14 execution errors included
11 JSON-boundary failures involving NumPy booleans or pandas interval keys and
three ordinary model-code errors.

The 171 rejected submissions collapsed to 38 distinct
episode-and-answer clusters. A single-reviewer semantic census classified the
continuation-weighted outcomes as follows:

| Preliminary adjudication | Rejected submissions | Interpretation |
| --- | ---: | --- |
| Clearly incorrect answer or task choice | 32 (18.7%) | Useful negative label |
| Reasonable equivalent rejected | 50 (29.2%) | Likely false negative |
| Underspecified task convention | 89 (52.0%) | Correctness cannot be inferred from the prompt alone |

The equivalent-answer cases were primarily reversed group orientation for
two-sided t-tests and Mann-Whitney tests, with the corresponding sign or
complementary U statistic, plus semantically identical quantile interval
labels. The underspecified cases comprised 78 outcomes affected by the hidden
rule that any column with at least 98% unique values is "identifier-like" and
11 outcomes using the defensible alternative of counting every member of a
duplicate group rather than only later occurrences.

This adjudication is diagnostic rather than ground truth: it was performed by
one reviewer after seeing expected and submitted answers, and the
identifier-like classification is itself a judgment. The strict lower-bound
problem is nevertheless material: even counting only mathematically
equivalent rejected answers gives 50 false-negative-looking outcomes, 29.2%
of submitted rejections. Including underspecified conventions gives 139/171.

The mechanical census is reproducible without reading test records:

```bash
uv run python scripts/experiments/audit_value_labels.py \
  --tasks-dir data/experiments/value-deepseek-replication/tasks \
  --values-dir data/experiments/value-deepseek-replication \
  --output /tmp/value-label-audit.json
```

The current canary therefore measures prediction of this procedural verifier,
including its hidden conventions and terminal protocol. It cannot, as
currently labeled, cleanly measure general CSV reasoning quality. The frozen
test evaluation may still be run once as a diagnostic of value selection under
that contract, but a positive result would not repair the label-validity
problem and a negative result would not isolate critic quality.

### Models and baselines

The primary learned selector is TF-IDF followed by pairwise logistic ranking.
Training examples are differences between candidates for the same question,
so question and dataset difficulty cannot by themselves satisfy the training
objective.

Secondary comparisons are:

- expected random selection;
- the original pointwise TF-IDF value model;
- simple execution and output-length signals; and
- an empirical-rollout selector that chooses the highest six-rollout training
  estimate, reported as a higher-compute reference rather than an equal-cost
  learned selector.

Hyperparameters and the selected checkpoint are frozen using train and
validation data. Test value records remain sealed until the checkpoint and
evaluation configuration are hashed. Test evaluation is run once.

### Boundaries

- No actor fine-tuning, PPO, process-hook rewards, causal task graphs, or task
  generator redesign.
- OpenRouter spending is capped at $15 for this replication.
- Code, configuration, actor model, source revision, value-data revision,
  seeds, checkpoint hash, and all negative results must be retained.
