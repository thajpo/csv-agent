# Value-guided selection canary

Date: 2026-07-20

## Question

Can a small learned model rank partial CSV-agent attempts on held-out CSV
datasets, and can that ranking improve which attempt is continued when random
and value-guided selection receive the same number of model calls?

## Frozen setup

- Source tasks: private Hugging Face snapshot
  `ThaJpo/csv-agent-template-episodes` at
  `e19fadf8d713c5afb7fe1476e2160b9bece1233a`.
- Collected records: private Hugging Face snapshot
  `ThaJpo/csv-agent-prefix-values-canary` at
  `8fb8e1d55133a7287d2ccdc98520fb74b845dead`.
- Collection code: `57ce0c6c89110d575466226bc7709fbca5506cd7`.
- Actor: `qwen/qwen3-30b-a3b-instruct-2507`, temperature 0.9, top-p 0.95,
  maximum 1,200 output tokens, and four total turns.
- Each question received three independently sampled first actions. The actor
  was required to execute one Python step without submitting on that first
  action.
- Each saved partial attempt received four independent continuations. The
  first three supplied the training or evaluation label; the fourth was kept
  hidden for the final selection comparison.
- CSV datasets were disjoint: four train datasets, two validation datasets,
  and two test datasets.
- The trainer was TF-IDF over the visible interaction followed by logistic
  regression. Repeated outcomes were fitted as binomial successes and
  failures. This intentionally tested the data and evaluation setup before
  introducing a neural trainer.

The collection produced 72 train, 24 validation, and 48 test partial attempts.
All 576 continuations received verifier labels; no record was excluded for a
provider, sandbox, replay, or verifier failure.

## Prediction results

| Split | Model | Log loss | Brier score | Within-question ranking |
| --- | --- | ---: | ---: | ---: |
| Validation | Average train success | 0.5668 | 0.1885 | 0.500 |
| Validation | Simple execution signals | 0.5603 | 0.1856 | 0.500 |
| Validation | Text value model | **0.4463** | **0.1379** | **0.625** |
| Test | Average train success | **0.6457** | **0.2267** | 0.500 |
| Test | Simple execution signals | 0.6481 | 0.2278 | 0.500 |
| Test | Text value model | 0.6679 | 0.2358 | **0.143** |

The text model looked useful on validation but failed on the two unseen test
datasets. Only seven unequal candidate pairs existed in the test labels, and
the text model ordered one correctly. The test result does not support a claim
that this trainer generalized across datasets.

## Equal-call selection result

Each method generated three candidate first actions and continued one. Scoring
the candidates locally did not consume an actor-model call.

- Seeded random selection: 10/16 correct, or 62.5%.
- Value-guided selection: 10/16 correct, or 62.5%.
- Oracle selection using the reserved candidate outcomes: 10/16 correct, or
  62.5%.
- Paired value-guided minus random difference: 0.0 percentage points.

The oracle result is decisive for interpreting this canary. For every test
question, all three reserved candidate outcomes agreed. Candidate choice could
not change the measured result, so no selection method had an opportunity to
beat random selection in phase 4.

## Conclusion

This is a negative and partly inconclusive result:

1. The linear text model did not generalize to the held-out CSV datasets.
2. More importantly, the chosen test tasks, first-action boundary, actor, and
   turn allowance produced almost no action-sensitive test states.
3. A larger or neural trainer would not repair the missing selection
   opportunity demonstrated by the oracle comparison.

The next experiment should change data collection before increasing trainer
complexity. Candidate actions must sometimes alter the chance of success. The
most direct levers are harder multi-step tasks, a less capable actor, fewer
remaining turns, or later decision boundaries. The held-out test results from
this run must not be used for tuning and then reported as fresh evidence.

Exact machine-readable metrics and per-question selections are stored in the
pinned Hugging Face snapshot.
