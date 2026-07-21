# Value-guided selection canary

Started on 2026-07-20 and completed on 2026-07-21.

## Question

Can a small learned model rank partial CSV-agent attempts on held-out CSV
datasets, and can that ranking improve which attempt is continued when random
and value-guided selection receive the same number of actor-model calls?

## Frozen setup

- Source tasks: private Hugging Face snapshot
  `ThaJpo/csv-agent-template-episodes` at
  `e19fadf8d713c5afb7fe1476e2160b9bece1233a`.
- Collected records and machine-readable results: private Hugging Face snapshot
  `ThaJpo/csv-agent-prefix-values-canary` at
  `ba6e39949798eb1918d6ec6e5a8119d74eaf8bb2`.
- Collection protocol commit:
  `b02ccf2e5bc1fc8dfe90d245860645d0bdfb64ae`.
- Actor: `qwen/qwen3-30b-a3b-instruct-2507`, temperature 0.9, top-p 0.95,
  maximum 1,200 output tokens, and four total turns.
- Each question received three role-diversified first Python actions: a
  prerequisite/check, a necessary computation, and a direct partial
  result/check. They remained distinct after normalizing local variable names,
  and none could submit the answer on that turn.
- Each saved state received four continuations. The first three supplied the
  value label; the fourth was reserved for the equal-call selection test.
- CSV datasets were disjoint: four train datasets, two validation datasets,
  and two test datasets.
- Every record stored the exact collection contract, including its source-input
  hash, code commit, policy, branching settings, tolerance, and split seed.
- The critic was TF-IDF over the visible interaction followed by logistic
  regression. Repeated outcomes were fitted as binomial successes and
  failures. This tests the data and evaluation method before a neural critic.

The final collection contains 72 train, 24 validation, and 48 test states.
Every question has three semantically distinct executed actions, and all 576
continuations received terminal-verifier labels. No record was excluded for a
provider, sandbox, replay, or verifier failure.

The action labels varied within 12 of 24 training questions, 1 of 8 validation
questions, and 7 of 16 test questions. Six test questions had different
reserved outcomes across candidates, so candidate choice could affect phase 4.

## Prediction results

| Split | Model | Log loss | Brier score | Within-question ranking |
| --- | --- | ---: | ---: | ---: |
| Validation | Average train success | 0.4376 | 0.1309 | 0.500 |
| Validation | Simple execution signals | 0.4497 | 0.1370 | **1.000** |
| Validation | Text value model | **0.3409** | **0.0953** | 0.500 |
| Test | Average train success | **0.6560** | **0.2300** | **0.500** |
| Test | Simple execution signals | 0.6656 | 0.2333 | 0.333 |
| Test | Text value model | 0.6997 | 0.2431 | 0.467 |

The text critic improved validation probability metrics, but validation had
only two unequal candidate pairs and provided no ranking advantage. On the two
unseen test datasets it was worse than the train-success baseline on log loss,
Brier score, calibration, and ranking. Its five-bin calibration error was
0.1438, versus 0.0880 for the train-success baseline. The critic did not learn
a reliable held-out success estimate or ordering.

## Equal-call selection result

Each method generated three candidate first actions and continued one reserved
trajectory. Local critic scoring consumed no actor-model call.

- Seeded random selection: 10/16 correct, or 62.5%.
- Value-guided selection: 10/16 correct, or 62.5%.
- Oracle selection using reserved outcomes: 15/16 correct, or 93.75%.
- Paired guided-minus-random difference: 0 percentage points.
- Paired bootstrap 95% interval: -18.75 to +18.75 percentage points.

The critic improved one question that random selection missed and harmed one
question that random selection solved. The oracle gap proves that this test had
real selection opportunity; unlike the invalid first attempt, the tie cannot be
explained by identical candidate states or identical reserved outcomes.

## Conclusion

This is a valid negative result for the tested linear text critic:

1. A shallow text representation improved validation probability metrics, but
   neither its probabilities nor its rankings generalized to unseen datasets.
2. It did not improve final correctness over seeded random selection at equal
   actor-model calls.
3. The result does not show that value functions are useless. It covers one
   actor, one decision boundary, two test datasets, one fixed collection per
   split, and a small linear critic.
4. Actor training is premature. The next value-model experiment should first
   explain the held-out misrankings and justify a better state representation
   or broader label set.

## Invalidated preliminary runs

The first supposedly corrected snapshot at
`c760ae8749272c9438ab218d1578f6520717d63a` was also invalidated during the
final audit. One test question contained two semantically identical missing
percentage actions that differed only in a local variable name, and its records
predated the exact collection contract. Its downstream numbers reproduced, but
the snapshot did not satisfy the experiment's evidence contract.

An earlier snapshot at `8fb8e1d55133a7287d2ccdc98520fb74b845dead`
was invalidated during audit. All three supposed candidates for every question
executed the same first action, usually `print(df.columns.tolist())`; their
different labels came only from stochastic continuations. Neither invalidated
snapshot is evidence about action selection; both have been replaced by the
pinned snapshot above.
