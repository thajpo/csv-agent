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
  `c760ae8749272c9438ab218d1578f6520717d63a`.
- Collection protocol commit:
  `9ae91e184378f8422cc5b50ffdf7561a437affc3`.
- Actor: `qwen/qwen3-30b-a3b-instruct-2507`, temperature 0.9, top-p 0.95,
  maximum 1,200 output tokens, and four total turns.
- Each question received three role-diversified, AST-distinct first Python
  actions: a prerequisite/check, a necessary computation, and a direct partial
  result/check. None could submit the answer on that turn.
- Each saved state received four continuations. The first three supplied the
  value label; the fourth was reserved for the equal-call selection test.
- CSV datasets were disjoint: four train datasets, two validation datasets,
  and two test datasets.
- The critic was TF-IDF over the visible interaction followed by logistic
  regression. Repeated outcomes were fitted as binomial successes and
  failures. This tests the data and evaluation method before a neural critic.

The final collection contains 72 train, 24 validation, and 48 test states.
Every question has three distinct executed actions, and all 576 continuations
received terminal-verifier labels. No record was excluded for a provider,
sandbox, replay, or verifier failure.

The action labels varied within 11 of 24 training questions, 3 of 8 validation
questions, and 8 of 16 test questions. Seven test questions had different
reserved outcomes across candidates, so candidate choice could affect phase 4.

## Prediction results

| Split | Model | Log loss | Brier score | Within-question ranking |
| --- | --- | ---: | ---: | ---: |
| Validation | Average train success | 0.4755 | 0.1490 | 0.500 |
| Validation | Simple execution signals | 0.4963 | 0.1563 | 0.417 |
| Validation | Text value model | **0.4242** | **0.1325** | **0.667** |
| Test | Average train success | 0.6117 | 0.2088 | 0.500 |
| Test | Simple execution signals | **0.6004** | **0.2049** | 0.594 |
| Test | Text value model | 0.6561 | 0.2199 | **0.625** |

The text critic again looked useful on validation. On the two unseen test
datasets, it ranked unequal pairs somewhat better than the baselines but had
worse probability error. Its five-bin calibration error was 0.1373, versus
0.0718 for the train-success baseline. The critic learned some ordering signal
but not a reliable estimate of success probability.

## Equal-call selection result

Each method generated three candidate first actions and continued one reserved
trajectory. Local critic scoring consumed no actor-model call.

- Seeded random selection: 11/16 correct, or 68.75%.
- Value-guided selection: 11/16 correct, or 68.75%.
- Oracle selection using reserved outcomes: 15/16 correct, or 93.75%.
- Paired guided-minus-random difference: 0 percentage points.
- Paired bootstrap 95% interval: -18.75 to +18.75 percentage points.

The critic improved one question that random selection missed and harmed one
question that random selection solved. The oracle gap proves that this test had
real selection opportunity; unlike the invalid first attempt, the tie cannot be
explained by identical candidate states or identical reserved outcomes.

## Conclusion

This is a valid negative result for the tested linear text critic:

1. A shallow text representation can pick up some within-question ordering
   signal, but its probabilities do not generalize to the unseen datasets.
2. That signal did not improve final correctness over seeded random selection
   at equal actor-model calls.
3. The result does not show that value functions are useless. It covers one
   actor, one decision boundary, two test datasets, one collection seed, and a
   small linear critic.
4. Actor training is premature. The next value-model experiment should first
   explain the held-out misrankings and justify a better state representation
   or broader label set.

## Invalidated preliminary run

An earlier snapshot at `8fb8e1d55133a7287d2ccdc98520fb74b845dead`
was invalidated during audit. All three supposed candidates for every question
executed the same first action, usually `print(df.columns.tolist())`; their
different labels came only from stochastic continuations. Its metrics are not
evidence about action selection and have been replaced by the pinned snapshot
above.
