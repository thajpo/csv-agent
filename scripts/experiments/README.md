Experimental and ad-hoc analysis scripts.

These are intentionally not part of the automated `tests/` suite.
Run them manually when needed.

- `investigate_failures.py`: ad-hoc failure diagnostics for triangulation output.
- `collect_prefix_values.py`: bounded real-model canary that replays one selected
  turn boundary and estimates its value from terminally verified continuations.
  It defaults to one episode and four continuations; pass the immutable
  Hugging Face dataset revision used for the input snapshot.
