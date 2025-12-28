# csv-spec

Type contracts for csv-agent: schemas, normalization, parsing.

## Purpose

This package defines the **contract** between the environment and trainer:

- **Types**: What actions and results look like (`ActionSpec`, `StepResult`, `EpisodeJSONL`)
- **Normalization**: How to canonicalize values for comparison (`normalize_value`)
- **Hashing**: How to hash values for verification (`hash_artifact`)
- **Parsing**: How to extract actions from model output (`parse_action`, `parse_step_result`)

## Contract Guarantee

If you change anything here, you **MUST** update both:
1. **Environment** (`csv_env.py`) - how it parses/validates
2. **Trainer** (`rl_env.py`, prompts) - how it formats/consumes

## Usage

```python
from csv_spec import (
    # Types
    EpisodeJSONL,
    ActionSpec,
    CodeAction,
    StepResult,
    HookDict,
    
    # Functions
    normalize_value,
    hash_artifact,
    parse_action,
    parse_step_result,
)

# Parse model output into action
action = parse_action(model_output)
if isinstance(action, CodeAction):
    # Execute code...
    pass

# Parse execution result
result = parse_step_result(stdout, stderr)
if result.terminal:
    # Episode done
    pass
```

## Version

Changes to this package require a version bump. The version is checked at import time
to ensure environment and trainer are compatible.
