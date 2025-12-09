# Refactor Issues & Concerns

## Performance

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| Inefficient `get_final_answer()` | [kernel.py:247-255](src/kernel.py#L247-L255) | Medium | Serializes entire namespace to get one variable. Called after every cell execution. |
| No parallelization | [teacher.py:119-163](src/teacher.py#L119-L163) | Low | Each trace runs sequentially. For batch triangulation, could run N consistency traces in parallel. |

### Fix for `get_final_answer()`:
```python
def get_final_answer(self):
    """Get submitted answer without serializing entire namespace."""
    result = self.execute("__SUBMITTED_ANSWER__")
    if result.success and result.result:
        # Parse the repr output
        return eval(result.result)  # Safe here - we control the kernel
    return None
```

---

## Correctness

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| False positive artifact matches | [kernel.py:183-213](src/kernel.py#L183-L213) | **High** | `df`, `pd`, `np`, `submit` are captured as artifacts. `df` will always match between teacher/student, inflating dense reward. |
| No baseline exclusion | [kernel.py:215-245](src/kernel.py#L215-L245) | **High** | Should track variables present after kernel setup and exclude them from artifact snapshots. |
| Stateful submit detection | [environment.py:240-243](src/environment.py#L240-L243) | Low | `__SUBMITTED_ANSWER__` persists across cells. Current code handles it but fragile if logic changes. |

### Recommended fix: Baseline snapshot

Add to `JupyterKernel.__init__()`:
```python
# After setup_kernel_builtins()
self.baseline_vars = set(self.get_locals().keys())
```

Update `snapshot_artifacts()`:
```python
def snapshot_artifacts(self) -> dict:
    """Capture only USER-CREATED DataFrames and scalars."""
    from src.types import Artifact, hash_artifact
    import pandas as pd

    locals_dict = self.get_locals()
    artifacts = {}

    for name, obj in locals_dict.items():
        # Skip baseline variables (df, pd, np, submit, etc.)
        if name in self.baseline_vars:
            continue

        if name.startswith('_'):
            continue

        if isinstance(obj, pd.DataFrame):
            artifacts[name] = Artifact(
                name=name,
                hash=hash_artifact(obj),
                type='DataFrame'
            )
        elif isinstance(obj, (int, float, str, bool, type(None))):
            artifacts[name] = Artifact(
                name=name,
                hash=hash_artifact(obj),
                type='scalar'
            )

    return artifacts
```

---

## Missing Functionality

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| No execution vs verification distinction | [teacher.py:166-174](src/teacher.py#L166-L174) | Medium | Can't tell if triangulation failed because question is ambiguous vs. code crashed. |
| Episode student fields unpopulated | [types.py:105-117](src/types.py#L105-L117) | Expected | `student_trace`, `intermediate_matches`, rewards are empty. Phase 6 concern. |
| No episode loading utility | - | Low | Can save episodes but no `load_episode()` for training loop. |

### Recommended: Add execution tracking

In `triangulate_teacher()`, track execution failures:
```python
# After consistency traces loop
n_succeeded = sum(1 for t in consistency_traces if t.execution_success)
n_failed = len(consistency_traces) - n_succeeded

if n_failed > 0:
    logger.warning("consistency_execution_failures", extra={
        "n_failed": n_failed,
        "n_succeeded": n_succeeded
    })

if n_succeeded == 0:
    # All traces failed - model/code issue, not question issue
    return gold_trace, consistency_traces, False
```

### Recommended: Episode loading utility

Add to new file `src/episode_utils.py`:
```python
import json
from pathlib import Path
from src.types import Episode

def load_episode(filepath: str | Path) -> Episode:
    """Load episode from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return Episode(**data)

def load_episodes_batch(episodes_dir: str | Path, verified_only: bool = True) -> list[Episode]:
    """Load all episodes from directory."""
    episodes = []
    for filepath in Path(episodes_dir).glob("*.json"):
        episode = load_episode(filepath)
        if verified_only and not episode.verified:
            continue
        episodes.append(episode)
    return episodes
```

---

## Design Questions (Need Your Input)

| Question | Context |
|----------|---------|
| Training approach? | RLHF/PPO? Supervised fine-tuning on teacher traces? Affects how rewards are used. |
| Student sees hint? | `STUDENT_PROMPT` has no hint. Intentional? |
| How to exclude `df` from matching? | Baseline snapshot? Explicit exclusion list? |
| Multi-dataset support? | Current code assumes single CSV. Future concern? |

---

## Minor/Cleanup

| Issue | Location | Description |
|-------|----------|-------------|
| Legacy types still present | [types.py:123-157](src/types.py#L123-L157) | `HookParams`, `Hook`, `QuestionGeneration`, `Answer` - marked deprecated but still there. |
| `target_questions` unused | [main.py:32](src/main.py#L32), [config.yaml:28](config.yaml#L28) | Legacy param from old question-gen mode. |
| Test file in root | [test_triangulation.py](test_triangulation.py) | Should move to `tests/` or delete after confirming triangulation works. |

### Recommended cleanup:
```bash
# Remove legacy types
# Edit src/types.py to delete lines 123-157

# Remove target_questions
# Edit config.yaml to remove line 27-28
# Edit src/main.py to remove references

# Move test
mkdir -p tests
mv test_triangulation.py tests/
```

---

## Priority Order (If Fixing Now)

1. **Baseline artifact exclusion** (correctness - HIGH)
   - Prevents `df` false positives in scavenger hunt rewards

2. **Efficient `get_final_answer()`** (performance - MEDIUM)
   - Reduces overhead on every code cell execution

3. **Execution vs verification tracking** (observability - MEDIUM)
   - Helps debug whether triangulation failures are model issues vs question issues

4. **Episode loading utility** (convenience - LOW)
   - Will be needed for Phase 6 anyway

5. **Cleanup legacy code** (maintainability - LOW)
   - Low priority, doesn't affect correctness
