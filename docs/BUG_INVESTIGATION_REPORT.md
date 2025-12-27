# Pipeline Bug Investigation Report

**Date:** 2025-12-27
**Investigated by:** Claude (autonomous analysis)

## Executive Summary

After thorough investigation of the csv-agent pipeline, I identified **17 potential bugs and issues** across 6 categories. The most critical issues relate to **silent failure handling** and **Docker container lifecycle management**.

---

## 1. Docker Container Utilization Issues

### 1.1 Orphaned Container Accumulation (HIGH)
**Location:** `src/envs/container_pool.py:898-911`

```python
async def stop(self):
    """Stop all containers in the pool."""
    stop_tasks = [c.stop() for c in self._containers]
    await asyncio.gather(*stop_tasks, return_exceptions=True)  # ← Exceptions silently ignored
```

**Problem:** When `return_exceptions=True` is used, failed container stops are silently swallowed. If a container fails to stop (network issue, Docker daemon hiccup), it becomes orphaned.

**Impact:** Container leaks accumulate over multiple pipeline runs, consuming resources.

**Fix suggestion:** Log exceptions from container stop operations.

---

### 1.2 No Container Health Checks (MEDIUM)
**Location:** `src/envs/container_pool.py` and `src/envs/csv_env.py`

**Problem:** Workers can die mid-execution (OOM, Python crash) with no detection mechanism. The host continues waiting on FIFO reads that will never complete, eventually timing out.

**Impact:** Slow failure detection, wasted time waiting for dead workers.

---

### 1.3 Worker Process Zombie Detection Missing (MEDIUM)
**Location:** `src/envs/container_pool.py:189-339` (worker script)

**Problem:** If a forked worker process exits unexpectedly, the parent process continues running but that worker slot is dead. No mechanism exists to detect or replace dead workers.

---

## 2. Silent Failure Patterns

### 2.1 Hook Parsing Silently Ignores Errors (HIGH)
**Location:** `src/datagen/teacher.py:36-58`

```python
def parse_hooks_from_stdout(stdout: str) -> list[HookDict]:
    for line in stdout.split("\n"):
        if "📍 Hook:" in line:
            try:
                hook_data = json.loads(line[json_start:])
                # ...
            except json.JSONDecodeError:
                continue  # ← Silent failure - malformed hooks lost
```

**Problem:** Malformed hook JSON is silently skipped. If model output formatting drifts, hooks silently disappear.

**Impact:** Hook count validation may pass with fewer hooks than expected if some were malformed.

---

### 2.2 Answer Parsing Fallback Chain (MEDIUM)
**Location:** `src/core/environment.py:96-124`

```python
def parse_submitted_answer(output: str) -> str | None:
    try:
        return json.loads(answer_str)
    except (json.JSONDecodeError, ValueError):
        try:
            return ast.literal_eval(answer_str)  # ← Silent fallback
        except (ValueError, SyntaxError):
            return answer_str  # ← Returns raw string silently
```

**Problem:** Three-tier fallback without logging. If JSON parsing fails but ast.literal_eval works, no indication is given. Raw string fallback could mask protocol violations.

---

### 2.3 Metadata Load Failures Silently Continue (LOW)
**Location:** `src/datagen/episode_gen.py:131-140`

```python
try:
    with open(meta_path) as f:
        meta_data = json.load(f)
except Exception as e:
    ui.base.print_warning(f"Failed to read metadata from {meta_path}: {e}")
    # Continues with dataset_description = None
```

**Problem:** Warning is printed but the dataset is still processed without description, which may affect question quality.

---

## 3. Data Flow Issues

### 3.1 Turn/Execution Result Alignment Not Validated (HIGH)
**Location:** `src/datagen/teacher.py:68-128`

```python
def build_trace_dict(final_state, conversation_messages):
    for turn_idx, assistant_msg in enumerate(assistant_messages):
        exec_results = (
            execution_results_per_turn[turn_idx]
            if turn_idx < len(execution_results_per_turn)  # ← Silent mismatch
            else []
        )
```

**Problem:** If assistant message count doesn't match execution results count, empty execution results are silently used. No warning or error logged.

**Impact:** Traces may have turns with code but no execution data, corrupting training data.

---

### 3.2 Difficulty Filter Returns Empty Without Clear Indicator (MEDIUM)
**Location:** `src/datagen/episode_gen.py:77-101`

```python
def filter_by_difficulty(questions, distribution, total_target):
    for difficulty, fraction in distribution.items():
        count_needed = round(total_target * fraction)
        matching = [q for q in questions if q.get("difficulty") == difficulty]
        if len(matching) < count_needed:
            return [], False  # ← Empty list returned
```

**Problem:** Returns `([], False)` but caller only logs a warning and skips. No structured tracking of which difficulties were insufficient.

---

### 3.3 normalize_value Injection is Fragile (MEDIUM)
**Location:** `src/envs/csv_env.py:110-126`

```python
def get_setup_code() -> str:
    source = inspect.getsource(normalize_value)
    source = source.replace(') -> Any:', '):')  # ← String manipulation
    source = source.replace('val: Any', 'val')
```

**Problem:** Uses `inspect.getsource()` and string replacement to inject normalize_value into containers. If normalization.py adds imports or type hints change, this breaks silently.

**Impact:** Container and host could have different normalization logic, causing answer mismatches.

---

## 4. Contract & Coupling Issues

### 4.1 answers_match Has Too Many Special Cases (HIGH)
**Location:** `src/datagen/teacher.py:131-266`

The `answers_match` function is 135 lines with:
- Float tolerance comparison
- DataFrame sorting and comparison
- Dict with special keys (answer, p_value)
- Recursive calls for nested structures
- Multiple fallback paths

**Problem:** Complex branching logic is hard to test exhaustively. Edge cases may cause false positives/negatives in answer matching.

---

### 4.2 Code Block Regex Doesn't Handle Edge Cases (LOW)
**Location:** `src/datagen/teacher.py:84`

```python
code_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
```

**Problem:** Requires `\n` after `python`. Variations like:
- ` ```python  ` (trailing space)
- ```` ```python3\n ```` (python3 variant)
- ```` ```Python\n ```` (capitalized)

...would not match.

---

### 4.3 Hook Grounding Uses Substring Match (MEDIUM)
**Location:** `src/core/environment.py:32-69`

```python
if normalized_code_line in normalized_all_code:
    grounded.append(hook)
```

**Problem:** Substring matching can cause false positives. If code is `x = 1` and later `x = 10`, the code_line `x = 1` matches both.

---

## 5. Functions Doing Too Much

### 5.1 Environment.process_turn (HIGH)
**Location:** `src/core/environment.py:439-540`

This 100-line method handles:
1. Code extraction
2. Code execution
3. Feedback building
4. Format validation with retry logic
5. Hook validation with retry logic
6. Completion checking
7. Conversation management

**Recommendation:** Split into smaller, testable functions.

---

### 5.2 batch_triangulate (MEDIUM)
**Location:** `src/datagen/teacher.py:608-789`

180-line function handling:
- Container lifecycle management
- Question slot allocation
- Parallel task execution
- Progress tracking
- Result aggregation

---

### 5.3 triangulate_teacher Returns 8-Tuple (LOW)
**Location:** `src/datagen/teacher.py:425-605`

```python
def triangulate_teacher(...) -> tuple[
    TraceDict, list[dict], str, list[tuple[TraceDict, list[dict]]],
    bool, dict, str | None, int
]:
```

**Problem:** 8-element tuple return type is a code smell. Hard to remember order, easy to mishandle.

**Recommendation:** Return a dataclass or TypedDict instead.

---

## 6. Model Output Consistency

### 6.1 No Validation That Model Uses Current Dataset (MEDIUM)
**Location:** Various prompts in `src/core/prompts.py`

**Problem:** No runtime verification that model's code references columns that exist in the current dataset. If model hallucinates column names from training data, execution fails but the error isn't specifically flagged as "wrong dataset" vs "code bug".

---

### 6.2 Ground Truth Hash Not Validated at Episode Generation (LOW)
**Location:** `src/datagen/episode_gen.py:268-301`

Synthetic questions include `ground_truth_hash` but:
- Not compared against gold trace answer hash
- Could detect teacher divergence from expected answer

---

## Profiling Script Created

A diagnostic tool was created at `scripts/pipeline_profiler.py`:

```bash
# Check for orphaned containers
uv run python scripts/pipeline_profiler.py containers

# Analyze episode data quality
uv run python scripts/pipeline_profiler.py episodes

# Validate hook grounding
uv run python scripts/pipeline_profiler.py hooks

# Analyze timing distributions
uv run python scripts/pipeline_profiler.py timing

# Detect silent failure patterns in source
uv run python scripts/pipeline_profiler.py silent

# Run all checks
uv run python scripts/pipeline_profiler.py all
```

---

## Priority Recommendations

### Immediate (High Impact, Easy Fix)
1. Add logging to `parse_hooks_from_stdout` when JSON parsing fails
2. Log container stop failures in `ContainerPool.stop()`
3. Validate turn count matches execution result count in `build_trace_dict`

### Medium Term
4. Refactor `Environment.process_turn` into smaller functions
5. Replace 8-tuple return with dataclass in `triangulate_teacher`
6. Add worker health checks to container pool

### Long Term
7. Consider structured logging (not print) throughout pipeline
8. Add integration tests for edge cases in `answers_match`
9. Replace `inspect.getsource()` injection with explicit code file

---

## How to Investigate Further

1. **Run the profiler:**
   ```bash
   uv run python scripts/pipeline_profiler.py all
   ```

2. **Check for container leaks after a failed run:**
   ```bash
   docker ps -a | grep csv
   ```

3. **Validate episode quality:**
   ```bash
   uv run python scripts/pipeline_profiler.py episodes
   ```

4. **Add debug logging temporarily:** In `teacher.py`, add logging to `parse_hooks_from_stdout` to see what's being skipped.
