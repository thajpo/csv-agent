# Prompt Token Optimization Guide

## Overview

This guide documents the prompt optimization system for reducing token usage in student RL training (GRPO). The optimizations reduce the student prompt from **2,350 tokens → 434 tokens (81.5% reduction)**, freeing up context space for teacher traces.

## Quick Start

### Basic Usage (Minimal Mode)

```python
from src.prompts import build_student_prompt, DEFAULT_DATASET_DESCRIPTION

# Default: minimal verbosity, no data overview (assumes it's in teacher trace)
prompt = build_student_prompt(
    dataset_description=DEFAULT_DATASET_DESCRIPTION,
    data_overview=data_overview,  # Only used if include_data_overview=True
    verbosity="minimal",            # NEW: function signatures
    include_data_overview=False,    # NEW: omit redundant data
)
# Result: ~655 tokens (72% reduction vs baseline)
```

### Advanced Usage (Lazy Tool Loading)

```python
from src.prompts import build_student_prompt, extract_tools_from_trace

# Extract tools from teacher trace
teacher_turns = [...]  # List of Turn objects from teacher
used_tools = extract_tools_from_trace(teacher_turns)

# Build prompt with only relevant tools
prompt = build_student_prompt(
    dataset_description=DEFAULT_DATASET_DESCRIPTION,
    data_overview=data_overview,
    verbosity="minimal",
    include_data_overview=False,
    filter_tools=used_tools,  # NEW: only include used tools
)
# Result: ~434 tokens (81% reduction vs baseline)
```

## Token Reduction Summary

| Configuration | Tokens | Savings vs Baseline |
|--------------|--------|---------------------|
| Baseline (compact + full data) | 2,350 | - |
| Minimal verbosity | 2,258 | -92 (3.9%) |
| Minimal + no data overview | 655 | -1,695 (72.1%) |
| Minimal + no data + lazy (8 tools) | 434 | -1,916 (81.5%) |

## Implementation Details

### 1. Minimal Verbosity Mode

**What it does**: Replaces verbose tool descriptions with compact function signatures.

**Format**:
```
tool_name(param1, param2, param3=default) → output_type
  Example: {"tool": "tool_name", ...}  # Only for complex tools
```

**Example**:

Before (compact mode):
```
**group_stat**: Aggregate a column grouped by another
  Required: group_col*, target_col*, agg*
```

After (minimal mode):
```
group_stat(group_col, target_col="None", agg="mean", ...) → scalar
```

**Complex tools** (with examples):
- `combine` - arithmetic on hook results
- `lookup` - dynamic group queries
- `aggregate_hooks` - min/max across hooks
- `group_extremum` - find best/worst group
- `filter_stat` - filtered aggregations
- `derive_stat` - computed columns
- `multi_group_stat` - multi-column aggregations

**Token savings**: ~92 tokens (17% of tool docs)

---

### 2. Structural Compression (No Data Overview)

**What it does**: Removes redundant data overview from student prompt.

**Rationale**: In RL training, the teacher trace already contains the data overview. No need to repeat it in the system prompt.

**Change**:
```python
# Before
DATA:
```
<full data overview with shape, head, dtypes, stats>
```

# After
DATA: See data overview in context above.
```

**Token savings**: ~1,600 tokens (68% of baseline)

**When to use**: Always in RL training. Set `include_data_overview=False`.

**When NOT to use**: If student needs to run standalone without teacher trace.

---

### 3. Lazy Tool Loading

**What it does**: Only includes tool docs for tools actually used by the teacher.

**Implementation**:
```python
from src.prompts import extract_tools_from_trace

# Extract tools from teacher trace
used_tools = extract_tools_from_trace(teacher_turns)
# Returns: {"group_stat", "combine", "lookup", ...}

# Build prompt with filtered tools
prompt = build_student_prompt(..., filter_tools=used_tools)
```

**Token savings**: Proportional to tools filtered out
- 28 tools → 8 tools: ~220 additional tokens (29% reduction in tool docs)
- 28 tools → 5 tools: ~320 additional tokens (43% reduction in tool docs)

**Trade-off**: Student can only use tools the teacher demonstrated. May limit creativity.

**Recommendation**:
- **Safe approach**: Always include 3-5 common exploratory tools (`inspect`, `describe`, `value_counts`) even if teacher didn't use them
- **Aggressive approach**: Only include exact tools from teacher trace (maximizes token savings)

---

### 4. Output Format Simplification

**What it does**: Replaces verbose JSON schema with reference to examples.

**Change**:
```python
# Before
Output your answer as JSON matching:
```json
{
  "question_text": "...",
  "difficulty": "MEDIUM | HARD | VERY_HARD",
  "hooks": [...],
  ...
}
```

# After
Output your answer as JSON (see examples in teacher trace for format).
```

**Token savings**: ~250 tokens (already included in measurements above)

---

## Testing Your Configuration

Run the test script to compare different configurations:

```bash
uv run test_prompt_optimization.py
```

This will show:
1. Token counts for each configuration
2. Percentage savings vs baseline
3. Sample output for minimal mode
4. Tool docs comparison table

---

## Migration Guide

### For Existing Code

If you're currently using `build_student_prompt()`:

```python
# Old (before optimization)
prompt = build_student_prompt(
    dataset_description=desc,
    data_overview=overview,
)

# New (optimized for RL training)
prompt = build_student_prompt(
    dataset_description=desc,
    data_overview=overview,
    verbosity="minimal",           # Add this
    include_data_overview=False,   # Add this
)
```

### For RL Training Pipeline

```python
# 1. Run teacher episode
teacher_turns = run_teacher_episode(...)

# 2. Extract tools from teacher trace
used_tools = extract_tools_from_trace(teacher_turns)

# 3. Build optimized student prompt
student_prompt = build_student_prompt(
    dataset_description=dataset_desc,
    data_overview=data_overview,
    verbosity="minimal",
    include_data_overview=False,
    filter_tools=used_tools,  # Optional: lazy loading
)

# 4. Construct full context for GRPO
context = [
    {"role": "system", "content": student_prompt},
    *teacher_turns_as_messages,  # Teacher trace
    {"role": "user", "content": "Solve this question: ..."}
]

# 5. Run student with full context
student_output = run_student(context)
```

---

## API Reference

### `build_student_prompt()`

```python
def build_student_prompt(
    dataset_description: str,
    data_overview: str,
    verbosity: str = "minimal",
    include_data_overview: bool = False,
    filter_tools: set[str] | None = None,
) -> str:
    """
    Build minimal prompt for student model (optimized for RL training).

    Args:
        dataset_description: Description of the dataset
        data_overview: Pre-computed data exploration output
        verbosity: "minimal" (signatures, default) | "compact" (descriptions) | "full" (all details)
        include_data_overview: Include full data in prompt (default False)
        filter_tools: Optional set of tool names to include (lazy loading)

    Returns:
        Formatted system prompt string
    """
```

### `extract_tools_from_trace()`

```python
def extract_tools_from_trace(turns: list) -> set[str]:
    """
    Extract unique tool names from teacher trace.

    Args:
        turns: List of Turn objects from teacher execution

    Returns:
        Set of tool names used in the trace
    """
```

### `format_tool_docs()`

```python
def format_tool_docs(
    verbosity: str = "full",
    filter_tools: set[str] | None = None
) -> str:
    """
    Generate tool documentation for system prompt.

    Args:
        verbosity: "minimal" (signatures) | "compact" (descriptions) | "full" (detailed)
        filter_tools: Optional set of tool names to include

    Returns:
        Formatted tool documentation string
    """
```

---

## Recommendations

### For GRPO Training (Recommended)

Use the most aggressive optimization:

```python
prompt = build_student_prompt(
    dataset_description=desc,
    data_overview=overview,
    verbosity="minimal",
    include_data_overview=False,
    filter_tools=extract_tools_from_trace(teacher_turns),
)
```

**Result**: ~434 tokens (81% reduction)

### For Safe Exploration

Keep data overview but use minimal tool docs:

```python
prompt = build_student_prompt(
    dataset_description=desc,
    data_overview=overview,
    verbosity="minimal",
    include_data_overview=True,  # Keep data
    # No filter_tools - keep all tools
)
```

**Result**: ~2,258 tokens (4% reduction, but safer)

### For Debugging

Use compact mode to see tool descriptions:

```python
prompt = build_student_prompt(
    dataset_description=desc,
    data_overview=overview,
    verbosity="compact",  # Readable descriptions
    include_data_overview=True,
)
```

**Result**: ~2,350 tokens (baseline)

---

## Configuration File Integration

You can add flags to `config.yaml` to control optimization:

```yaml
# Student prompt optimization
student_prompt:
  verbosity: "minimal"           # minimal | compact | full
  include_data_overview: false   # true | false
  use_lazy_tool_loading: true    # true | false
```

Then in code:

```python
from src.config import load_config

config = load_config()

prompt = build_student_prompt(
    dataset_description=desc,
    data_overview=overview,
    verbosity=config.student_prompt.verbosity,
    include_data_overview=config.student_prompt.include_data_overview,
    filter_tools=extract_tools_from_trace(teacher_turns) if config.student_prompt.use_lazy_tool_loading else None,
)
```

---

## Validation

To ensure optimizations don't hurt performance:

1. **A/B test**: Run student with both baseline and optimized prompts
2. **Compare metrics**: Success rate, hook accuracy, final answer accuracy
3. **Measure context usage**: Ensure you're fitting more teacher turns
4. **Check creativity**: Verify student can still explore when needed

---

## Example Output

### Minimal Mode Tool Docs (excerpt)

```
AVAILABLE TOOLS:

inspect(aspect, n=5) → table

describe(include="number") → table

group_stat(group_col, target_col="None", agg="mean", ...) → scalar

combine(expr, depends_on, new_id) → scalar
  Example: {"tool": "combine", "expr": "@a / @b", "depends_on": ["a", "b"], "new_id": "ratio"}

lookup(hook_id, lookup_col, target_col, agg) → scalar
  Example: {"tool": "lookup", "hook_id": "best_tr", "lookup_col": "TR", "target_col": "TL", "agg": "mean"}

aggregate_hooks(hooks, agg) → scalar
  Example: {"tool": "aggregate_hooks", "hooks": ["cv1", "cv2", "cv3"], "agg": "min"}
```

---

## Files Modified

1. **`src/tools.py`** (lines 778-864)
   - Added `verbosity="minimal"` support to `format_tool_docs()`
   - Added `filter_tools` parameter for lazy loading
   - Defined `COMPLEX_TOOLS` set for selective examples

2. **`src/prompts.py`** (lines 373-416, 474-492)
   - Updated `build_student_prompt()` with new parameters
   - Added `extract_tools_from_trace()` helper function
   - Simplified output format instructions

3. **`test_prompt_optimization.py`** (new file)
   - Comprehensive test suite for token optimization
   - Comparison of all configurations
   - Sample output display

---

## Future Enhancements

Potential further optimizations (not yet implemented):

1. **Compressed parameter names**: Use `grp`, `agg`, `tgt` instead of `group_col`, `agg`, `target_col`
   - Estimated savings: ~100-150 tokens (15-20%)
   - Risk: May confuse models not trained on abbreviations

2. **Dynamic complexity detection**: Automatically determine which tools need examples based on teacher trace complexity
   - Estimated savings: ~50-80 tokens
   - Implementation: Count tool usage frequency, add examples only for frequently-used complex tools

3. **Tool description removal**: Remove all descriptions, rely purely on signatures + examples
   - Estimated savings: Already achieved in minimal mode

4. **Prompt caching**: Use API-level prompt caching to cache system prompt across requests
   - Not a token reduction, but a cost reduction
   - Works with Anthropic Claude API (prompt caching feature)

---

## Support

For questions or issues:
1. Run `uv run test_prompt_optimization.py` to verify setup
2. Check token counts match expectations
3. Review this guide for configuration options
4. Test with your specific RL pipeline

## Changelog

- **2025-01-XX**: Initial implementation
  - Added minimal verbosity mode
  - Added lazy tool loading
  - Added structural compression
  - Created test suite and documentation
