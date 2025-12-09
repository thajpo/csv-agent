# Tool-less Refactor Plan: Scavenger Hunt Architecture

## Overview
Transform from tool-based verification to state-matching rewards in a stateful Jupyter environment.

**Core Philosophy**: Code-as-Policy + Execution-as-Verification

---

## Phase 1: Remove Tool System

### Files to DELETE entirely
- `src/tools.py` (1490 lines) - entire tool specification and implementation
- `tests/test_tool_call.py` - tool-specific tests

### Files to MODIFY
- `src/environment.py`:
  - **REMOVE**: `parse_response_code_blocks()`, `_execute_and_structure_tools()`, `_build_feedback_from_tool_calls()`
  - **REMOVE**: All `run_tool()` calls
  - **KEEP**: `init_state()`, `get_model_response()`, basic turn processing structure

- `src/conversation.py`:
  - **REMOVE**: `ToolCall` class entirely
  - **MODIFY**: `Turn` class - remove `tool_calls: list[ToolCall]` field
  - **KEEP**: `ConversationManager`, turn archiving, context management logic

- `src/prompts.py`:
  - **REMOVE**: `format_tool_docs()`, `format_hook_tools_doc()`, `extract_tools_from_trace()`
  - **REMOVE**: All TOOL_CALL_SYNTAX, HOOK_CHAIN_PATTERNS fragments
  - **REMOVE**: EPISODE_SCHEMA, QUESTION_PLAN_SCHEMA (will rebuild)
  - **KEEP**: `generate_data_overview()` (still useful for initial context)
  - **KEEP**: `DEFAULT_DATASET_DESCRIPTION` (still useful)

### What Happens to Hook/Episode Types
**Current design uses tools** - hooks reference tool names and parameters:
```python
class Hook(BaseModel):
    tool: str                  # ← TOOL REFERENCE
    params: HookParams
```

**New design**: Hooks become DataFrame state hashes, not tool calls. We'll handle this in Phase 3.

---

## Phase 2: Refactor Teacher to Free-Form Solver

### New Teacher Behavior
Teacher is now **unconstrained** - no tools, just raw Python code generation.

### New Prompt Design

**src/prompts.py** - Complete rewrite of teacher prompts:

```python
# NEW: Teacher "Tutor Mode" System Prompt
TEACHER_TUTOR_MODE_PROMPT = """
You are a data analysis tutor solving pandas problems step-by-step.

DATASET:
{dataset_description}

{data_overview}

QUESTION:
{question_text}

HINT:
{hint}

RULES:
1. Write verbose, educational Python code
2. Use meaningful intermediate variable names (df_filtered, df_grouped, etc.)
3. Avoid complex one-liners - break down into steps
4. Print intermediate results to verify your work
5. Call submit(final_answer) when done

Your code will execute in a stateful Jupyter kernel. You can:
- Inspect data: df.head(), df.info(), df.describe()
- Debug errors: try different approaches across turns
- Print intermediate results: print(df_filtered.shape)
- Build incrementally: define variables across multiple cells

Write ONE code cell per turn. End with submit(final_answer) when you have the answer.
"""

# NEW: Teacher Consistency Mode (no hint)
TEACHER_CONSISTENCY_PROMPT = """
[Same as above, but WITHOUT the hint parameter]
"""
```

### New Turn Structure
**Before**: Teacher generates `<code>{"tool": "group_stat", "params": {...}}</code>`
**After**: Teacher generates raw Python:

```python
# Example teacher response (Turn 1)
Let me first filter the data to the control group.

```python
df_control = df[df['fertilizer'] == 'control']
print(f"Control group: {len(df_control)} rows")
print(df_control['yield'].describe())
```

# Example teacher response (Turn 2)
Now I'll calculate the mean yield.

```python
mean_yield = df_control['yield'].mean()
print(f"Mean yield: {mean_yield:.2f}")
submit(mean_yield)
```
```

### Files to CREATE
- **src/prompts.py** - New functions:
  - `build_teacher_tutor_prompt(question: str, hint: str)` → str
  - `build_teacher_consistency_prompt(question: str)` → str
  - `build_student_prompt(question: str, teacher_trace: str)` → str

### Files to MODIFY
- **src/environment.py**:
  - Replace `parse_response_code_blocks()` with `extract_python_cells(response: str)`
  - Extract ```python...``` blocks instead of `<code>...</code>`
  - Pass directly to kernel instead of tool parsing

---

## Phase 3: Implement State-Matching Reward System

### New Core Concept: Artifact Hashing

**What is an artifact?**
Any DataFrame or scalar value in the kernel's local namespace.

**The Scavenger Hunt**:
1. Teacher executes code → produces variables: `df_filtered`, `df_grouped`, `mean_yield`
2. Hash each variable's state → "Bag of Artifacts"
3. Student executes code → hash their variables
4. **Reward**: +1 for each hash match (dense), +5 for final answer match (sparse)

### New Types (`src/types.py`)

```python
from hashlib import sha256
import pandas as pd
import pickle

def hash_artifact(obj: Any) -> str:
    """Hash a DataFrame or scalar value for state matching."""
    if isinstance(obj, pd.DataFrame):
        # Hash columns, dtypes, and values
        content = pickle.dumps((obj.columns.tolist(), obj.dtypes.tolist(), obj.values.tobytes()))
    else:
        # Scalar: just pickle it
        content = pickle.dumps(obj)
    return sha256(content).hexdigest()[:16]  # 16 chars is enough

class Artifact(BaseModel):
    """A checkpoint variable from code execution."""
    name: str              # Variable name (e.g., 'df_filtered')
    hash: str             # Hash of its state
    type: str             # 'DataFrame' | 'scalar'

class TeacherTrace(BaseModel):
    """Teacher's solution execution record."""
    question: str
    hint: str | None
    code_cells: list[str]              # Raw Python code per turn
    artifacts: dict[str, Artifact]     # name → Artifact
    final_answer: Any                  # The submit() value
    final_answer_hash: str
    execution_success: bool

class StudentTrace(BaseModel):
    """Student's solution attempt."""
    question: str
    code_cells: list[str]
    artifacts: dict[str, Artifact]
    final_answer: Any | None
    final_answer_hash: str | None
    execution_success: bool

class Episode(BaseModel):
    """Complete training episode."""
    question: str
    teacher_trace: TeacherTrace
    student_trace: StudentTrace | None  # May be None if student not run yet

    # Rewards
    intermediate_matches: list[str]     # Artifact names student matched
    final_match: bool                   # Did final answer match?
    dense_reward: int                   # +1 per intermediate match
    sparse_reward: int                  # +5 if final match
    total_reward: float                 # Combined score
```

### New Kernel Interface (`src/kernel.py`)

**Current kernel.py**: Already has `execute_code()` and `get_locals()`.

**Add**:
```python
class JupyterKernel:
    # ... existing code ...

    def snapshot_artifacts(self) -> dict[str, Artifact]:
        """Capture all DataFrames and scalars in namespace."""
        locals_dict = self.get_locals()
        artifacts = {}

        for name, obj in locals_dict.items():
            if name.startswith('_'):
                continue  # Skip private vars

            if isinstance(obj, pd.DataFrame):
                artifacts[name] = Artifact(
                    name=name,
                    hash=hash_artifact(obj),
                    type='DataFrame'
                )
            elif isinstance(obj, (int, float, str, bool)):
                artifacts[name] = Artifact(
                    name=name,
                    hash=hash_artifact(obj),
                    type='scalar'
                )

        return artifacts

    def get_final_answer(self) -> Any | None:
        """Retrieve value passed to submit()."""
        # submit() will store in a special variable
        return self.get_locals().get('__SUBMITTED_ANSWER__', None)
```

### Files to CREATE
- **src/rewards.py**: Reward calculation logic

```python
def calculate_scavenger_hunt_reward(
    teacher_artifacts: dict[str, Artifact],
    student_artifacts: dict[str, Artifact],
    teacher_final_hash: str,
    student_final_hash: str | None
) -> tuple[list[str], bool, int, int]:
    """
    Returns:
        - intermediate_matches: List of artifact names student matched
        - final_match: Whether final answers match
        - dense_reward: Number of intermediate matches
        - sparse_reward: 5 if final match, else 0
    """
    # Find hash intersections
    teacher_hashes = {a.hash: a.name for a in teacher_artifacts.values()}
    student_hashes = {a.hash: a.name for a in student_artifacts.values()}

    matched_names = []
    for student_hash, student_name in student_hashes.items():
        if student_hash in teacher_hashes:
            teacher_name = teacher_hashes[student_hash]
            matched_names.append(f"{student_name}←→{teacher_name}")

    dense_reward = len(matched_names)

    final_match = (student_final_hash == teacher_final_hash) if student_final_hash else False
    sparse_reward = 5 if final_match else 0

    return matched_names, final_match, dense_reward, sparse_reward
```

### Files to MODIFY
- **src/environment.py**: After each code cell execution, call `kernel.snapshot_artifacts()`

---

## Phase 4: Build Teacher Triangulation Protocol

### The Consistency Check
Teacher solves the problem N times (with and without hints) to verify correctness.

**Algorithm**:
1. Generate question + hint
2. Teacher solves WITH hint → `gold_trace`
3. Teacher solves WITHOUT hint N times → `consistency_traces`
4. Compare final answers: `gold_trace.final_answer` vs `consistency_traces[i].final_answer`
5. **Keep episode** if: gold execution succeeds AND gold answer matches majority of consistency answers

### New Pipeline Mode

**src/main.py** - Add new mode:

```python
PIPELINE_MODES = {
    "generate-questions": generate_questions,      # Phase 1: question ideas
    "generate-episodes": generate_episodes,        # Phase 2: teacher traces with triangulation
    "train-student": train_student,                # Phase 3: RL training (future)
}
```

### New Function: `generate_episodes()`

```python
def generate_episodes(config: EnvironmentConfig, n_episodes: int = 10) -> list[Episode]:
    """
    Generate verified teacher traces using triangulation.

    Flow:
    1. Generate question + hint
    2. Teacher solves WITH hint (tutor mode)
    3. Teacher solves WITHOUT hint 3x (consistency mode)
    4. Verify: gold trace matches majority
    5. Save Episode with TeacherTrace
    """
    episodes = []

    for i in range(n_episodes):
        # 1. Generate question
        question, hint = generate_question_with_hint()

        # 2. Gold trace (with hint)
        gold_trace = execute_teacher_trace(
            question=question,
            hint=hint,
            mode="tutor"
        )

        if not gold_trace.execution_success:
            continue  # Skip failed episodes

        # 3. Consistency traces (no hint)
        consistency_traces = [
            execute_teacher_trace(question=question, hint=None, mode="consistency")
            for _ in range(3)
        ]

        # 4. Verify majority match
        consistency_answers = [t.final_answer for t in consistency_traces if t.execution_success]

        if not consistency_answers:
            continue  # All consistency attempts failed

        # Majority vote
        from collections import Counter
        answer_counts = Counter(consistency_answers)
        majority_answer, majority_count = answer_counts.most_common(1)[0]

        if gold_trace.final_answer != majority_answer:
            continue  # Gold doesn't match majority

        if majority_count < 2:
            continue  # No clear majority (need at least 2/3)

        # Success! Save episode
        episode = Episode(
            question=question,
            teacher_trace=gold_trace,
            student_trace=None,  # Not run yet
            intermediate_matches=[],
            final_match=False,
            dense_reward=0,
            sparse_reward=0,
            total_reward=0.0
        )
        episodes.append(episode)

    return episodes
```

### Files to CREATE
- **src/teacher.py**: Teacher execution logic
  - `execute_teacher_trace(question, hint, mode)` → `TeacherTrace`
  - `generate_question_with_hint()` → `(question, hint)`

---

## Phase 5: Refactor Environment for Jupyter MDP

### New Environment Design

**Current**: Multi-turn tool execution loop
**New**: Multi-turn code cell execution loop

### Key Changes to `src/environment.py`

**REMOVE**:
- `parse_response_code_blocks()` (tool JSON parsing)
- `_execute_and_structure_tools()`
- `_build_feedback_from_tool_calls()`
- `check_done_signal()` (no more DONE marker - use submit() instead)

**ADD**:
```python
class Environment:
    def __init__(self, csv_path, config, kernel: JupyterKernel, logger):
        self.kernel = kernel  # Stateful kernel
        # ... rest same ...

    def extract_python_cells(self, response: str) -> list[str]:
        """Extract ```python...``` code blocks from response."""
        import re
        pattern = r'```python\n(.*?)```'
        return re.findall(pattern, response, re.DOTALL)

    def execute_code_cell(self, code: str) -> dict:
        """
        Execute code in kernel and return execution result.

        Returns:
            {
                'success': bool,
                'stdout': str,
                'stderr': str,
                'artifacts': dict[str, Artifact],
                'submitted_answer': Any | None
            }
        """
        # Execute
        result = self.kernel.execute_code(code)

        # Snapshot state
        artifacts = self.kernel.snapshot_artifacts()

        # Check for submit()
        submitted_answer = self.kernel.get_final_answer()

        return {
            'success': result['success'],
            'stdout': result.get('stdout', ''),
            'stderr': result.get('stderr', ''),
            'artifacts': artifacts,
            'submitted_answer': submitted_answer
        }

    def process_turn(self, state: StateConfig) -> Turn:
        """
        Process one turn of code execution.

        Flow:
        1. Get model response
        2. Extract Python cells
        3. Execute each cell
        4. Build feedback from execution results
        5. Check if submit() was called
        """
        response = self.get_model_response(state)

        code_cells = self.extract_python_cells(response)

        if not code_cells:
            # No code - just commentary
            feedback = "No code was provided. Please write Python code in ```python blocks."
            turn = Turn(
                turn_number=state.current_turn,
                model_response=response,
                code_cells=[],
                execution_results=[],
                done_signal=False,
                feedback_message=feedback
            )
            return turn

        # Execute all cells
        execution_results = []
        submitted_answer = None

        for cell_code in code_cells:
            result = self.execute_code_cell(cell_code)
            execution_results.append(result)

            if result['submitted_answer'] is not None:
                submitted_answer = result['submitted_answer']
                break  # Stop on submit()

        # Build feedback
        feedback_parts = []
        for i, result in enumerate(execution_results):
            if result['success']:
                feedback_parts.append(f"Cell {i+1}: ✓ Executed")
                if result['stdout']:
                    feedback_parts.append(result['stdout'])
            else:
                feedback_parts.append(f"Cell {i+1}: ✗ Error")
                feedback_parts.append(result['stderr'])

        feedback = "\n".join(feedback_parts)

        # Create turn
        turn = Turn(
            turn_number=state.current_turn,
            model_response=response,
            code_cells=code_cells,
            execution_results=execution_results,
            done_signal=(submitted_answer is not None),
            feedback_message=feedback
        )

        return turn
```

### New Turn Type

**src/conversation.py**:
```python
class CodeCellResult(BaseModel):
    """Result of executing one code cell."""
    code: str
    success: bool
    stdout: str
    stderr: str
    artifacts: dict[str, Artifact]
    submitted_answer: Any | None

class Turn(BaseModel):
    turn_number: int
    timestamp: datetime

    model_response: str
    code_cells: list[str]
    execution_results: list[CodeCellResult]

    done_signal: bool           # True if submit() was called
    feedback_message: str
    estimated_tokens: int | None = None
```

### Built-in Functions for Kernel

**src/kernel.py** - Add to kernel initialization:
```python
def setup_kernel_builtins(self):
    """Inject helper functions into kernel namespace."""
    builtin_code = """
import pandas as pd
import numpy as np

__SUBMITTED_ANSWER__ = None

def submit(answer):
    '''Submit your final answer.'''
    global __SUBMITTED_ANSWER__
    __SUBMITTED_ANSWER__ = answer
    print(f"✓ Submitted: {answer}")
    return answer

# Load dataset
df = pd.read_csv({csv_path!r})
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
"""
    self.execute_code(builtin_code.format(csv_path=self.csv_path))
```

---

## Phase 6: Update Types and Add Episode Logging

### New Core Types (`src/types.py`)

**Full rewrite** - Replace Hook/Episode with:

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Any

# Artifact types (from Phase 3)
class Artifact(BaseModel): ...
class TeacherTrace(BaseModel): ...
class StudentTrace(BaseModel): ...

# Episode type (updated)
class Episode(BaseModel):
    id: str                              # Unique episode ID
    timestamp: datetime
    question: str
    hint: str | None

    # Teacher data
    teacher_trace: TeacherTrace
    consistency_traces: list[TeacherTrace]  # For triangulation verification

    # Student data (populated during training)
    student_trace: StudentTrace | None

    # Rewards
    intermediate_matches: list[str]
    final_match: bool
    dense_reward: int
    sparse_reward: int
    total_reward: float

    # Metadata
    difficulty: str | None                # Optional tagging
    verified: bool                        # Passed triangulation?

# Configuration (simplified)
class EnvironmentConfig(BaseModel):
    csv_path: str = "data.csv"
    model: str = "grok-4.1-fast"
    max_turns: int = 10
    max_active_turns: int = 5
    max_context_tokens: int = 80_000
```

### Episode Logging System

**New file**: `src/episode_logger.py`

```python
import json
from pathlib import Path
from datetime import datetime

class EpisodeLogger:
    """Serialize episodes for future verifier training."""

    def __init__(self, output_dir: str = "episodes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_episode(self, episode: Episode):
        """Save episode as JSON."""
        filename = f"{episode.id}_{episode.timestamp.isoformat()}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(episode.model_dump(), f, indent=2, default=str)

    def save_batch(self, episodes: list[Episode], batch_name: str):
        """Save multiple episodes to single file."""
        filename = f"{batch_name}_{datetime.now().isoformat()}.jsonl"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            for episode in episodes:
                f.write(json.dumps(episode.model_dump(), default=str) + '\n')

    def load_episodes(self, filepath: str) -> list[Episode]:
        """Load episodes from JSONL."""
        episodes = []
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                episodes.append(Episode(**data))
        return episodes
```

**Strategic Value**:
- **Hard Negatives**: Code that executed but produced wrong artifacts
- **Near Misses**: Matched intermediate steps but failed final answer
- **Perfect Dataset**: For training discriminator models later

---

## Migration Strategy

### Step-by-Step Execution

**Week 1**: Removal and simplification
1. Delete `src/tools.py` and `tests/test_tool_call.py`
2. Remove tool references from `environment.py`, `conversation.py`, `prompts.py`
3. Verify codebase still runs (will be broken, but should import)

**Week 2**: Teacher refactor
4. Rewrite `src/prompts.py` with new teacher prompts
5. Update `environment.py` to extract Python cells instead of tools
6. Test teacher can generate and execute free-form code

**Week 3**: Rewards and kernel
7. Add `hash_artifact()` and new types to `types.py`
8. Update `kernel.py` with `snapshot_artifacts()` and `submit()`
9. Create `src/rewards.py` with scavenger hunt logic
10. Test artifact matching on simple examples

**Week 4**: Triangulation
11. Create `src/teacher.py` with triangulation logic
12. Test question generation + consistency checking
13. Verify episodes are properly filtered

**Week 5**: Integration
14. Update `main.py` with new pipeline modes
15. Create `src/episode_logger.py`
16. End-to-end test: generate 10 verified episodes

**Week 6**: Cleanup and testing
17. Remove dead code (old prompt fragments, unused types)
18. Write new tests for artifact hashing, rewards, triangulation
19. Update README with new architecture docs

---

## Testing Strategy

### Unit Tests
- **test_artifact_hashing.py**: Verify DataFrame and scalar hashing
- **test_rewards.py**: Verify scavenger hunt reward calculation
- **test_teacher_triangulation.py**: Verify consistency checking
- **test_kernel_builtins.py**: Verify submit() and artifact snapshotting

### Integration Tests
- **test_teacher_episode_generation.py**: Full teacher trace with triangulation
- **test_student_episode_execution.py**: Student trace with reward calculation
- **test_episode_logging.py**: Serialization and deserialization

### End-to-End Tests
- Generate 10 episodes with triangulation
- Verify all episodes have `verified=True`
- Check artifact counts (should have >1 intermediate checkpoint each)
- Verify episode logs can be loaded back

---

## What Remains vs What's Removed vs What's Added

### REMAINS (Keep as-is or minor refactor)
- ✅ `src/types.py` structure (rewrite internals, keep pattern)
- ✅ `src/conversation.py` - ConversationManager, context management
- ✅ `src/model.py` - APILLM, LLM interfaces
- ✅ `src/kernel.py` - Core kernel execution (add artifact snapshotting)
- ✅ `src/rich_logger.py` - Logging and UI
- ✅ `config.yaml` - Configuration system
- ✅ `main.py` structure (change pipeline modes)

### REMOVED (Delete entirely)
- ❌ `src/tools.py` - 1490 lines of tool specs and implementations
- ❌ `tests/test_tool_call.py` - Tool-specific tests
- ❌ Tool-based prompts in `prompts.py`:
  - TOOL_CALL_SYNTAX
  - EPISODE_SCHEMA (old version)
  - QUESTION_PLAN_SCHEMA (old version)
  - format_tool_docs()
  - format_hook_tools_doc()
  - HOOK_CHAIN_PATTERNS
  - WORKED_EPISODE_EXAMPLE (tool-based)

### ADDED (New files and features)
- ✨ `src/rewards.py` - Scavenger hunt reward system
- ✨ `src/teacher.py` - Teacher triangulation protocol
- ✨ `src/episode_logger.py` - Episode serialization for verifier research
- ✨ New prompt templates in `prompts.py`:
  - TEACHER_TUTOR_MODE_PROMPT
  - TEACHER_CONSISTENCY_PROMPT
  - STUDENT_PROMPT (simplified)
- ✨ Artifact hashing in `types.py`
- ✨ Built-in functions in `kernel.py`: `submit()`
- ✨ New tests:
  - test_artifact_hashing.py
  - test_rewards.py
  - test_teacher_triangulation.py
  - test_episode_logging.py

---

## Risk Mitigation

### Git Safety
1. Create branch: `git checkout -b refactor-toolless`
2. Commit after each phase
3. Keep main branch stable

### Rollback Points
- After Phase 1: Can still view old tool code in git history
- After Phase 3: Can test rewards independently
- After Phase 5: Can test environment independently

### Validation Checkpoints
- [ ] Phase 2: Teacher generates valid Python (not tool JSON)
- [ ] Phase 3: Artifacts hash consistently
- [ ] Phase 4: Triangulation filters bad episodes
- [ ] Phase 5: Student executes in stateful kernel
- [ ] Phase 6: Episodes serialize/deserialize correctly

---

## Open Questions

1. **Question Generation**: How do we generate initial questions without tools?
   - Option A: Hand-write 10 seed questions
   - Option B: Use teacher to propose questions in natural language
   - Option C: Keep minimal question-gen mode with examples

2. **Hint Generation**: What makes a good hint?
   - Could be: "Focus on the control group first"
   - Could be: "This requires grouping by fertilizer type"
   - Needs experimentation

3. **Artifact Granularity**: Do we hash every variable or only DataFrames?
   - Current plan: DataFrames + scalars
   - Could expand: lists, dicts if needed

4. **Error Handling**: What if teacher code crashes during triangulation?
   - Current: Skip that episode
   - Alternative: Retry with different sampling params

5. **Student Timeout**: How long to wait for student code execution?
   - Current: Same as teacher (120s per cell)
   - Alternative: Shorter timeout to discourage infinite loops

---

## Success Criteria

After refactor completion, we should be able to:

1. ✅ Generate 10 verified episodes using teacher triangulation
2. ✅ Each episode has >1 intermediate artifact checkpoint
3. ✅ Run student on episode and calculate scavenger hunt reward
4. ✅ Serialize episodes to disk for future verifier training
5. ✅ Zero tool references in codebase
6. ✅ All tests passing

---

## Next Steps

Once you approve this plan, I can:
1. Start with Phase 1 (removal) - safest, most mechanical
2. Create a branch and begin systematic refactor
3. Commit after each phase for easy rollback
4. Validate at each checkpoint before moving forward

Ready to proceed?
