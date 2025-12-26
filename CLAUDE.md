# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**csv-agent** is a synthetic training data generation pipeline for CSV analysis agents. It uses teacher triangulation to create verified question-answer pairs with execution traces for supervised fine-tuning (SFT) and reinforcement learning (RL).

**Core workflow:**
1. **Question Generation** - LLM explores dataset and generates questions with hints
2. **Teacher Triangulation** - Validate questions by running teacher WITH hint (gold) vs WITHOUT hint N times (consistency)
3. **Episode Export** - Save verified traces as training data (JSONL format)

## Package Manager

This project uses **`uv`** as the package manager. All commands should be prefixed with `uv run`.

## Development Commands

### Setup
```bash
uv sync                    # Install dependencies
uv sync --extra kaggle     # Include optional Kaggle integration
```

### Running the Pipeline

**Stage 1: Generate Questions**
```bash
# LLM-based exploration (default)
uv run python -m src.datagen.question_gen

# Synthetic/compositional (template-based, faster, deterministic)
uv run python -m src.datagen.synthetic.generator
```

**Stage 2: Generate Training Data**
```bash
uv run python -m src.datagen.episode_gen
uv run python -m src.datagen.episode_gen --parallel  # Multi-CSV parallel mode
```

### Testing
```bash
uv run pytest tests/ -v              # Run all tests
uv run pytest tests/test_foo.py -v   # Run single test file
uv run pytest tests/test_foo.py::test_bar -v  # Run single test
```

## Architecture

### Core Execution Flow

**Sandboxed Execution via Docker**
- Agent code runs in isolated Docker containers (`verifiers.PythonEnv`)
- Each container has pandas, numpy, scipy, sklearn, statsmodels pre-installed
- The `submit(answer)` and `hook(value, name=...)` functions are injected into the sandbox
- Communication: Agent prints JSON to stdout → Host parses stdout

**Answer Submission Protocol**
```python
# In sandbox (agent's code):
submit(42)
# Prints: ✓ Submitted: {"__csv_agent_answer__": 42, "hooks": [...]}

# Host side (src/core/environment.py):
parse_submitted_answer(stdout) → extracts wrapped answer
```

**Teacher Triangulation** (`src/datagen/teacher.py`)
```
1. Run gold trace (WITH hint) → gold_answer
2. Run N consistency traces (WITHOUT hint) → [ans1, ans2, ..., ansN]
3. Cluster consistency answers with float tolerance
4. Verify: gold_answer matches majority cluster
5. Only keep episodes where verified=True
```

### Key Modules

**`src/core/`**
- `environment.py` - Main RL-style environment, orchestrates rollouts (multi-turn episodes)
- `conversation.py` - Manages conversation history with context window limits
- `model.py` - LLM API wrapper (supports OpenAI, Anthropic via `verifiers`)
- `config.py` - Pydantic config with global singleton (`from src.core.config import config`)
- `types.py` - Core data structures (Question, TraceDict, TurnDict, Hook, etc.)

**`src/envs/`**
- `csv_env.py` - Docker sandbox environment for Python code execution
  - Injects `normalize_value()` from host to ensure hash consistency
  - Provides `submit(answer)` and `hook(value, name=...)` helpers

**`src/datagen/`**
- `teacher.py` - Teacher triangulation protocol implementation
  - `execute_teacher_trace()` - Single trace execution
  - `triangulate_teacher()` - Gold + consistency verification
  - `batch_triangulate()` - Batch processing with container pooling
- `question_gen.py` - LLM-based dataset exploration and question generation
- `episode_gen.py` - Convert triangulation results to episode JSONL format
- `synthetic/` - Template-based compositional question generation
  - `generator.py` - Main generator class and CLI
  - `templates.py` - Composition templates (e.g., aggregation, filtering, stats)
  - `profiler.py` - Dataset profiling for template selection
  - `verbalizer.py` - LLM-based code-to-NL verbalization

**`src/utils/`**
- `normalization.py` - Normalize values before hashing (DataFrame → dict, numpy → list, etc.)
- `hashing.py` - Hash normalized values for answer comparison
- `docker.py` - Docker container management utilities

### Data Flow

```
CSV → Question Generation → Questions JSON
                ↓
        Teacher Triangulation (gold + N consistency traces)
                ↓
        Verified Episodes → episodes.jsonl
```

**Episode Schema** (see `docs/episode_schema.md`):
- `verified`: bool - Gold matches majority
- `gold_trace`: TraceDict with hooks and turn-level granularity
- `consistency_traces`: List[TraceDict]
- `timing_metadata`: Execution timing per trace (gold_elapsed, consistency_elapsed, total_elapsed, avg_elapsed)

### Configuration

**All config lives in `src/core/config.py` (Pydantic models)**:
- `Config` class - Main application config (models, paths, pipeline settings)
- Global singleton: `from src.core.config import config`
- Override via CLI args in scripts (e.g., `--csv path/to/data.csv`)

**Key config fields**:
- `teacher_model` / `question_gen_model` - Model identifiers
- `question_source` - "llm" (exploration-based) or "synthetic" (template-based)
- `max_turns` - Max conversation turns per episode
- `n_consistency` - Number of no-hint traces for triangulation (default: 5)
- `float_tolerance` - Tolerance for float comparison in answer matching (default: 0.1)

### Container Pooling Optimization

For batch processing, containers are created once and reused (`batch_triangulate` with `use_container_pool=True`):
- Pool size = 1 (gold) + N (consistency traces)
- Containers created in parallel at start
- Reset state between episodes instead of destroy/recreate
- Massive speedup for large batches

### Answer Matching Logic

**Two-tier comparison** (`src/datagen/teacher.py:answers_match`):
1. **Exact hash match** (fast path) - Compare SHA256 hashes
2. **Tolerant comparison** (fallback):
   - Floats: `abs(a - b) <= float_tol` (default ±0.1)
   - DataFrames: Sort columns/rows, compare with `pd.testing.assert_frame_equal`
   - Dicts with p-values: Stricter tolerance for statistical answers (±0.002)
   - Lists/tuples: Element-wise comparison with float tolerance

**Majority voting**: Cluster answers using `answers_match()` with tolerance, find largest cluster.

### Important Patterns

**Strict Submission Protocol** (ENFORCED)
- Agents MUST use `submit(answer)` to submit answers
- `submit()` wraps answer in `{"__csv_agent_answer__": value, "hooks": [...]}`
- Protocol violations (unwrapped answers) will raise `ValueError` and fail the trace
- Rationale: If we decided `submit()` is the protocol, we enforce it strictly
- No defensive fallbacks - fail fast on protocol violations

**Code Smells to Avoid**
- Misleading variable names (e.g., `valid_answers` when just filtering `!= None`)
- Comments describing future code instead of current line
- Defensive checks for variables that should always exist (check control flow exhaustiveness instead)

## Directory Structure

```
csv-agent/
├── src/                    # Main source code
├── tests/                  # Tests
├── training/               # Standalone SFT training package (separate venv)
├── data/                   # All data directories
│   ├── csv/                # Input CSV datasets
│   ├── questions/          # Generated questions
│   ├── episodes/           # Training episodes (JSONL)
│   ├── kaggle/             # Kaggle downloaded datasets
│   ├── fixtures/           # Test fixtures
│   └── mock/               # Mock data for testing
├── docs/                   # Documentation
├── configs/                # Configuration files
├── scripts/                # Utility scripts (incl. kaggle/)
├── README.md
├── CLAUDE.md
└── pyproject.toml
```
