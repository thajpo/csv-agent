# Multi-Agent Implementation: Episode Factory & Procedural Analysis

## Overview
This document coordinates parallel implementation of episode factory centralization, dead code detection, and procedural question analysis.

## Architecture Decisions

### Episode Factory Interface
```python
# src/datagen/shared/episode_factory.py

from typing import Literal
from dataclasses import dataclass

async def create_episode(
    question: dict,
    verification_result: VerificationResult,
    source: Literal["synthetic", "llm", "procedural"],
    csv_path: str,
) -> EpisodeJSONL:
    """Create episode from verification result.
    
    Args:
        question: Question metadata (must include id, question_text, hint, etc.)
        verification_result: Output from verify_question()
        source: Origin of question
        csv_path: Path to source CSV
    
    Returns:
        EpisodeJSONL with verification metadata embedded (success flag, error info, etc.)
    """

# Helper for ground-truth strategy (synthetic/procedural)
async def create_episode_from_ground_truth(
    question: dict,
    csv_path: str,
    model: str,
    **kwargs
) -> EpisodeJSONL:
    """Convenience wrapper for ground-truth verification."""

# Helper for consistency strategy (LLM)
async def create_episode_from_consistency(
    question: dict,
    csv_path: str,
    model: str,
    n_consistency: int = 5,
    **kwargs
) -> EpisodeJSONL:
    """Convenience wrapper for consistency verification."""
```

### Dead Code Detection
- **Policy**: REJECT chains with dead code immediately
- **Method**: Static analysis using operator produces/consumes metadata
- **Location**: `src/datagen/synthetic/programs/dead_code_validator.py`

### Analysis Output Format
- **Internal**: JSON for machine parsing
- **CLI Display**: Human-readable table
- **Location**: `src/datagen/analyze_procedural.py` outputs JSON, CLI formats as table

## Agent Assignments

### Agent 1: Episode Factory Core
**Files:**
- `src/datagen/shared/episode_factory.py` (NEW)
- `tests/test_episode_factory.py` (NEW)

**Deliverables:**
1. `create_episode()` function with ground_truth and consistency strategies
2. Helper functions for common use cases
3. Unit tests (test first, then implement)
4. Report to progress log when tests pass

**Dependencies:** None (uses existing shared/verification.py)

### Agent 2: Dead Code Detection
**Files:**
- `src/datagen/synthetic/programs/operators.py` (MODIFY - add produces/consumes)
- `src/datagen/synthetic/programs/dead_code_validator.py` (NEW)
- `tests/test_dead_code_validator.py` (NEW)

**Deliverables:**
1. Add `produces: List[str]` and `consumes: List[str]` to Op dataclass
2. Update existing operators with metadata
3. `validate_no_dead_code(chain: List[OpInstance]) -> bool`
4. Unit tests (test first, then implement)
5. Report to progress log when tests pass

**Dependencies:** None (self-contained enhancement)

### Agent 3: Analysis Pipeline
**Files:**
- `src/datagen/analyze_procedural.py` (NEW)
- `tests/test_analyze_procedural.py` (NEW)
- `tests/fixtures/mock_episodes.jsonl` (NEW - mock data)

**Deliverables:**
1. CLI tool that reads episodes/questions
2. Groups by: name prefix, operator sequence, both
3. Calculates pass rates per group
4. Outputs JSON + CLI table
5. Unit tests with mock data (test first, then implement)
6. Report to progress log when tests pass

**Dependencies:** None (uses mock data)

### Agent 4: Integration & Verification
**Status:** WAIT for Agents 1-3 to complete

**Files:**
- `src/datagen/validate_synthetic.py` (MODIFY - use episode factory)
- `src/datagen/episode_gen.py` (MODIFY - use episode factory)
- `src/datagen/shared/questions_io.py` (MODIFY - add is_procedural flag)
- `src/cli.py` (MODIFY - add procedural commands)
- `docs/AGENTS_PROGRESS.md` (MONITOR - review before starting)

**Deliverables:**
1. Refactor validate_synthetic.py to use episode factory
2. Refactor episode_gen.py to use episode factory
3. Add `is_procedural: bool` to QuestionRecord schema
4. Integrate dead code validator into program generator
5. End-to-end tests
6. Update documentation
7. Final verification report

**Dependencies:** Agents 1, 2, 3 complete (tests passing)

## Progress Log

**Location:** `docs/AGENTS_PROGRESS.md`

**Format:**
```markdown
## Agent X - [Status]

**Started:** YYYY-MM-DD HH:MM
**Files:** list of files being modified
**Progress:** 
- [ ] Task 1
- [ ] Task 2
**Blockers:** None / description
**Tests:** X/Y passing
**Completed:** YYYY-MM-DD HH:MM (or omit if in progress)
```

## Communication Protocol

1. **Start:** Agent appends "Started" entry to progress log
2. **Progress:** Agent updates progress log every 30 minutes or on milestone
3. **Blockers:** Agent reports blockers immediately, I will coordinate
4. **Completion:** Agent reports "Tests passed: X/Y" and "Completed"
5. **Conflicts:** If agent encounters "unexpected linter errors" or "file locked", report immediately

## Git Workflow

1. **Branches:** Each agent works on feature branch `agent/X-description`
2. **Commits:** Commit after each test pass (red → green)
3. **Messages:** Clear, descriptive commit messages
4. **No force push:** Never force push to shared branches
5. **Agent 4:** Will merge branches after verification

## Success Criteria

- All unit tests pass
- No dead code in procedural questions
- Episode factory used by all generators
- Analysis tool can report pass rates
- Integration tests pass
- Documentation updated

---

## Current Status

**Phase:** Planning complete, ready to launch agents
**Next Action:** Launch Agents 1, 2, 3 in parallel
