# TODO: Future Improvements

This file tracks planned improvements and technical debt for future sessions.

---

## Loose Ends

### 1. Delete Stale Plan File

**Description:** Remove the obsolete plan file at `/home/j/.claude/plans/keen-percolating-diffie.md` which describes completed work.

**Reason:** Keep the workspace clean and avoid confusion in future sessions.

**Estimated Time:** 1 minute

---

### 2. Remove Backwards Compatibility Alias

**Description:** Remove `ContainerPool = MultiTenantContainer` alias from `src/envs/container_pool.py:654`.

**Reason:** The alias was added during the rename but nothing uses it. Dead code should be removed.

**Estimated Time:** 2 minutes

---

### 3. Add Triangulation Integration Test

**Description:** Create `tests/test_triangulation.py` that tests `batch_triangulate()` with the new `MultiTenantContainer` using mocked LLM responses.

**Reason:** Current tests cover components (container, adapter, security) but not the full triangulation flow. Integration tests catch interface mismatches.

**Estimated Time:** 1-2 hours

---

### 4. End-to-End Testing with Real LLM

**Description:** Run `episode_gen.py --parallel` with 2-3 small CSVs and real API calls to validate the full pipeline.

**Reason:** Architecture is theoretically sound but edge cases (API timeouts, container races, memory pressure) only emerge under real load.

**Estimated Time:** 30 minutes (plus debugging if issues arise)

---

## Near-Term Features

### 5. Add Progress Indicator for Parallel Mode

**Description:** Show progress feedback during parallel CSV processing. Currently parallel mode is silent until all CSVs complete.

**Reason:** Long-running parallel jobs with no feedback make users uncertain if progress is being made.

**Estimated Time:** 1-2 hours

**Implementation Notes:**
- Could use `asyncio.as_completed()` instead of `gather()` to report as each CSV finishes
- Or periodic status updates via a background task

---

### 6. Memory Profiling

**Description:** Measure actual memory savings from fork-based worker sharing vs. separate containers.

**Reason:** We estimated 6x memory reduction but haven't validated. Real numbers inform capacity planning.

**Estimated Time:** 2-3 hours

**Implementation Notes:**
- Use `docker stats` or `psutil` to measure container memory
- Compare: N separate containers vs. 1 container with N fork workers
- Document findings in `docs/architecture.md`

---

## Medium-Term Features

### ~~7. Semaphore for Max Concurrent Containers~~ ✅ DONE

Implemented via `config.max_concurrent_containers` (default: 10) with asyncio.Semaphore in `episode_gen.py`.

---

### 8. Resume Capability for Parallel Processing

**Description:** If parallel processing fails mid-batch, ability to resume from last checkpoint.

**Reason:** Long-running jobs shouldn't lose all progress on failure. Especially important for large batches.

**Estimated Time:** 4-6 hours

**Implementation Notes:**
- Write progress to a state file after each CSV completes
- On startup, check for incomplete state file and resume
- Consider using SQLite for atomic progress tracking

---

### 9. Per-CSV Progress Logging

**Description:** In parallel mode, write progress to per-CSV log files for debugging.

**Reason:** Interleaved stdout is unusable for debugging. Separate logs enable post-hoc analysis.

**Estimated Time:** 1-2 hours

**Implementation Notes:**
- Create `logs/{dataset_name}.log` for each CSV
- Use Python's `logging` module with file handlers

---

### 10. Container Health Monitoring

**Description:** Detect and restart workers that become unresponsive (e.g., stuck in infinite loop, OOM killed).

**Reason:** A stuck worker blocks progress. Auto-recovery improves reliability for long batches.

**Estimated Time:** 4-6 hours

**Implementation Notes:**
- Add timeout to `run_on_worker()` (already partially exists)
- On timeout, kill and respawn the worker process
- Log incidents for debugging

---

## Longer-Term / Exploratory

### 11. Distributed Execution

**Description:** Run containers across multiple machines (Kubernetes, cloud VMs, RunPod).

**Reason:** Scale beyond single-machine limits for large-scale data generation.

**Estimated Time:** 2-4 days

---

### 12. Persistent Worker Pools

**Description:** Keep containers warm between `episode_gen.py` runs to eliminate startup overhead.

**Reason:** Container startup is ~30-60 seconds. For iterative development, this adds up.

**Estimated Time:** 4-8 hours

---

### 13. Question Generation Parallelization

**Description:** Apply same multi-tenant container pattern to `question_gen.py`.

**Reason:** Question generation also runs code in sandbox. Same memory benefits apply.

**Estimated Time:** 2-4 hours (mostly copy-paste from episode_gen pattern)

---

## Completed

- [x] Fix Kaggle download script for new API (`7b2aa49`)
- [x] Add sandbox security restrictions (`57112de`)
- [x] Implement MultiTenantContainer with fork-based workers (`2ec8866`)
- [x] Integrate with triangulation pipeline (`b8beffc`)
- [x] Add `--parallel` flag to episode_gen.py
- [x] Clean up config (remove unused `n_containers`, `n_workers_per_csv`)
- [x] Add `max_concurrent_containers` config with semaphore throttling
