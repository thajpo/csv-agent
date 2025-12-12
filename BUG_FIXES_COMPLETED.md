# Bug Fixes Completed

## Summary

Two critical bugs in `src/core/kernel.py` have been fixed:

### 1. get_locals() RuntimeError (FIXED)
**Severity**: CRITICAL
**Impact**: Entire artifact tracking system was broken

**Problem**:
- `get_locals()` always returned empty dict `{}`
- Caused by `RuntimeError: dictionary changed size during iteration`
- Line 331 iterated over `globals().items()` while modifying globals

**Fix Applied**:
```python
# Line 331 - Changed from:
for _k, _v in globals().items():

# To:
for _k, _v in list(globals().items()):
```

**Verification**: ✅ All tests pass
- User variables captured correctly
- Artifact tracking works
- Final answer retrieval works

---

### 2. Baseline Artifact Exclusion (FIXED)
**Severity**: HIGH
**Impact**: False positive matches in reward calculation

**Problem**:
- Baseline variables (`df`, `pd`, `np`, `submit`) were captured as artifacts
- These exist in EVERY execution → always match → inflated rewards

**Fix Applied**:
```python
# Lines 79-86 - Track baseline variables in __init__:
if csv_path:
    self.setup_kernel_builtins(csv_path)
    self.baseline_vars = {
        'np', 'pd', 'scipy', 'sklearn', 'statsmodels', 'sm',
        'df', 'submit', '__SUBMITTED_ANSWER__'
    }

# Lines 368-371 - Exclude baseline in snapshot_artifacts():
for name, obj in locals_dict.items():
    if hasattr(self, 'baseline_vars') and name in self.baseline_vars:
        continue  # Skip baseline variables
```

**Verification**: ✅ All tests pass
- User-created variables captured: `mean_tl`, `df_control`, `count`
- Baseline variables excluded: `df`, `pd`, `np`, `submit`

---

## Test Results

```
=== COMPREHENSIVE get_locals() FIX VERIFICATION ===

Test 1: Basic get_locals() functionality
  ✓ PASS: User variables captured correctly

Test 2: Artifact capture with baseline exclusion
  Captured 5 artifacts: ['count', 'df_control', 'mean_tl', 'x', 'y']
  ✓ PASS: User vars captured, baseline vars excluded

Test 3: Final answer retrieval
  ✓ PASS: Final answer retrieved correctly: 42.5

✅ ALL TESTS PASSED
```

## Status

Both bugs are **FULLY FIXED** and verified. The artifact tracking system is now working correctly.

---

**Date Fixed**: 2025-12-12
**Files Modified**: `src/core/kernel.py` (lines 79-86, 331, 368-371)
