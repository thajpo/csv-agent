# Synthetic Question Generation: Improvement Spec

## Current State (Jan 2026)

**Pass Rate:** 26% (8/31 questions on insurance dataset)
**Failure Breakdown:**
- 30% Column Selection ambiguity
- 26% Include/Exclude target ambiguity
- 17% Parameter choice ambiguity
- 13% Method choice ambiguity
- 13% Unknown/other

---

## Implemented Fixes

### 1. Mechanical Questions (bypass verbalization)
- **Status:** ✅ Done
- **Config:** `config.synthetic_skip_verbalization = True`
- **Impact:** Removes LLM verbalization errors, questions are template descriptions

### 2. ANOVA Exclude-Target Alternative
- **Status:** ✅ Done
- **Change:** Added `alternative_code_templates` that uses second-highest variance column
- **Impact:** Accepts both "include target" and "exclude target" interpretations

---

## Pending Improvements

### Priority 1: Add Exclude-Target Alternatives (High Impact)

Templates needing "exclude target" alternatives:
- [ ] `ttest_discovered_groups` - binary categorical t-test
- [ ] `strongest_correlation` - already has some alts, verify coverage
- [ ] `spearman_rank_correlation` - rank correlation
- [ ] `regression_most_predictive` - predictive modeling
- [ ] `correlation_change_analysis` - log transform comparison

**Pattern:** In each, add alternative that excludes highest-variance column before analysis.

### Priority 2: Make Parameters Explicit (Medium Impact)

Templates with implicit parameters:
- [ ] `iterative_outlier_removal` - specify z-threshold (e.g., "z > 3.0")
- [ ] `count_outlier_columns` - specify z-threshold in description
- [ ] `iqr_outlier_analysis` - specify IQR multiplier (e.g., "1.5 × IQR")

**Pattern:** Update `description` field to include explicit parameter values.

### Priority 3: Multi-Path Column Selection (Medium Impact)

For templates that pick "first matching" column, compute all valid paths:
- [ ] Modify generator to enumerate all valid columns
- [ ] Execute template for each valid column
- [ ] Accept any valid answer

**Affected templates:**
- `anova_discovered_groups` - first categorical with 3-10 groups
- `ttest_discovered_groups` - first binary categorical
- `category_highest_target_mean` - first categorical
- `quantile_bin_best_mean` - first numeric for binning

### Priority 4: Method Choice Clarification (Low Impact)

Templates where multiple methods are valid:
- [ ] `ks_normality_test` - K-S vs Shapiro-Wilk vs Anderson-Darling
- [ ] `adaptive_two_sample_test` - decision logic for t-test vs Mann-Whitney

**Options:**
1. Be explicit in description ("use Kolmogorov-Smirnov")
2. Accept all standard methods as alternatives

---

## Future R&D: Verbalization Pipeline

**Goal:** Generate natural language questions that preserve semantic equivalence with template code.

### Fast Iteration Harness
```python
# 30-second feedback loop
CANARY_TEMPLATES = [max_variance_mean, strongest_correlation, ...]

for prompt_variant in [v1, v2, v3]:
    for template in CANARY_TEMPLATES:
        question = verbalize(template.code, prompt_variant)
        # Check for banned words
        assert "feature" not in question.lower()
        assert "target" not in question.lower()
        assert "excluding" not in hint.lower()
```

### Banned Words List
```python
ML_LOADED_TERMS = [
    "feature", "target", "predictor", "dependent", "independent",
    "excluding", "except", "without", "other than"
]
```

### Prompt Principles
1. No information loss from code → question
2. No added assumptions
3. Use neutral data analysis language ("column" not "feature")

---

## Automated Alternative Discovery (Future)

**Concept:** When student submits structurally valid but different answer, automatically propose it as alternative.

**Safeguards:**
- Human review before adding
- Max 3-5 alternatives per template
- Must use same statistical method
- Must answer same question (different valid interpretation)

**Pipeline:**
1. Student solves question
2. If hash mismatch but structure valid → flag for review
3. Human approves → add to `alternative_code_templates`
4. Re-run validation → higher pass rate

---

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Pass rate (insurance) | 26% | 70%+ |
| Templates with alternatives | ~10 | 25+ |
| Avg alternatives per template | 0.5 | 2-3 |
| Verbalization pass rate | N/A | 80%+ (future) |

---

## Quick Wins Checklist

- [x] Bypass verbalization (mechanical questions)
- [x] ANOVA exclude-target alternative
- [ ] T-test exclude-target alternative
- [ ] Explicit z-threshold in outlier templates
- [ ] Strongest correlation exclude-target alternative
- [ ] Re-run on multiple datasets to validate improvements
