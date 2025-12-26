# Research Directions: Process Reward Models for CSV Agents

*Last updated: December 2024*
*Status: Exploration phase - pre-SFT/GRPO training*

This document captures research ideas, failure modes, and future directions discussed during the design of a PRM-based training pipeline for CSV analysis agents.

---

## Table of Contents

1. [Current Position](#current-position)
2. [DeepSeek-Math-V2 Insights](#deepseek-math-v2-insights)
3. [What We Have vs What We Need](#what-we-have-vs-what-we-need)
4. [Failure Modes and Concerns](#failure-modes-and-concerns)
5. [Potential Solutions](#potential-solutions)
6. [The Core Tension](#the-core-tension)
7. [Phased Approach](#phased-approach)
8. [Concrete Experiments](#concrete-experiments)
9. [Open Questions](#open-questions)
10. [Immediate Next Steps](#immediate-next-steps)

---

## Current Position

### What We've Built

- **Synthetic data generation pipeline**: Templates generate deterministic code trajectories, then verbalize into natural language questions
- **Hook system**: Captures intermediate computational values with `hook(value, code_line, name, depends_on)`
- **Teacher triangulation**: Gold trace (with hint) + N consistency traces (without hint) to verify answers
- **Ground truth**: We always know the correct final answer and intermediate values because we execute the template first

### What We Haven't Done Yet

- SFT training on generated episodes
- GRPO/RL training with hook-based rewards
- Process reward model training
- Evaluation of trained models

### Key Insight

We have ~70% of the infrastructure for PRM-style training. The hooks are step boundaries. The triangulation gives us multiple traces per question. The missing pieces:
1. Store raw hook values (not just hashes) for tolerance-based comparison
2. Compute step-level consensus across traces
3. Build or use a learned verifier

---

## DeepSeek-Math-V2 Insights

### What They Did

DeepSeek-Math-V2 (Nov 2025) trains a **verifier** that scores proofs (0, 0.5, 1), then uses the verifier as a reward model to train a **generator**. Key innovations:

1. **Verifier Training**: LLM learns to identify issues in proofs and assign scores
2. **Meta-Verification**: A meta-verifier checks if the verifier's identified issues are real (prevents hallucinated critiques)
3. **Self-Verification**: Generator produces solution + self-critique in one pass, gets rewarded for both correctness and calibration
4. **Synergy Loop**: Better generator → harder proofs → better verifier → better generator

### Mapping to Our Domain

| DeepSeek-Math-V2 | CSV-Agent |
|------------------|-----------|
| Verifier scores proofs | Could train verifier to score code solutions |
| Step-by-step proof analysis | Hook-by-hook execution trace |
| Meta-verification | Triangulation (gold vs consistency majority) |
| Self-verification prompts | Not implemented yet |
| Formal proof structure | Deterministic code execution |

### Key Difference

DeepSeek works in a domain where verification is hard (natural language proofs). We work in a domain where verification is easy (just execute the code). This is both an advantage and a limitation - see [The Core Tension](#the-core-tension).

---

## What We Have vs What We Need

### Current Hook Data

```python
# Each hook stores:
{
    "variable_name": "filtered_df",
    "value_hash": "abc123...",  # SHA256 of normalized value
    "code_line": "filtered_df = df[df['col'] > 5]",
    "depends_on": ["df"]
}
```

### What We Need for PRM

```python
# Enhanced hook with raw value:
{
    "variable_name": "filtered_df",
    "value_hash": "abc123...",
    "value": {...},  # Actual normalized value (for tolerance comparison)
    "code_line": "filtered_df = df[df['col'] > 5]",
    "depends_on": ["df"],
    "step_confidence": 0.8,  # Fraction of traces that match at this step
    "is_consensus_correct": true  # Majority of traces agree with gold here
}
```

### Immediate Code Change Needed

**TODO**: Add `value` field to hooks in:
- `src/envs/csv_env.py` - `hook()` function in `_SETUP_SUBMIT`
- `src/core/types.py` - `Hook` model and `HookDict`
- `src/datagen/teacher.py` - hook extraction

This enables tolerance-based step comparison using existing `answers_match()` function.

---

## Failure Modes and Concerns

### 1. Questions Aren't Questions

**Problem**: Templates produce procedures ("find column with highest variance, compute its mean") not questions ("why did sales drop?"). The model learns to execute recipes, not form hypotheses.

**Evidence**: A human reading the verbalized question can probably reverse-engineer the template structure.

**Severity**: High. This limits transfer to real-world data analysis.

### 2. No Dead Ends

**Problem**: Templates always work. The model never learns to recognize "this approach isn't working, let me try something else."

**Evidence**: Every template is designed to produce an answer.

**Severity**: Medium. Real analysis involves lots of dead ends.

### 3. Verbalization is Lipstick

**Problem**: We generate the solution first, then make it sound like a question. The procedural structure leaks through. This is the same problem as #1.

**Severity**: High. Tightly coupled to #1.

### 4. No Ambiguity

**Problem**: Each template has ONE right answer. Real questions like "what predicts churn?" have multiple valid approaches. The model never learns to make judgment calls.

**Severity**: Medium-High. We have deterministic script generation from data.

### 5. Hooks Might Be Too Helpful

**Problem**: Dense step rewards mean the model gets credit for every correct intermediate. In the real world, there's no hook oracle. The model might become dependent on scaffolding.

**Counter**: PRMs are specifically designed to provide dense signal. The question is whether the model internalizes the verification or games it.

**Severity**: Medium. Needs empirical testing.

### 6. Reward Hacking Risk

**Problem**: Model might learn to pattern-match templates because it knows certain intermediate values get rewarded. It could produce correct hooks without understanding why.

**Mitigation**: Process rewards should evaluate reasoning, not just value matching. But that requires a learned verifier.

---

## Potential Solutions

### For "Questions Aren't Questions" (#1, #3)

**Option A**: Flip the loop
- Let model explore dataset freely and propose hypotheses
- Use templates as verifiers: "the model claimed X, can we construct a template that checks X?"
- Model learns to ask questions, not just answer them

**Option B**: Ground in real questions
- Collect real analyst questions (Stack Overflow, Kaggle, data science forums)
- Match templates that could answer them
- Train on questions that humans actually ask

**Option C**: Adversarial verbalization
- Train discriminator to detect template-generated vs real questions
- Use discriminator signal to push verbalizer toward natural phrasing

**Option D**: Skip verbalization for training
- Train on (dataset, trajectory, outcome) directly
- Model learns what analysis is appropriate for what data patterns
- Add natural language conditioning only at eval time

### For "No Dead Ends" (#2)

**Option A**: Generate distractor templates
- Create templates that plausibly look like they should work but don't for a given dataset
- "Find correlation between X and Y" where X and Y are independent
- Model must learn to recognize and backtrack

**Option B**: Partial credit for recognizing stuck
- Reward model for saying "this approach isn't leading anywhere"
- Requires training signal for when to give up

### For "No Ambiguity" (#4)

**Option A**: Multiple valid answers
- For questions like "what predicts churn?", encode several valid approaches
- Reward isn't "match THE answer" but "find A valid answer"
- Verification becomes: "does the conclusion follow from the computation?"

**Option B**: Learned verifier
- Train model to judge whether reasoning chain justifies conclusion
- Not hash matching, but semantic evaluation
- Bootstrapping this is tricky and error-prone

### For "Hooks Too Helpful" (#5)

**Option A**: Train with hooks, eval without
- Use dense rewards during training
- Remove hook oracle at inference time
- Test if model internalized verification

**Option B**: Hook curriculum
- Start with hook after every line
- Gradually reduce to major checkpoints only
- Model learns to self-verify intermediate steps

---

## The Core Tension

**PRMs are about judging process, not outcome. Our synthetic setup gives perfect outcome labels and imperfect process labels.**

We know the final answer is right. We can verify intermediate values match. But we can't verify that the *reasoning* is right - only that the *computation* is right.

A model could learn to produce correct hooks by pattern matching templates, never understanding why those steps make sense. The PRM would say "yes, each step matches ground truth" but the model learned nothing about reasoning.

**The question**: Does correct computation imply correct reasoning for this domain?

Maybe yes: if you get all intermediate values right, you probably understood the problem.

Maybe no: the model memorizes "when you see 'highest variance', do `df.var().idxmax()`" without understanding what variance means.

---

## Phased Approach

### Phase 1: Prove Dense Rewards Help (Tractable Now)

**Goal**: Show that hook-based step rewards improve training vs sparse outcome-only rewards.

**Method**:
1. Train model A with outcome reward only (final answer correct/incorrect)
2. Train model B with dense hook rewards (each step correct/incorrect)
3. Compare on held-out questions

**Success criteria**: Model B outperforms Model A on same question distribution.

**Why this matters**: Publishable result even with formulaic synthetic questions. Establishes that dense process supervision helps.

### Phase 2: Question Quality (Separate Problem)

**Goal**: Generate questions that require judgment, not just execution.

**Method**:
- Experiment with question generation approaches (Options A-D above)
- Metric: Can a human distinguish synthetic vs real analyst questions?
- Metric: Does the model generalize to real questions after training on improved synthetic ones?

**Why this matters**: Addresses the "questions aren't questions" problem without conflating it with PRM machinery.

### Phase 3: Self-Improvement Loop

**Goal**: Build the DeepSeek-style feedback loop where generator learns to verify itself.

**Method**:
1. Train verifier on (solution, hooks, ground_truth_hooks) → score
2. Add self-verification to generator prompts
3. Reward = correctness + calibration (did it know when it was wrong?)
4. As generator improves, generate harder questions
5. Scale verification compute for hard cases to improve verifier

**Why this matters**: This is the actual research contribution - showing self-improvement works in code execution domain.

---

## Concrete Experiments

### Experiment 1: SFT Baseline

**Question**: Does SFT on gold traces improve model performance on CSV analysis?

**Setup**:
- Train on `conversation_for_sft` from verified episodes
- Eval on held-out questions from same distribution
- Metric: Final answer accuracy

**Expected outcome**: Yes, obviously. But need to establish baseline.

### Experiment 2: Dense vs Sparse Rewards

**Question**: Do hook-based rewards outperform outcome-only rewards?

**Setup**:
- GRPO with reward = final_answer_correct only
- GRPO with reward = weighted sum of hook_correct + final_answer_correct
- Compare sample efficiency and final performance

**Expected outcome**: Dense rewards should help, especially for multi-step questions.

### Experiment 3: Generalization Test

**Question**: Does the model learn reasoning or template memorization?

**Setup**:
- Training: "find column with highest variance, compute its mean"
- Test: "find column with highest skewness, compute its median"
- Same reasoning structure, different specifics

**Expected outcome**: If model generalizes, it learned something. If it fails, it memorized.

### Experiment 4: Step Consensus Analysis

**Question**: Do hooks that agree across traces predict final answer correctness?

**Setup**:
- For each episode, compute step_confidence = fraction of traces that match at each hook
- Correlate step_confidence with final verification success
- Identify where divergence typically occurs

**Expected outcome**: Early divergence predicts failure. This validates using hooks for process supervision.

### Experiment 5: Long-Context Hallucination

**Question**: Can we detect hallucination using execution traces?

**Setup**:
- Generate long chains (20+ steps)
- Track where model claims differ from actual execution output
- Measure hallucination rate as function of trace length

**Expected outcome**: Hallucination increases with length. Hooks catch it precisely.

**Potential contribution**: "Using execution traces to detect and penalize hallucination in multi-step reasoning"

---

## Open Questions

### Fundamental

1. **What would convince us that a model learned to reason vs pattern match?**
   - Generalization to new template structures?
   - Transfer to real analyst questions?
   - Ability to explain why each step is necessary?

2. **Is "correct computation" sufficient for "correct reasoning" in this domain?**
   - Maybe they're close enough for CSV analysis
   - Maybe we need semantic verification, not just value matching

3. **How do we generate questions that require judgment?**
   - Multiple valid approaches to same question
   - Questions with implicit rather than explicit structure
   - Questions that require recognizing dead ends

### Technical

4. **How do we bootstrap a learned verifier without reward hacking?**
   - Start from triangulation signal?
   - Use LLM-as-judge (expensive)?
   - Self-consistency signals?

5. **What's the right hook density for training?**
   - Too dense: model depends on external verification
   - Too sparse: back to outcome-only rewards
   - Curriculum?

6. **How do we handle tolerance in step comparison?**
   - Hash matching misses equivalent answers with float differences
   - `answers_match()` exists for final answers, need to apply to hooks
   - Need raw values stored (not just hashes)

---

## Immediate Next Steps

### Before Training

1. **Add raw values to hooks** (code change)
   - Store actual normalized value alongside hash
   - Enable tolerance-based step comparison

2. **Compute step consensus** (analysis)
   - For existing episodes, compare hooks across traces
   - Identify typical divergence points
   - Validate that consensus predicts success

3. **Check data quantity**
   - How many verified episodes do we have?
   - Is it enough for SFT? (probably need 1000+)
   - Generate more if needed

### Training Phase 1

4. **SFT baseline**
   - Fine-tune on gold traces
   - Eval on held-out questions
   - Establish baseline metrics

5. **GRPO with hooks**
   - Implement hook-based reward in `rl_rubric.py`
   - Compare dense vs sparse rewards
   - Document findings

### Analysis

6. **Generalization test**
   - Create held-out questions with same structure, different specifics
   - Measure transfer performance
   - Determine if model learned reasoning or memorized

---

## References

- DeepSeek-Math-V2 paper (Nov 2025): Self-verifiable mathematical reasoning with verifier-generator loop
- PRM800K: Human-labeled step-level correctness for math reasoning
- Math-Shepherd (2024): Monte Carlo estimation of step correctness
- PRIME (2025): Process reinforcement through implicit rewards

---

## Notes

*This document is a living record. Update as experiments are run and understanding evolves.*

The core research question: **Can a model learn to know when it's wrong, and use that to improve?**

Our domain is a clean testbed because:
- Verification is cheap (just execute)
- Ground truth is absolute (deterministic templates)
- Dense signals are available (hooks)
- Difficulty is controllable (template complexity)

The main risk is that the domain is too easy / too formulaic to produce interesting findings. But that's an empirical question.
