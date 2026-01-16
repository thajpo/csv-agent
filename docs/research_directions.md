# Research Directions

> **Context**: This document captures research directions enabled by the csv-agent infrastructure, derived from a deep exploration of the codebase, literature review, and design discussions.

---

## Infrastructure Assets

What makes this system unique for research:

| Asset | What It Enables |
|-------|-----------------|
| **Hooks with `depends_on`** | DAG-based credit assignment, process reward models |
| **Ground truth hook values** | Oracle PRM comparisons, direct step verification |
| **Teacher triangulation** | Verification without human labels, hint ablation studies |
| **Difficulty labels** | Curriculum learning experiments |
| **Template metadata** | Holdout experiments, generalization studies |
| **Dual question sources** | Synthetic vs LLM-generated comparison |
| **Multi-outcome validation** | `ground_truth_hashes` for ambiguous questions |

---

## Tier 1: Ready Now

These experiments require no additional infrastructure.

### 1. PRM vs ORM Comparison

**Question**: Does dense process reward improve sample efficiency over outcome-only reward?

**Setup**:
- ORM: Reward = 1 if final answer correct, else 0
- PRM-fixed: Reward = Σ(hook correctness) with hand-designed weights
- PRM-learned: Train reward model on (partial_trace → correctness)

**Metrics**: Samples to reach 80% accuracy, final performance, learning curves

**Relevance**: CodePRM (ACL 2025), DreamPRM-Code show PRMs help for code—your hooks provide cleaner step boundaries than line-level approaches.

### 2. DAG-Aware vs Flat Credit Assignment

**Question**: Does propagating credit through `depends_on` structure help learning?

**Setup**:
```python
# Flat: penalize all wrong steps equally
reward = sum(correct[i] for i in steps) / len(steps)

# DAG-aware: only penalize root causes
reward = sum(correct[i] for i in steps
             if all(correct[p] for p in depends_on[i])) / len(steps)
```

**Hypothesis**: DAG-aware helps when failures cascade (step 3 wrong → steps 4-9 wrong). Flat is sufficient when failures are independent.

**Relevance**: "Towards Causal Credit Assignment" (arXiv 2212.11636) shows causal structure helps—your `depends_on` is explicit causal structure.

### 3. Curriculum by Difficulty/Steps

**Question**: What training progression yields fastest learning?

**Conditions**:
- Random sampling across difficulties
- EASY → MEDIUM → HARD → VERY_HARD progression
- Reverse curriculum (HARD → EASY)
- Adaptive (sample at model's frontier)

**Metrics**: Learning efficiency, final accuracy, generalization to held-out difficulty levels

### 4. Template Holdout (Generalization)

**Question**: Does training on N templates generalize to unseen template types?

**Setup**: Train on 22 templates, evaluate on 5 held-out templates

**What it tests**: Whether the model learns generalizable data analysis skills vs. memorizing template-specific algorithms

### 5. LLM vs Synthetic Question Comparison

**Question**: Which question source produces better training data?

**Conditions**:
- Train on synthetic only → eval on LLM-generated
- Train on LLM-generated only → eval on synthetic
- Train on both → eval on held-out of each

**What it tests**: Transfer between question sources, complementarity of data types

### 6. Hook Density as Convergence Metric

**Question**: Is hook-level accuracy predictive of final performance?

**Metrics**:
- `hook_hit_rate`: % of expected hooks produced
- `hook_accuracy`: % of produced hooks with correct values
- `hook_density`: hit_rate × accuracy

**Use case**: Early stopping, curriculum adaptation, debugging training

### 7. Hint Ablation Analysis

**Question**: What makes questions hint-dependent?

**Data**: Your triangulation compares gold (with hint) vs consistency (without hint)

**Analysis**:
- Correlation between hint-dependence and difficulty
- Characterize questions where hint is necessary vs redundant
- Proxy for question ambiguity?

### 8. Scaling Experiments

**Question**: How does performance scale with synthetic data quantity?

**Setup**: Train on {10%, 25%, 50%, 100%} of data, measure accuracy

**What it reveals**: Saturation point, diminishing returns, whether more data helps

---

## Tier 2: Needs Minor Fix

### 9. Multi-Outcome Validation

**Question**: How often do agents find valid alternative interpretations?

**Enabled by**: `ground_truth_hashes` field (now implemented)

**Analysis**:
- % of answers matching primary vs alternative interpretations
- Do some templates have higher alternative-match rates?
- Does training on multi-outcome improve robustness?

---

## Tier 3: Needs New Data/Infrastructure

### 10. Compositional Generalization

**Question**: Can agents compose learned operations in new ways?

**What's needed**: Either held-out template combinations, or a compositional grammar generating novel questions

**Current limitation**: 27 fixed templates don't compose

### 11. Line-Level vs Hook-Level Comparison

**Question**: Is semantic step granularity (hooks) better than syntactic (lines)?

**What's needed**: Capture line-level execution traces alongside hooks

**What it tests**: Whether hook abstraction level is optimal

### 12. Human Question Evaluation

**Question**: Do template-trained models generalize to real analyst questions?

**What's needed**: Collection of real analyst questions on held-out datasets

**Ultimate test**: Real-world task performance

---

## Literature Context

### Process Reward Models for Code

1. **CodePRM** — Li et al., "Execution Feedback-enhanced Process Reward Model for Code Generation," ACL Findings 2025, pp. 8169-8182. [[paper]](https://aclanthology.org/2025.findings-acl.428/)
   - Uses execution feedback to score thought steps in code generation
   - Key finding: "thought processes in code generation lack effective process supervision"
   - *Your hooks provide explicit step boundaries that CodePRM lacks*

2. **DreamPRM-Code** — "Function-as-Step Process Reward Model with Label Correction for LLM Coding," arXiv:2512.15000, December 2025. [[paper]](https://arxiv.org/abs/2512.15000)
   - Treats functions as reasoning steps via Chain-of-Function prompting
   - Achieves 80.9 pass@1 on LiveCodeBench (surpasses o4-mini)
   - *Your hooks are finer-grained than function boundaries*

3. **ThinkPRM** — "Process Reward Models That Think," arXiv:2504.16828, April 2025. [[paper]](https://arxiv.org/abs/2504.16828)
   - Generative long Chain-of-Thought PRMs that scale test-time compute
   - Outperforms discriminative verifiers trained on PRM800K
   - *Your ground-truth hook values could train or evaluate such models*

4. **GenPRM** — Liu et al., "Scaling Test-Time Compute of Process Reward Models via Generative Reasoning," AAAI 2026. [[code]](https://github.com/RyanLiu112/GenPRM)
   - Released GenPRM-1.5B and GenPRM-7B models
   - Focus on scaling test-time compute for verification

### Credit Assignment in Multi-Step Reasoning

5. **MT-GRPO** — "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment," ICML 2025. [[paper]](https://arxiv.org/abs/2505.11821)
   - Turn-level credit assignment for multi-turn agents
   - Key insight: "effective multi-turn reasoning requires more precise, turn-level credit assignment"
   - *Your hooks enable finer-than-turn-level (step-level) credit assignment*

6. **CAPO** — "Towards Enhancing LLM Reasoning through Generative Credit Assignment," arXiv:2508.02298, August 2025. [[paper]](https://arxiv.org/abs/2508.02298)
   - Uses LLM-as-GenPRM for step-wise critique
   - Addresses coarse-grained feedback in RLVR
   - *You have ground truth step values—no need for LLM-as-judge*

7. **OREO** — "Offline Reinforcement Learning for LLM Multi-step Reasoning," ACL Findings 2025. [[paper]](https://aclanthology.org/2025.findings-acl.464/)
   - Offline RL via soft Bellman Equation optimization
   - Joint policy and value function learning
   - *Could directly use your verified traces*

8. **GRPO-λ** — "Credit Assignment improves LLM Reasoning," arXiv:2510.00194, October 2025. [[paper]](https://arxiv.org/abs/2510.00194)
   - Re-parametrization of generalized advantage estimation for critic-free methods

### Causal/Structural Credit Assignment

9. **Towards Causal Credit Assignment** — arXiv:2212.11636, December 2022. [[paper]](https://arxiv.org/abs/2212.11636)
   - Hindsight Credit Assignment with causal structure
   - Key finding: "effectively exploits a given causal structure, which greatly decreases the workload"
   - *Your `depends_on` field IS explicit causal structure*

10. **MACCA** — "Offline Multi-agent Reinforcement Learning with Causal Credit Assignment," arXiv:2312.03644, December 2023. [[paper]](https://arxiv.org/abs/2312.03644)
    - Dynamic Bayesian Networks for causal relationships
    - Reformulates reward decomposition via causal structure

11. **Survey of Temporal Credit Assignment** — arXiv:2312.01072, December 2023. [[paper]](https://arxiv.org/abs/2312.01072)
    - Comprehensive survey of credit assignment in deep RL
    - Notes: "causal structure... is often a valuable property to incorporate"

### Dense Rewards

12. **G4RL** — "Towards better dense rewards in Reinforcement Learning Applications," arXiv:2512.04302, December 2025. [[paper]](https://arxiv.org/abs/2512.04302)
    - Graph-Guided subGoal representation for dense rewards
    - *Your hook dependency graph could inform reward shaping*

13. **Dense Reasoning Reward via Inverse RL** — "Learning a Dense Reasoning Reward Model from Expert Demonstration," arXiv:2510.01857, October 2025. [[paper]](https://arxiv.org/abs/2510.01857)
    - Learns token-level rewards from expert demonstrations
    - *Your teacher traces ARE expert demonstrations with step labels*

### PRM Lessons and Challenges

14. **Lessons of Developing PRMs** — "The Lessons of Developing Process Reward Models in Mathematical Reasoning," arXiv:2501.07301, January 2025. [[paper]](https://arxiv.org/abs/2501.07301)
    - Practical lessons from building PRMs for math
    - Discusses challenges in step definition and label noise

---

## Non-Obvious Hypotheses

Interesting because smart people could disagree:

1. **Strict verification hurts generalization** — Over-verification creates artificially clean data; models never learn to handle ambiguity

2. **Template diversity beats difficulty** — 50 EASY templates outperform 10 VERY_HARD for generalization (breadth > depth)

3. **Synthetic templates create negative transfer** — Template-trained models perform worse than base on truly novel questions

4. **Trace structure matters more than question diversity** — 5 templates with rich hook traces > 50 templates with input/output only

---

## Experimental Design Principles

For rigorous results:

1. **Controlled comparisons** — Same model, same eval, only training data varies
2. **Multiple seeds** — Report variance, not just best run
3. **Held-out evaluation** — Eval must be truly independent of training data design
4. **Mechanism, not just effect** — Explain WHY something works, not just THAT it works
5. **Predict failure modes** — Strong hypotheses specify when they'll fail

---

## Quick Reference: Metadata for Experiments

All filtering done post-generation via episode metadata:

```python
# Template holdout
train = [e for e in episodes if e["question"]["template_name"] not in HELD_OUT]

# Difficulty curriculum
easy_first = sorted(episodes, key=lambda e: DIFFICULTY_ORDER[e["question"]["difficulty"]])

# CSV holdout
train = [e for e in episodes if e["csv_source"] not in HELD_OUT_CSVS]

# Multi-outcome scoring
correct = answer_hash in episode["question"]["ground_truth_hashes"]
```

---

*Last updated: 2025-01-04*
