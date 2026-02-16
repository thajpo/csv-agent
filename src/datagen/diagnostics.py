"""
Failure diagnostics for triangulation results.

Provides analysis tools to classify WHY a triangulation failed:
- AMBIGUOUS: Multiple valid interpretations (high entropy)
- TOO_HARD: Model consistently gets wrong answer (low entropy, wrong cluster)
- HINT_NECESSARY: Gold only works with hint

Usage:
    from src.datagen.diagnostics import analyze_failure

    diagnostics = analyze_failure(gold_trace, consistency_traces)
    print(diagnostics["failure_category"])  # "ambiguous", "too_hard", etc.
"""

import math
from typing import Any

from csv_spec import (
    hash_artifact,
    TraceDict,
    AnswerClusterDict,
    AnswerDistributionDict,
    DiagnosticMetadataDict,
    FailureCategory,
)


def compute_entropy(counts: list[int]) -> float:
    """
    Compute Shannon entropy of a distribution.

    High entropy = answers spread across many clusters (ambiguous)
    Low entropy = answers concentrated in few clusters (deterministic)

    Returns:
        Entropy in bits. 0 = single cluster, log2(n) = uniform across n clusters.
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def analyze_answer_distribution(
    consistency_traces: list[TraceDict],
    float_tol: float = 0.1,
) -> AnswerDistributionDict:
    """
    Cluster consistency answers and compute distribution statistics.

    This is the core diagnostic function - it reveals whether failures
    are due to ambiguity (multiple clusters) or difficulty (single wrong cluster).

    Args:
        consistency_traces: List of traces from consistency runs (without hint)
        float_tol: Tolerance for float comparison when clustering

    Returns:
        Distribution analysis with clusters, entropy, and confidence metrics.
    """
    # Import here to avoid circular dependency
    from src.datagen.teacher import answers_match

    total = len(consistency_traces)

    # Separate successful vs failed traces
    successful_answers: list[tuple[int, Any]] = []  # (index, answer)
    for i, trace in enumerate(consistency_traces):
        if trace.get("success", False) and trace.get("final_answer") is not None:
            successful_answers.append((i, trace["final_answer"]))

    execution_failures = total - len(successful_answers)

    if not successful_answers:
        return AnswerDistributionDict(
            total_traces=total,
            successful_traces=0,
            execution_failures=execution_failures,
            cluster_count=0,
            entropy=0.0,
            majority_confidence=0.0,
            clusters=[],
        )

    # Cluster answers using tolerance-aware matching
    clusters: list[AnswerClusterDict] = []

    for idx, answer in successful_answers:
        answer_hash = hash_artifact(answer)
        found = False

        for cluster in clusters:
            if answers_match(
                cluster["answer_hash"],
                answer_hash,
                cluster["representative_answer"],
                answer,
                float_tol=float_tol,
            ):
                cluster["member_count"] += 1
                cluster["member_indices"].append(idx)
                found = True
                break

        if not found:
            clusters.append(AnswerClusterDict(
                answer_hash=answer_hash,
                member_count=1,
                representative_answer=answer,
                member_indices=[idx],
            ))

    # Sort by count descending (majority first)
    clusters.sort(key=lambda c: c["member_count"], reverse=True)

    # Compute entropy
    counts = [c["member_count"] for c in clusters]
    entropy = compute_entropy(counts)

    # Majority confidence = what fraction agreed on the top answer
    majority_count = clusters[0]["member_count"] if clusters else 0
    majority_confidence = majority_count / len(successful_answers) if successful_answers else 0.0

    return AnswerDistributionDict(
        total_traces=total,
        successful_traces=len(successful_answers),
        execution_failures=execution_failures,
        cluster_count=len(clusters),
        entropy=entropy,
        majority_confidence=majority_confidence,
        clusters=clusters,
    )


def classify_failure(
    gold_trace: TraceDict,
    distribution: AnswerDistributionDict,
    float_tol: float = 0.1,
) -> DiagnosticMetadataDict:
    """
    Classify a triangulation outcome into diagnostic categories.

    Classification logic:
    1. GOOD: gold matches majority cluster
    2. EXECUTION_FAILURE: most traces failed to execute
    3. AMBIGUOUS: high entropy (multiple clusters with similar sizes)
    4. TOO_HARD: low entropy but gold not in majority
    5. HINT_NECESSARY: gold succeeds but consistency mostly fails

    Args:
        gold_trace: The trace from running WITH hint
        distribution: Answer distribution from analyze_answer_distribution()
        float_tol: Tolerance for float comparison

    Returns:
        Diagnostic metadata with category, reasoning, and confidence.
    """
    from src.datagen.teacher import answers_match

    gold_success = gold_trace.get("success", False)
    gold_answer = gold_trace.get("final_answer")
    gold_hash = hash_artifact(gold_answer) if gold_answer is not None else None

    # Check if gold matches any cluster
    gold_cluster_idx = -1
    gold_matches_majority = False

    if gold_answer is not None and distribution["clusters"]:
        for i, cluster in enumerate(distribution["clusters"]):
            if answers_match(
                gold_hash,
                cluster["answer_hash"],
                gold_answer,
                cluster["representative_answer"],
                float_tol=float_tol,
            ):
                gold_cluster_idx = i
                gold_matches_majority = (i == 0)  # First cluster is majority
                break

    # Classification logic
    category: FailureCategory
    confidence: float
    reasoning: str

    # Case 1: Success - gold matches majority
    if gold_matches_majority:
        category = FailureCategory.GOOD
        confidence = distribution["majority_confidence"]
        reasoning = f"Gold matches majority cluster ({distribution['majority_confidence']:.0%} agreement)"

    # Case 2: Execution failures dominate
    elif distribution["execution_failures"] > distribution["successful_traces"]:
        category = FailureCategory.EXECUTION_FAILURE
        confidence = distribution["execution_failures"] / distribution["total_traces"]
        reasoning = f"{distribution['execution_failures']}/{distribution['total_traces']} traces failed to execute"

    # Case 3: Gold didn't execute but consistency did
    elif not gold_success and distribution["successful_traces"] > 0:
        category = FailureCategory.HINT_NECESSARY
        confidence = 0.8
        reasoning = "Gold trace failed but consistency traces succeeded - hint may be misleading"

    # Case 4: High entropy = AMBIGUOUS (multiple valid interpretations)
    # Threshold: entropy > 0.8 means answer spread across clusters
    elif distribution["cluster_count"] >= 2 and distribution["entropy"] > 0.8:
        category = FailureCategory.AMBIGUOUS
        confidence = min(1.0, distribution["entropy"] / 1.5)
        top_clusters = distribution["clusters"][:3]
        cluster_desc = ", ".join(
            f"C{i}:{c['member_count']}" for i, c in enumerate(top_clusters)
        )
        reasoning = f"High answer diversity: {distribution['cluster_count']} clusters, entropy={distribution['entropy']:.2f}. Votes: [{cluster_desc}]"

    # Case 5: Low entropy but gold not in majority = TOO_HARD
    elif distribution["majority_confidence"] > 0.5 and not gold_matches_majority:
        category = FailureCategory.TOO_HARD
        confidence = distribution["majority_confidence"]
        if gold_cluster_idx >= 0:
            reasoning = f"Strong majority ({distribution['majority_confidence']:.0%}) but gold in cluster {gold_cluster_idx}"
        else:
            reasoning = f"Strong majority ({distribution['majority_confidence']:.0%}) and gold doesn't match any cluster"

    # Case 6: Gold success + consistency mostly fails = HINT_NECESSARY
    elif gold_success and distribution["successful_traces"] < distribution["total_traces"] * 0.5:
        category = FailureCategory.HINT_NECESSARY
        success_rate = distribution["successful_traces"] / distribution["total_traces"]
        confidence = 1.0 - success_rate
        reasoning = f"Gold succeeded with hint but only {distribution['successful_traces']}/{distribution['total_traces']} consistency traces succeeded"

    # Default: AMBIGUOUS (unclear pattern)
    else:
        category = FailureCategory.AMBIGUOUS
        confidence = 0.5
        reasoning = f"Unclear pattern: {distribution['cluster_count']} clusters, entropy={distribution['entropy']:.2f}, majority={distribution['majority_confidence']:.0%}"

    return DiagnosticMetadataDict(
        failure_category=category.value,
        answer_distribution=distribution,
        gold_answer_hash=gold_hash,
        gold_execution_success=gold_success,
        gold_matches_majority=gold_matches_majority,
        gold_cluster_index=gold_cluster_idx if gold_cluster_idx >= 0 else None,
        classification_confidence=confidence,
        classification_reasoning=reasoning,
    )


def analyze_failure(
    gold_trace: TraceDict,
    consistency_traces: list[TraceDict],
    float_tol: float = 0.1,
) -> DiagnosticMetadataDict:
    """
    Convenience function: analyze and classify a triangulation failure in one call.

    Args:
        gold_trace: The trace from running WITH hint
        consistency_traces: List of traces from running WITHOUT hint
        float_tol: Tolerance for float comparison

    Returns:
        Complete diagnostic metadata with category, distribution, and reasoning.
    """
    distribution = analyze_answer_distribution(consistency_traces, float_tol)
    return classify_failure(gold_trace, distribution, float_tol)


def format_diagnostic_summary(diagnostics: DiagnosticMetadataDict) -> str:
    """
    Format diagnostics as a human-readable summary string.

    Useful for logging and CLI output.
    """
    dist = diagnostics["answer_distribution"]
    lines = [
        f"Category: {diagnostics['failure_category'].upper()}",
        f"Confidence: {diagnostics['classification_confidence']:.0%}",
        f"Reasoning: {diagnostics['classification_reasoning']}",
        "",
        "Distribution:",
        f"  Traces: {dist['successful_traces']}/{dist['total_traces']} succeeded",
        f"  Clusters: {dist['cluster_count']}",
        f"  Entropy: {dist['entropy']:.2f}",
        f"  Majority: {dist['majority_confidence']:.0%}",
    ]

    if diagnostics.get("gold_matches_majority"):
        lines.append("  Gold: matches majority ✓")
    elif diagnostics.get("gold_cluster_index") is not None:
        lines.append(f"  Gold: in cluster {diagnostics['gold_cluster_index']} (not majority)")
    else:
        lines.append("  Gold: no cluster match")

    return "\n".join(lines)
