"""Build canonical PRM process reports from episode traces."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from csv_spec import (
    ProcessReportDict,
    ProcessReportSummaryDict,
    ProcessStepEvidenceDict,
    ProcessStepReportDict,
    TraceDict,
    hash_artifact,
)
from src.core.environment import validate_hooks_grounded

PROCESS_REPORT_VERSION = "prm_process_report_v1"
EpisodeSource = Literal["llm_gen", "template", "procedural"]


def build_process_report(
    *,
    source: EpisodeSource,
    gold_trace: TraceDict,
    consistency_traces: list[TraceDict],
    verified: bool,
    majority_count: int = 0,
) -> ProcessReportDict:
    """Build a step-level process report from gold and consistency traces."""
    steps: list[ProcessStepReportDict] = []
    code_cells = [
        turn.get("code", "")
        for turn in gold_trace.get("turns", [])
        if turn.get("code")
    ]
    consensus = _consensus_counts(consistency_traces)
    consensus_total = sum(1 for trace in consistency_traces if trace.get("success"))
    seen_names: set[str] = set()
    seen_keys: Counter[tuple[str, str | None, str | None]] = Counter()

    for turn_index, turn in enumerate(gold_trace.get("turns", [])):
        execution = turn.get("execution", {})
        hooks = execution.get("hooks", [])
        hook_copies = [dict(hook) for hook in hooks]
        grounded_hooks, _ungrounded_hooks = validate_hooks_grounded(
            hook_copies, code_cells
        )
        grounded_ids = {id(hook) for hook in grounded_hooks}

        for hook_index, hook in enumerate(hooks):
            evidence = _hook_evidence(
                hook=hook,
                trace_success=gold_trace.get("success", False),
                final_verified=verified,
                grounded=id(hook_copies[hook_index]) in grounded_ids,
                seen_names=seen_names,
                seen_keys=seen_keys,
                consensus=consensus,
                consensus_total=consensus_total,
            )
            label, confidence, label_source = _label_hook(
                source=source,
                evidence=evidence,
            )
            step = ProcessStepReportDict(
                step_index=len(steps),
                turn_index=turn_index,
                hook_index=hook_index,
                step_type="hook",
                code_line=hook.get("code_line", ""),
                variable_name=hook.get("variable_name"),
                value=hook.get("value"),
                value_hash=hook.get("value_hash"),
                description=hook.get("description"),
                depends_on=hook.get("depends_on", []),
                semantic_role=_semantic_role(hook),
                label=label,
                confidence=confidence,
                label_source=label_source,
                evidence=evidence,
            )
            steps.append(step)

            name = hook.get("variable_name")
            if name:
                seen_names.add(name)
            seen_keys[_step_key(step)] += 1

        submitted = execution.get("submitted_answer")
        if submitted is not None:
            answer_hash = gold_trace.get("final_answer_hash") or hash_artifact(submitted)
            evidence = ProcessStepEvidenceDict(
                final_verified=verified,
                trace_success=gold_trace.get("success", False),
                code_line_grounded=True,
                dependency_valid=True,
                duplicate=False,
                consensus_matches=majority_count,
                consensus_total=max(consensus_total, majority_count),
                reasons=[],
            )
            label, confidence, label_source = _label_submit(
                source=source,
                evidence=evidence,
            )
            steps.append(
                ProcessStepReportDict(
                    step_index=len(steps),
                    turn_index=turn_index,
                    hook_index=None,
                    step_type="submit",
                    code_line="submit(...)",
                    variable_name="answer",
                    value=submitted,
                    value_hash=answer_hash,
                    description="Final submitted answer",
                    depends_on=[],
                    semantic_role="final_answer",
                    label=label,
                    confidence=confidence,
                    label_source=label_source,
                    evidence=evidence,
                )
            )

    return ProcessReportDict(
        version=PROCESS_REPORT_VERSION,
        source=source,
        summary=_summary(steps),
        steps=steps,
    )


def _consensus_counts(consistency_traces: list[TraceDict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trace in consistency_traces:
        if not trace.get("success"):
            continue
        hashes_in_trace = {
            hook.get("value_hash")
            for turn in trace.get("turns", [])
            for hook in turn.get("execution", {}).get("hooks", [])
            if hook.get("value_hash")
        }
        for value_hash in hashes_in_trace:
            counts[value_hash] += 1
    return counts


def _hook_evidence(
    *,
    hook: dict[str, Any],
    trace_success: bool,
    final_verified: bool,
    grounded: bool,
    seen_names: set[str],
    seen_keys: Counter[tuple[str, str | None, str | None]],
    consensus: Counter[str],
    consensus_total: int,
) -> ProcessStepEvidenceDict:
    reasons: list[str] = []
    name = hook.get("variable_name")
    depends_on = hook.get("depends_on", [])
    dependency_valid = all(dep in seen_names for dep in depends_on)
    key = ("hook", name, hook.get("value_hash"))
    duplicate = seen_keys[key] > 0

    if not grounded:
        reasons.append("ungrounded_code_line")
    if not dependency_valid:
        reasons.append("invalid_dependency")
    if duplicate:
        reasons.append("duplicate_step")

    return ProcessStepEvidenceDict(
        final_verified=final_verified,
        trace_success=trace_success,
        code_line_grounded=grounded,
        dependency_valid=dependency_valid,
        duplicate=duplicate,
        consensus_matches=consensus.get(hook.get("value_hash", ""), 0),
        consensus_total=consensus_total,
        reasons=reasons,
    )


def _label_hook(
    *,
    source: EpisodeSource,
    evidence: ProcessStepEvidenceDict,
) -> tuple[float | None, Literal["gold", "strong", "weak", "unlabeled"], str]:
    if _is_bad_step(evidence):
        return 0.0, "gold" if source in {"template", "procedural"} else "strong", (
            "deterministic_process" if source in {"template", "procedural"} else "trace_evidence"
        )

    if source in {"template", "procedural"}:
        if evidence["final_verified"] and evidence["trace_success"]:
            return 1.0, "gold", "deterministic_process"
        return None, "unlabeled", "insufficient_evidence"

    if not evidence["final_verified"] or not evidence["trace_success"]:
        return None, "unlabeled", "insufficient_evidence"

    matches = int(evidence.get("consensus_matches", 0))
    total = int(evidence.get("consensus_total", 0))
    ratio = matches / total if total else 0.0
    if matches >= 2 or ratio >= 0.67:
        return 1.0, "strong", "trace_consensus"
    if matches >= 1:
        return 1.0, "weak", "trace_consensus"
    return None, "unlabeled", "insufficient_evidence"


def _label_submit(
    *,
    source: EpisodeSource,
    evidence: ProcessStepEvidenceDict,
) -> tuple[float | None, Literal["gold", "strong", "weak", "unlabeled"], str]:
    if not evidence["final_verified"]:
        return 0.0, "strong", "final_answer"
    if source in {"template", "procedural"}:
        return 1.0, "gold", "deterministic_process"
    if int(evidence.get("consensus_matches", 0)) >= 2:
        return 1.0, "strong", "trace_consensus"
    return 1.0, "weak", "final_answer"


def _is_bad_step(evidence: ProcessStepEvidenceDict) -> bool:
    return (
        not evidence.get("code_line_grounded", False)
        or not evidence.get("dependency_valid", False)
        or evidence.get("duplicate", False)
    )


def _semantic_role(hook: dict[str, Any]) -> str | None:
    text = " ".join(
        str(part or "")
        for part in [
            hook.get("variable_name"),
            hook.get("description"),
            hook.get("code_line"),
        ]
    ).lower()
    if any(token in text for token in ("filter", "where", "rows after")):
        return "filter"
    if any(token in text for token in ("group", "mean", "median", "sum", "count")):
        return "aggregation"
    if any(token in text for token in ("corr", "p_value", "test", "stat")):
        return "statistical_test"
    if any(token in text for token in ("sort", "top", "rank", "min", "max")):
        return "selection"
    return None


def _step_key(step: ProcessStepReportDict) -> tuple[str, str | None, str | None]:
    return (step["step_type"], step.get("variable_name"), step.get("value_hash"))


def _summary(steps: list[ProcessStepReportDict]) -> ProcessReportSummaryDict:
    return ProcessReportSummaryDict(
        total_steps=len(steps),
        labeled_steps=sum(1 for step in steps if step.get("label") is not None),
        gold_steps=sum(1 for step in steps if step.get("confidence") == "gold"),
        strong_steps=sum(1 for step in steps if step.get("confidence") == "strong"),
        weak_steps=sum(1 for step in steps if step.get("confidence") == "weak"),
        unlabeled_steps=sum(
            1 for step in steps if step.get("confidence") == "unlabeled"
        ),
        positive_steps=sum(1 for step in steps if step.get("label") == 1.0),
        negative_steps=sum(1 for step in steps if step.get("label") == 0.0),
    )
