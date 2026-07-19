"""Build diagnostic process reports from episode traces.

Terminal submissions can carry externally verified labels. Hook judgments are
explicitly heuristic: grounding, dependency, duplicate, and consensus evidence
does not prove that a hook made useful computational progress.
"""

from __future__ import annotations

import ast
import math
from collections import Counter
from typing import Any, Literal

from csv_spec import (
    ProcessReportDict,
    ProcessReportSummaryDict,
    ProcessStepEvidenceDict,
    ProcessStepReportDict,
    TraceDict,
    hash_artifact,
    normalize_value,
    validate_hook_event_line,
)
from src.core.environment import validate_hooks_grounded
from src.datagen.shared.submission import parse_submission
from src.datagen.shared.verification import trace_answer_hash

EpisodeSource = Literal["llm_gen", "template", "procedural"]


def build_process_report(
    *,
    source: EpisodeSource,
    gold_trace: TraceDict,
    consistency_traces: list[TraceDict],
    verifier_verdict: bool | None,
) -> ProcessReportDict:
    """Build a step-level process report from gold and consistency traces."""
    validate_trace_submissions(gold_trace, path="gold_trace")
    for trace_index, trace in enumerate(consistency_traces):
        validate_trace_submissions(trace, path=f"consistency_traces[{trace_index}]")
    steps: list[ProcessStepReportDict] = []
    code_cells = [turn.get("code", "") for turn in gold_trace.get("turns", [])]
    consensus = _consensus_counts(consistency_traces)
    consensus_total = sum(1 for trace in consistency_traces if trace.get("success"))
    seen_names: set[str] = set()
    seen_hashes: set[str] = set()
    submitted_turns = [
        turn_index
        for turn_index, turn in enumerate(gold_trace.get("turns", []))
        if turn.get("execution", {}).get("submitted_answer") is not None
    ]
    accepted_submit_turn = (
        submitted_turns[-1]
        if gold_trace.get("success", False) and submitted_turns
        else None
    )

    for turn_index, turn in enumerate(gold_trace.get("turns", [])):
        execution = turn.get("execution", {})
        hooks = execution.get("hooks", [])
        for hook_index, hook in enumerate(hooks):
            grounded, event_provenance_reason = _hook_grounding(
                hook=hook,
                earlier_code_cells=code_cells[:turn_index],
                current_code=code_cells[turn_index],
            )
            evidence = _hook_evidence(
                hook=hook,
                trace_success=gold_trace.get("success", False),
                final_verified=verifier_verdict,
                grounded=grounded,
                event_provenance_reason=event_provenance_reason,
                seen_names=seen_names,
                seen_hashes=seen_hashes,
                consensus=consensus,
                consensus_total=consensus_total,
            )
            label, label_kind, label_source = _label_hook(
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
                label_kind=label_kind,
                label_source=label_source,
                evidence=evidence,
            )
            steps.append(step)

            if event_provenance_reason is None:
                name = hook.get("variable_name")
                if isinstance(name, str) and name:
                    seen_names.add(name)
                value_hash = hook.get("value_hash")
                if isinstance(value_hash, str) and value_hash:
                    seen_hashes.add(value_hash)

        submitted = execution.get("submitted_answer")
        if submitted is not None:
            accepted = turn_index == accepted_submit_turn
            answer_hash = hash_artifact(submitted)
            evidence = ProcessStepEvidenceDict(
                final_verified=verifier_verdict if accepted else None,
                trace_success=gold_trace.get("success", False),
                code_line_grounded=True,
                dependency_valid=True,
                duplicate=False,
                consensus_matches=_submission_consensus_matches(
                    consistency_traces, answer_hash
                ),
                consensus_total=consensus_total,
                reasons=[] if accepted else ["submission_not_accepted"],
            )
            if not accepted:
                label, label_kind, label_source = (
                    None,
                    "unlabeled",
                    "rejected_submission",
                )
            else:
                label, label_kind, label_source = _label_submit(evidence=evidence)
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
                    label_kind=label_kind,
                    label_source=label_source,
                    evidence=evidence,
                )
            )

    return ProcessReportDict(
        summary=_summary(steps),
        steps=steps,
    )


def _consensus_counts(consistency_traces: list[TraceDict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trace in consistency_traces:
        if not trace.get("success"):
            continue
        hashes_in_trace: set[str] = set()
        code_cells = [turn.get("code", "") for turn in trace.get("turns", [])]
        for turn_index, turn in enumerate(trace.get("turns", [])):
            for hook in turn.get("execution", {}).get("hooks", []):
                _grounded, provenance_reason = _hook_grounding(
                    hook=hook,
                    earlier_code_cells=code_cells[:turn_index],
                    current_code=code_cells[turn_index],
                )
                value_hash = hook.get("value_hash")
                if (
                    provenance_reason is None
                    and isinstance(value_hash, str)
                    and value_hash
                ):
                    hashes_in_trace.add(value_hash)
        for value_hash in hashes_in_trace:
            counts[value_hash] += 1
    return counts


def _submission_consensus_matches(
    consistency_traces: list[TraceDict], answer_hash: str
) -> int:
    return sum(
        1
        for trace in consistency_traces
        if trace.get("success") and trace_answer_hash(trace) == answer_hash
    )


def validate_trace_submissions(trace: TraceDict, *, path: str = "trace") -> None:
    turns = trace.get("turns", [])
    submissions: list[tuple[int, Any, bool]] = []
    for turn_index, turn in enumerate(turns):
        execution = turn.get("execution", {})
        submitted_answer = execution.get("submitted_answer")
        stdout = execution.get("stdout", "")
        if not isinstance(stdout, str):
            raise ValueError(f"{path}.turns[{turn_index}].execution.stdout is invalid")
        stdout_lines = stdout.splitlines()
        submission_line_indexes = [
            line_index
            for line_index, line in enumerate(stdout_lines)
            if "✓ Submitted:" in line
        ]
        turn_path = f"{path}.turns[{turn_index}]"
        if len(submission_line_indexes) > 1:
            raise ValueError(f"{turn_path} contains multiple submissions")
        if submitted_answer is not None and not submission_line_indexes:
            raise ValueError(f"{turn_path} submission record is missing")
        if submission_line_indexes and submitted_answer is None:
            raise ValueError(f"{turn_path} submission was not captured")
        if submission_line_indexes:
            submission_line_index = submission_line_indexes[0]
            submission_record, parsed = parse_submission(
                stdout_lines[submission_line_index]
            )
            if (
                not parsed
                or not isinstance(submission_record, dict)
                or "__csv_agent_answer__" not in submission_record
            ):
                raise ValueError(f"{turn_path} submission record is invalid")
            if _exact_value_identity(
                submission_record["__csv_agent_answer__"]
            ) != _exact_value_identity(submitted_answer):
                raise ValueError(
                    f"{turn_path} submission record does not match captured answer"
                )
            if any(
                "📍 Hook:" in line for line in stdout_lines[submission_line_index + 1 :]
            ):
                raise ValueError(f"{turn_path} contains a hook after submission")
        if submitted_answer is None:
            continue
        submissions.append(
            (turn_index, submitted_answer, execution.get("success") is True)
        )

        code = turn.get("code", "")
        if not isinstance(code, str):
            raise ValueError(f"{turn_path}.code is invalid")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"{turn_path} code is not parseable") from exc

        parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        submit_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "submit"
        ]
        if len(submit_calls) != 1:
            raise ValueError(
                f"{turn_path} must contain exactly one submitted operation"
            )
        submit_call = submit_calls[0]
        submit_statement = parents.get(id(submit_call))
        if (
            not isinstance(submit_statement, ast.Expr)
            or submit_statement.value is not submit_call
            or not tree.body
            or tree.body[-1] is not submit_statement
        ):
            raise ValueError(f"{turn_path} submission must be the terminal operation")

    canonical_answer_hash = trace_answer_hash(trace)
    if trace.get("success"):
        accepted_turn, accepted_submission, accepted_execution_succeeded = (
            submissions[-1] if submissions else (None, None, False)
        )
        if (
            not submissions
            or canonical_answer_hash is None
            or hash_artifact(trace.get("final_answer")) != canonical_answer_hash
            or _exact_value_identity(accepted_submission)
            != _exact_value_identity(trace.get("final_answer"))
        ):
            raise ValueError(f"{path} final answer is not the accepted submission")
        if accepted_turn != len(turns) - 1:
            raise ValueError(f"{path} accepted submission is not in the final turn")
        if not accepted_execution_succeeded:
            raise ValueError(f"{path} accepted submission execution failed")
    elif trace.get("final_answer") is not None:
        raise ValueError(f"unsuccessful {path} has a final answer")


def _hook_grounding(
    *, hook: dict[str, Any], earlier_code_cells: list[str], current_code: str
) -> tuple[bool, str | None]:
    """Check grounding only against code that preceded the runtime hook event."""
    if "event_provenance_reason" not in hook:
        return False, "missing_event_provenance_state"
    provenance_reason = hook["event_provenance_reason"]
    if provenance_reason is not None and (
        not isinstance(provenance_reason, str) or not provenance_reason
    ):
        return False, "invalid_hook_record_provenance"
    if provenance_reason is not None:
        return False, provenance_reason
    if (
        not isinstance(hook.get("code_line"), str)
        or (
            hook.get("variable_name") is not None
            and not isinstance(hook.get("variable_name"), str)
        )
        or not isinstance(hook.get("value_hash"), str)
        or not isinstance(hook.get("depends_on"), list)
        or not all(isinstance(dep, str) for dep in hook.get("depends_on", []))
        or (
            hook.get("description") is not None
            and not isinstance(hook.get("description"), str)
        )
    ):
        return False, "invalid_hook_record_provenance"
    event_line = hook.get("event_line")
    current_lines = current_code.splitlines()
    if event_line is None:
        return False, "missing_or_ambiguous_event_provenance"
    if type(event_line) is not int or event_line < 1 or event_line > len(current_lines):
        return False, "invalid_event_provenance"
    if validate_hook_event_line(current_code, event_line) is None:
        return False, "missing_or_ambiguous_event_provenance"
    executed_prefix = "\n".join(current_lines[: event_line - 1])
    hook_copy = dict(hook)
    grounded, _ungrounded = validate_hooks_grounded(
        [hook_copy], [*earlier_code_cells, executed_prefix]
    )
    return bool(grounded), None


def _exact_value_identity(value: Any) -> tuple:
    """Return type-preserving identity for accepted-submission provenance."""
    value = normalize_value(value)
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if math.isnan(value):
            return ("float", "nan")
        return ("float", value.hex())
    if type(value) is str:
        return ("str", value)
    if type(value) is list:
        return ("list", tuple(_exact_value_identity(item) for item in value))
    if type(value) is tuple:
        return ("tuple", tuple(_exact_value_identity(item) for item in value))
    if type(value) is dict:
        items = [
            (_exact_value_identity(key), _exact_value_identity(item))
            for key, item in value.items()
        ]
        return ("dict", tuple(sorted(items, key=repr)))
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _hook_evidence(
    *,
    hook: dict[str, Any],
    trace_success: bool,
    final_verified: bool | None,
    grounded: bool,
    event_provenance_reason: str | None,
    seen_names: set[str],
    seen_hashes: set[str],
    consensus: Counter[str],
    consensus_total: int,
) -> ProcessStepEvidenceDict:
    reasons: list[str] = []
    depends_on = hook.get("depends_on", [])
    if not isinstance(depends_on, list) or not all(
        isinstance(dep, str) for dep in depends_on
    ):
        depends_on = []
    dependency_valid = all(dep in seen_names for dep in depends_on)
    value_hash = hook.get("value_hash")
    if not isinstance(value_hash, str):
        value_hash = ""
    duplicate = bool(value_hash and value_hash in seen_hashes)

    if event_provenance_reason is not None:
        reasons.append(event_provenance_reason)
    elif not grounded:
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
        consensus_matches=consensus.get(value_hash, 0),
        consensus_total=consensus_total,
        reasons=reasons,
    )


def _label_hook(
    *,
    source: EpisodeSource,
    evidence: ProcessStepEvidenceDict,
) -> tuple[float | None, Literal["heuristic", "unlabeled"], str]:
    """Assign an optional hook heuristic without claiming process truth."""
    if any("provenance" in reason for reason in evidence["reasons"]):
        return None, "unlabeled", "event_provenance_unavailable"
    if _is_bad_step(evidence):
        return 0.0, "heuristic", "structural_hook_heuristic"

    if evidence.get("final_verified") is not True or not evidence["trace_success"]:
        return None, "unlabeled", "insufficient_evidence"

    if source in {"template", "procedural"}:
        return 1.0, "heuristic", "successful_trace_heuristic"

    if int(evidence.get("consensus_matches", 0)) >= 1:
        return 1.0, "heuristic", "trace_consensus_heuristic"
    return None, "unlabeled", "insufficient_evidence"


def _label_submit(
    *,
    evidence: ProcessStepEvidenceDict,
) -> tuple[float | None, Literal["verified", "unlabeled"], str]:
    """Label only the terminal answer from the task's external verifier."""
    verdict = evidence.get("final_verified")
    if verdict is None:
        return None, "unlabeled", "verifier_unavailable"
    return (1.0 if verdict else 0.0), "verified", "terminal_verifier"


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


def _summary(steps: list[ProcessStepReportDict]) -> ProcessReportSummaryDict:
    return ProcessReportSummaryDict(
        total_steps=len(steps),
        labeled_steps=sum(1 for step in steps if step.get("label") is not None),
        verified_steps=sum(1 for step in steps if step.get("label_kind") == "verified"),
        heuristic_steps=sum(
            1 for step in steps if step.get("label_kind") == "heuristic"
        ),
        unlabeled_steps=sum(
            1 for step in steps if step.get("label_kind") == "unlabeled"
        ),
        positive_steps=sum(1 for step in steps if step.get("label") == 1.0),
        negative_steps=sum(1 for step in steps if step.get("label") == 0.0),
    )
