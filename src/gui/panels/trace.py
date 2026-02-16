"""
Trace viewer panel - view execution traces, code, outputs.
"""

import json

import dearpygui.dearpygui as dpg


def create_trace_panel(parent: int | str):
    """Create the trace viewer panel."""
    with dpg.child_window(parent=parent, width=-1, height=-1, border=True):
        dpg.add_text("Trace Viewer", color=(150, 200, 255))
        dpg.add_separator()

        # Question header
        with dpg.group(tag="trace_question_group"):
            dpg.add_text("No trace selected", tag="trace_question", wrap=500)
            dpg.add_text("", tag="trace_hint", color=(150, 150, 150), wrap=500)
            dpg.add_text("", tag="trace_metadata", color=(100, 150, 200))

        dpg.add_separator()

        # Trace content
        with dpg.child_window(tag="trace_content", height=-1, border=False):
            dpg.add_text(
                "Select a question or episode from the explorer", color=(150, 150, 150)
            )


def show_question(question: dict):
    """Display a question in the trace viewer."""
    # Update header
    q_text = question.get("question_text") or question.get("question_mechanical") or ""
    hint = question.get("hint", "")
    difficulty = question.get("difficulty", "?")
    template = question.get("template_name", "")
    n_steps = question.get("n_steps", "?")

    dpg.set_value("trace_question", q_text)
    dpg.set_value("trace_hint", f"Hint: {hint}" if hint else "")
    dpg.set_value(
        "trace_metadata",
        f"Difficulty: {difficulty} | Steps: {n_steps} | Template: {template}",
    )

    # Clear content
    dpg.delete_item("trace_content", children_only=True)

    # Show ground truth if available
    gt = question.get("ground_truth")
    if gt is not None:
        with dpg.group(parent="trace_content"):
            dpg.add_text("Ground Truth:", color=(100, 255, 100))
            gt_text = (
                json.dumps(gt, indent=2, default=str)
                if isinstance(gt, (dict, list))
                else str(gt)
            )
            dpg.add_input_text(
                default_value=gt_text,
                multiline=True,
                readonly=True,
                width=-1,
                height=100,
            )

    # Show output schema
    schema = question.get("output_schema", "")
    if schema:
        with dpg.group(parent="trace_content"):
            dpg.add_text("Expected Output:", color=(150, 200, 255))
            dpg.add_text(schema, wrap=500)


def show_trace(episode: dict):
    """Display an episode trace in the viewer."""
    # Update header
    q = episode.get("question", {})
    q_text = q.get("question_text", "")
    hint = q.get("hint", "")
    difficulty = q.get("difficulty", "?")
    template = q.get("template_name", "")
    n_steps = q.get("n_steps", "?")
    verified = episode.get("verified", False)

    dpg.set_value("trace_question", q_text)
    dpg.set_value("trace_hint", f"Hint: {hint}" if hint else "")

    status = "[VERIFIED]" if verified else "[FAILED]"
    status_color = (100, 255, 100) if verified else (255, 100, 100)
    meta_text = f"{status} | Difficulty: {difficulty} | Steps: {n_steps}"
    if template:
        meta_text += f" | Template: {template}"
    dpg.set_value("trace_metadata", meta_text)
    dpg.configure_item("trace_metadata", color=status_color)

    # Clear and rebuild content
    dpg.delete_item("trace_content", children_only=True)

    # Gold trace
    gold_trace = episode.get("gold_trace", {})
    if gold_trace:
        _render_trace(gold_trace, "Gold Trace (with hint)", (100, 255, 100))

    # Consistency traces
    consistency_traces = episode.get("consistency_traces", [])
    for i, trace in enumerate(consistency_traces):
        _render_trace(trace, f"Consistency Trace {i + 1} (no hint)", (255, 200, 100))

    # Triangulation info
    tri = episode.get("triangulation", {})
    if tri:
        with dpg.group(parent="trace_content"):
            dpg.add_separator()
            dpg.add_text("Triangulation:", color=(150, 200, 255))
            dpg.add_text(f"  Consistency runs: {tri.get('n_consistency_runs', 0)}")
            dpg.add_text(f"  Succeeded: {tri.get('n_consistency_succeeded', 0)}")
            dpg.add_text(f"  Majority count: {tri.get('majority_count', 0)}")
            matches = tri.get("gold_matches_majority", False)
            match_color = (100, 255, 100) if matches else (255, 100, 100)
            dpg.add_text(f"  Gold matches majority: {matches}", color=match_color)

    # Timing info
    timing = episode.get("timing", {})
    if timing:
        with dpg.group(parent="trace_content"):
            dpg.add_separator()
            dpg.add_text("Timing:", color=(150, 200, 255))
            dpg.add_text(f"  Gold elapsed: {timing.get('gold_elapsed', 0):.1f}s")
            dpg.add_text(f"  Total elapsed: {timing.get('total_elapsed', 0):.1f}s")
            dpg.add_text(f"  Average: {timing.get('avg_elapsed', 0):.1f}s")


def _render_trace(trace: dict, title: str, color: tuple):
    """Render a single trace with its turns."""
    with dpg.tree_node(label=title, parent="trace_content", default_open=True):
        dpg.configure_item(dpg.last_item(), label_color=color)

        # Final answer
        answer = trace.get("final_answer")
        if answer is not None:
            answer_text = (
                json.dumps(answer, default=str)
                if isinstance(answer, (dict, list))
                else str(answer)
            )
            dpg.add_text(f"Answer: {answer_text[:100]}", color=color)

        success = trace.get("success", False)
        dpg.add_text(f"Success: {success}", color=color if success else (255, 100, 100))

        # Turns
        turns = trace.get("turns", [])
        for turn in turns:
            _render_turn(turn)


def _render_turn(turn: dict):
    """Render a single turn."""
    turn_idx = turn.get("turn_index", 0)

    with dpg.tree_node(label=f"Turn {turn_idx}", default_open=False):
        # Reasoning
        reasoning = turn.get("reasoning", "")
        if reasoning:
            dpg.add_text("Reasoning:", color=(150, 200, 255))
            dpg.add_input_text(
                default_value=reasoning,
                multiline=True,
                readonly=True,
                width=-1,
                height=60,
            )

        # Code
        code = turn.get("code", "")
        if code:
            dpg.add_text("Code:", color=(255, 200, 100))
            dpg.add_input_text(
                default_value=code,
                multiline=True,
                readonly=True,
                width=-1,
                height=100,
            )

        # Execution result
        execution = turn.get("execution", {})
        if execution:
            success = execution.get("success", False)
            status_color = (100, 255, 100) if success else (255, 100, 100)
            dpg.add_text(
                f"Execution: {'Success' if success else 'Failed'}", color=status_color
            )

            stdout = execution.get("stdout", "")
            if stdout:
                dpg.add_text("stdout:", color=(150, 150, 150))
                dpg.add_input_text(
                    default_value=stdout[:500],
                    multiline=True,
                    readonly=True,
                    width=-1,
                    height=60,
                )

            stderr = execution.get("stderr", "")
            if stderr:
                dpg.add_text("stderr:", color=(255, 100, 100))
                dpg.add_input_text(
                    default_value=stderr[:500],
                    multiline=True,
                    readonly=True,
                    width=-1,
                    height=60,
                )

            # Hooks
            hooks = execution.get("hooks", [])
            if hooks:
                with dpg.tree_node(label=f"Hooks ({len(hooks)})"):
                    for hook in hooks:
                        name = hook.get("variable_name", "?")
                        value = hook.get("value")
                        value_str = (
                            json.dumps(value, default=str)[:100] if value else "None"
                        )
                        dpg.add_text(f"{name}: {value_str}")
