"""
Action and result parsing - the contract boundary between trainer and environment.

This module defines HOW actions are extracted from model output and HOW
execution results are parsed into structured format.

IMPORTANT: This is a CONTRACT file. If you change these functions, you MUST update both:
1. Environment (csv_env.py) - how it parses model output
2. Trainer (rl_env.py, prompts) - how it formats actions and consumes results
"""

import ast
import json
import re
from typing import Any

from csv_spec.types import (
    ActionSpec,
    CodeAction,
    StepResult,
    HookDict,
)


def parse_hook_record(hook_data: Any) -> HookDict | None:
    """Parse one hook protocol record without trusting its source location."""
    if (
        not isinstance(hook_data, dict)
        or hook_data.get("__csv_agent_hook__") is not True
    ):
        return None

    event_line = hook_data.get("event_line")
    if event_line is not None and (type(event_line) is not int or event_line < 1):
        event_line = None

    return HookDict(
        variable_name=hook_data.get("variable_name"),
        code_line=hook_data.get("code_line", ""),
        value=hook_data.get("value"),
        value_hash=hook_data.get("value_hash", ""),
        depends_on=hook_data.get("depends_on", []),
        description=hook_data.get("description"),
        event_line=event_line,
    )


def validate_hook_event_line(code: str, event_line: Any) -> int | None:
    """Retain an event line only when it identifies one direct hook() call."""
    if type(event_line) is not int or event_line < 1:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    matching_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hook"
        and node.lineno == event_line
    ]
    return event_line if len(matching_calls) == 1 else None


def parse_action(model_output: str) -> ActionSpec | None:
    """
    Extract action from model's text output.

    The model is expected to output Python code in markdown code blocks:
    ```python
    <code here>
    ```

    Args:
        model_output: Raw text from model completion

    Returns:
        CodeAction if code block found
        None if no valid action found

    Note:
        SubmitAction is not returned here - it's detected from execution output
        via parse_step_result() when submit() is called.
    """
    # Extract ```python ... ``` blocks
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)

    if matches:
        # Take the first code block (model should only write one per turn)
        return CodeAction(code=matches[0])

    return None


def parse_step_result(
    execution_output: str,
    stderr: str = "",
) -> StepResult:
    """
    Parse environment execution output into structured StepResult.

    Handles:
    - Hook extraction (📍 Hook: {...})
    - Submit extraction (✓ Submitted: {...})
    - Error detection (Traceback, *Error:)
    - Protocol validation

    Args:
        execution_output: stdout from code execution
        stderr: stderr from code execution (optional)

    Returns:
        StepResult with parsed hooks, submission, and terminal status

    Protocol:
        - Hooks are logged as: 📍 Hook: {"__csv_agent_hook__": true, ...}
        - Submission is logged as: ✓ Submitted: {"__csv_agent_answer__": value, ...}
        - Both must include their marker keys to be valid
    """
    hooks: list[HookDict] = []
    submitted_answer: Any = None
    success = True
    terminal = False
    terminal_reason = None

    # Combine stdout and stderr for full context
    full_output = execution_output
    if stderr:
        full_output = f"{execution_output}\n{stderr}"

    # Parse hooks (📍 Hook: {...})
    for line in execution_output.split("\n"):
        if "📍 Hook:" in line:
            json_start = line.find("{")
            if json_start == -1:
                continue
            try:
                hook = parse_hook_record(json.loads(line[json_start:]))
                if hook is not None:
                    hooks.append(hook)
            except json.JSONDecodeError:
                # Malformed hook - skip but don't fail
                pass

    # Parse submission (✓ Submitted: {...})
    submit_match = re.search(r"✓ Submitted: (.+)", execution_output)
    if submit_match:
        try:
            data = json.loads(submit_match.group(1))
            if "__csv_agent_answer__" in data:
                submitted_answer = data["__csv_agent_answer__"]
                terminal = True
                terminal_reason = "submit"
                # Also capture hooks from submission if present
                if "hooks" in data and isinstance(data["hooks"], list):
                    for hook_data in data["hooks"]:
                        hook = parse_hook_record(hook_data)
                        if hook is not None and hook not in hooks:
                            hooks.append(hook)
        except json.JSONDecodeError:
            # Malformed submission - treat as error
            success = False

    # Detect errors
    error_patterns = [
        "Traceback (most recent call last):",
        "Error:",
        "Exception:",
    ]
    if any(pattern in full_output for pattern in error_patterns):
        success = False

    return StepResult(
        success=success,
        stdout=execution_output,
        stderr=stderr,
        hooks=hooks,
        submitted_answer=submitted_answer,
        terminal=terminal,
        terminal_reason=terminal_reason,
    )


def extract_python_cells(response: str) -> list[str]:
    """
    Extract all Python code blocks from a response.

    Args:
        response: Model response text

    Returns:
        List of code strings (without the ```python markers)
    """
    pattern = r"```python\n(.*?)```"
    return re.findall(pattern, response, re.DOTALL)
