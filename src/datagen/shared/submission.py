"""Submission parsing utilities.

Centralizes parsing of submitted answers from execution stdout.
Enforces strict protocol wrapper for consistency.
"""

import re
import json
from typing import Any


def parse_submission(stdout: str) -> tuple[Any, bool]:
    """Parse submitted answer from execution stdout.

    Looks for "✓ Submitted: {...}" pattern and expects the protocol wrapper
    (for example `{ "__csv_agent_answer__": value, "hooks": [...] }`).

    Args:
        stdout: Raw stdout from code execution.

    Returns:
        (submission_dict, success)
        success=False if no submission found or parse error.
    """
    match = re.search(r"✓ Submitted: (.+)", stdout)
    if not match:
        return None, False

    answer_str = match.group(1).strip()

    try:
        answer = json.loads(answer_str)
    except (json.JSONDecodeError, ValueError):
        return None, False

    return answer, True


def parse_all_submissions(stdout: str) -> list[dict]:
    """Parse all submitted answers from execution stdout.

    Looks for all "✓ Submitted: {...}" lines and returns parsed submission dicts.

    Args:
        stdout: Raw stdout from code execution.

    Returns:
        List of parsed submission dicts (empty if none or parse failures).
    """
    submissions: list[dict] = []
    marker = "✓ Submitted: "
    pos = 0

    while True:
        idx = stdout.find(marker, pos)
        if idx == -1:
            break

        start = idx + len(marker)
        end = stdout.find("\n", start)
        json_str = stdout[start:] if end == -1 else stdout[start:end]

        try:
            submission = json.loads(json_str.strip())
            if isinstance(submission, dict):
                submissions.append(submission)
        except json.JSONDecodeError:
            pass

        pos = start

    return submissions
