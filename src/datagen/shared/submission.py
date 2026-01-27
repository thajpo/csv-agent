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
