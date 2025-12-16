"""Validation utilities for conversation structure."""

import re


def get_turn_validation_feedback(response: str, code_cells: list[str], min_reasoning_length: int = 10) -> str:
    """
    Get validation feedback for turn structure.
    Returns empty string if valid, otherwise returns error message.
    """
    # Check for exactly one code block
    if len(code_cells) == 0:
        return "❌ No code block found. You must write exactly ONE ```python code block."

    if len(code_cells) > 1:
        return f"❌ Found {len(code_cells)} code blocks. Write exactly ONE ```python code block per turn."

    # Check for reasoning text before code
    # Extract text before first code block
    code_block_pattern = r"```python\n.*?```"
    parts = re.split(code_block_pattern, response, maxsplit=1, flags=re.DOTALL)

    reasoning_text = parts[0].strip() if parts else ""

    # Require at least some minimal reasoning (not just whitespace)
    if not reasoning_text or len(reasoning_text) < min_reasoning_length:
        return "❌ Write your reasoning first: explain what you'll do and why (1-3 sentences), then write the code block."
    
    return ""
