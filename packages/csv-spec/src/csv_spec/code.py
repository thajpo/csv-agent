"""Shared extraction for executable Python cells in model responses."""

import re
import textwrap


def extract_python_cells(response: str) -> list[str]:
    """Extract cleaned Python code blocks from a model response."""
    pattern = r"```(?:python|py)\n([\s\S]*?)(?:```|$)"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    cleaned_blocks = []
    for block in matches:
        cleaned = textwrap.dedent(block).replace("\r\n", "\n").strip()
        cleaned = cleaned.rstrip("`").strip()
        if cleaned:
            cleaned_blocks.append(cleaned)
    return cleaned_blocks
