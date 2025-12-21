"""
Code parsing utilities.

Shared utilities for extracting code blocks from LLM responses.
"""

import re
import textwrap


def extract_python_cells(response: str) -> list[str]:
    """
    Extract python code blocks from LLM response.
    
    Handles:
    - ```python and ```py fences
    - Missing closing backticks (malformed fences)
    - Trailing backticks in code content
    
    Args:
        response: Raw LLM response text
        
    Returns:
        List of extracted Python code strings, cleaned and dedented
    """
    # Match ```python or ```py blocks, tolerating missing closing ```
    pattern = r"```(?:python|py)\n([\s\S]*?)(?:```|$)"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    cleaned_blocks = []
    for block in matches:
        # Dedent, normalize line endings, strip
        cleaned = textwrap.dedent(block).replace("\r\n", "\n").strip()
        # Remove trailing backticks (from malformed fences)
        cleaned = cleaned.rstrip("`").strip()
        if cleaned:
            cleaned_blocks.append(cleaned)
    
    return cleaned_blocks
