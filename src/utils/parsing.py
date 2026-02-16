"""
Interaction utilities: Parsing, Execution, and Validation.

Combines functionality for:
1. Extracting code from LLM responses (parsing)
2. Parsing execution results from the environment (execution)
3. Validating turn structure (validation)
"""

import re
import textwrap

# Use CodeCellResult from conversation.py instead of ExecutionResult
from src.core.conversation import CodeCellResult


# =============================================================================
# 1. PARSING (formerly src/utils/parsing.py)
# =============================================================================

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


# =============================================================================
# 2. EXECUTION (formerly src/utils/execution.py)
# =============================================================================

def parse_execution_result(output: str) -> CodeCellResult:
    """
    Parse verifiers PythonEnv output string into CodeCellResult.
    
    The verifiers format returns:
    - stdout lines
    - "stderr:\n..." if there's stderr
    - Traceback if there's an error
    - "Out[N]: ..." for results
    - "(no output)" if empty
    
    Returns:
        CodeCellResult(success, stdout, stderr, code="", submitted_answer=None)
        Note: code and submitted_answer must be filled by caller if needed for context.
    """
    # Default values
    code = ""  # Not available from output string alone
    
    if not output or output == "(no output)":
        return CodeCellResult(success=True, stdout="", stderr="", code=code)
    
    # Check for error indicators (Python traceback patterns)
    # More specific patterns to avoid false positives like "Standard Error: 0.5"
    error_indicators = [
        "Traceback (most recent call last):",
        "\nError: ",       # Newline + Error: space (common in tracebacks)
        "Error:\n",        # Error: followed by newline
        "\nException: ",   # Exception with context
        "NameError:",
        "TypeError:",
        "ValueError:",
        "KeyError:",
        "IndexError:",
        "AttributeError:",
        "ImportError:",
        "SyntaxError:",
        "ZeroDivisionError:",
    ]

    is_error = any(indicator in output for indicator in error_indicators)
    
    if is_error:
        return CodeCellResult(success=False, stdout="", stderr=output, code=code)
    
    # Check for stderr section
    if "stderr:\n" in output:
        parts = output.split("stderr:\n", 1)
        output_stdout = parts[0].strip()
        output_stderr = parts[1].strip() if len(parts) > 1 else ""
        # stderr doesn't always mean failure, but in this env it often does or is mixed
        return CodeCellResult(success=True, stdout=output_stdout, stderr=output_stderr, code=code)
    
    # Normal output
    return CodeCellResult(success=True, stdout=output, stderr="", code=code)


# =============================================================================
# 3. VALIDATION (formerly src/utils/validation.py)
# =============================================================================

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
