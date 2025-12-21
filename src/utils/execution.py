"""
Execution result parsing utilities.

Shared utilities for parsing Python execution output from the sandbox environment.
"""

from src.core.types import ExecutionResult


def parse_execution_result(output: str) -> ExecutionResult:
    """
    Parse verifiers PythonEnv output string into ExecutionResult.
    
    The verifiers format returns:
    - stdout lines
    - "stderr:\n..." if there's stderr
    - Traceback if there's an error
    - "Out[N]: ..." for results
    - "(no output)" if empty
    
    Returns:
        ExecutionResult(success, stdout, stderr)
    """
    if not output or output == "(no output)":
        return ExecutionResult(True, "", "")
    
    # Check for error indicators (Python traceback)
    error_indicators = [
        "Traceback (most recent call last):",
        "Error:",
        "Exception:",
    ]
    
    is_error = any(indicator in output for indicator in error_indicators)
    
    if is_error:
        return ExecutionResult(False, "", output)
    
    # Check for stderr section
    if "stderr:\n" in output:
        parts = output.split("stderr:\n", 1)
        stdout = parts[0].strip()
        stderr = parts[1].strip() if len(parts) > 1 else ""
        # stderr doesn't always mean failure
        return ExecutionResult(True, stdout, stderr)
    
    # Normal output
    return ExecutionResult(True, output, "")
