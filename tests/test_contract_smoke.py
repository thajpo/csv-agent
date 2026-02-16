"""
Smoke tests for csv-spec contract between trainer and environment.

Tests the action→step→result flow using csv_spec types.
"""

from csv_spec import (
    parse_action,
    parse_step_result,
    CodeAction,
    StepResult,
)


class TestParseAction:
    """Test action parsing from model output."""

    def test_parse_python_code_block(self):
        """Extract code from markdown code block."""
        model_output = """
I'll compute the mean of the age column.

```python
mean_age = df['age'].mean()
print(mean_age)
```

This will give us the average age.
"""
        action = parse_action(model_output)

        assert action is not None
        assert isinstance(action, CodeAction)
        assert "mean_age = df['age'].mean()" in action.code

    def test_parse_no_code_block(self):
        """Return None when no code block found."""
        model_output = "I'm thinking about how to solve this problem..."
        action = parse_action(model_output)

        assert action is None

    def test_parse_multiple_code_blocks_takes_first(self):
        """When multiple code blocks, take the first one."""
        model_output = """
First I'll try this:

```python
x = 1
```

Actually, let me do this instead:

```python
x = 2
```
"""
        action = parse_action(model_output)

        assert action is not None
        assert "x = 1" in action.code


class TestParseStepResult:
    """Test step result parsing from execution output."""

    def test_parse_successful_execution(self):
        """Parse successful code execution."""
        output = """Loaded CSV: 100 rows, 5 columns
54.3
Out[1]: 54.3"""

        result = parse_step_result(output)

        assert isinstance(result, StepResult)
        assert result.success is True
        assert result.terminal is False
        assert result.submitted_answer is None
        assert "54.3" in result.stdout

    def test_parse_execution_with_error(self):
        """Parse execution that failed with error."""
        output = """Traceback (most recent call last):
  File "<cell>", line 1, in <module>
KeyError: 'nonexistent_column'"""

        result = parse_step_result(output)

        assert result.success is False
        assert result.terminal is False

    def test_parse_submission(self):
        """Parse output with submit() call."""
        output = """Computing result...
✓ Submitted: {"__csv_agent_answer__": 42.5}"""

        result = parse_step_result(output)

        assert result.success is True
        assert result.terminal is True
        assert result.terminal_reason == "submit"
        assert result.submitted_answer == 42.5

    def test_parse_hooks(self):
        """Parse output with hook() calls."""
        output = """📍 Hook: {"__csv_agent_hook__": true, "variable_name": "mean_age", "value": 54.3, "value_hash": "abc123", "code_line": "mean_age = df['age'].mean()"}
📍 Hook: {"__csv_agent_hook__": true, "variable_name": "filtered", "value": {"type": "DataFrame", "shape": [10, 3]}, "value_hash": "def456", "code_line": "filtered = df[df['age'] > 50]"}
✓ Submitted: {"__csv_agent_answer__": 54.3, "hooks": []}"""

        result = parse_step_result(output)

        assert result.success is True
        assert result.terminal is True
        assert len(result.hooks) == 2
        assert result.hooks[0]["variable_name"] == "mean_age"
        assert result.hooks[1]["variable_name"] == "filtered"

    def test_parse_hooks_in_submission(self):
        """Parse hooks embedded in submission."""
        output = """✓ Submitted: {"__csv_agent_answer__": 42, "hooks": [{"__csv_agent_hook__": true, "variable_name": "result", "value": 42, "value_hash": "xyz", "code_line": "result = 42"}]}"""

        result = parse_step_result(output)

        assert result.terminal is True
        assert len(result.hooks) == 1
        assert result.hooks[0]["variable_name"] == "result"


class TestContractFlow:
    """Test the full action→step flow."""

    def test_action_to_step_flow(self):
        """Verify the contract flow from action to step."""
        # 1. Model outputs code
        model_output = """
I'll compute the answer.

```python
result = 42
submit(result)
```
"""

        # 2. Parse action
        action = parse_action(model_output)
        assert isinstance(action, CodeAction)
        assert "result = 42" in action.code

        # 3. Simulate execution output (from environment)
        execution_output = """result = 42
✓ Submitted: {"__csv_agent_answer__": 42, "hooks": []}"""

        # 4. Parse step result
        result = parse_step_result(execution_output)
        assert result.terminal is True
        assert result.submitted_answer == 42

        # 5. The contract is maintained: trainer got structured result
        #    from environment, ready for rubric scoring

    def test_multiline_code_preserves_formatting(self):
        """Ensure multiline code is properly extracted."""
        model_output = """
```python
def compute_mean():
    mean_val = df['age'].mean()
    return mean_val

result = compute_mean()
submit(result)
```
"""
        action = parse_action(model_output)

        assert "def compute_mean():" in action.code
        assert "    mean_val = df['age'].mean()" in action.code
        assert "result = compute_mean()" in action.code
