"""
Test hook validation logic.

Tests:
- Hook grounding validation (code_line must be substring of executed code)
- Whitespace normalization in matching
- Edge cases (empty hooks, missing code_line)
"""

import pytest
from src.core.environment import validate_hooks_grounded


class TestHookGrounding:
    """Test hook grounding validation."""

    def test_exact_match(self):
        """Hook code_line that exactly matches executed code should be grounded."""
        hooks = [{"code_line": "result = df['col'].mean()"}]
        code_cells = ["result = df['col'].mean()"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 1
        assert len(ungrounded) == 0

    def test_substring_match(self):
        """Hook code_line as substring of larger code block should be grounded."""
        hooks = [{"code_line": "result = df['col'].mean()"}]
        code_cells = [
            "# Calculate mean\nresult = df['col'].mean()\nprint(result)"
        ]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 1
        assert len(ungrounded) == 0

    def test_whitespace_normalization(self):
        """Whitespace differences should not prevent matching."""
        hooks = [{"code_line": "result  =  df['col'].mean()"}]  # Extra spaces
        code_cells = ["result = df['col'].mean()"]  # Normal spacing

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 1
        assert len(ungrounded) == 0

    def test_multiline_normalization(self):
        """Newlines in code_line should match after normalization."""
        hooks = [{"code_line": "result = df.groupby('a')\n    .mean()"}]
        code_cells = ["result = df.groupby('a')     .mean()"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 1
        assert len(ungrounded) == 0

    def test_ungrounded_hook(self):
        """Hook with code_line not in executed code should be ungrounded."""
        hooks = [{"code_line": "result = df['nonexistent'].sum()"}]
        code_cells = ["result = df['col'].mean()"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 0
        assert len(ungrounded) == 1

    def test_missing_code_line(self):
        """Hook without code_line should be ungrounded."""
        hooks = [{"variable_name": "result"}]  # No code_line
        code_cells = ["result = df['col'].mean()"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 0
        assert len(ungrounded) == 1

    def test_empty_code_line(self):
        """Hook with empty code_line should be ungrounded."""
        hooks = [{"code_line": ""}]
        code_cells = ["result = df['col'].mean()"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 0
        assert len(ungrounded) == 1

    def test_multiple_hooks_mixed(self):
        """Should correctly categorize mix of grounded and ungrounded hooks."""
        hooks = [
            {"code_line": "a = df['x'].mean()"},  # grounded
            {"code_line": "b = df['y'].sum()"},   # grounded
            {"code_line": "c = df['z'].max()"},   # NOT grounded
        ]
        code_cells = [
            "a = df['x'].mean()",
            "b = df['y'].sum()",
        ]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 2
        assert len(ungrounded) == 1
        assert ungrounded[0]["code_line"] == "c = df['z'].max()"

    def test_multiple_code_cells(self):
        """Hook should match across multiple code cells."""
        hooks = [{"code_line": "result = final.mean()"}]
        code_cells = [
            "filtered = df[df['a'] > 0]",
            "final = filtered.groupby('b').sum()",
            "result = final.mean()",
        ]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 1
        assert len(ungrounded) == 0

    def test_empty_inputs(self):
        """Should handle empty hooks or code_cells."""
        # Empty hooks
        grounded, ungrounded = validate_hooks_grounded([], ["some code"])
        assert len(grounded) == 0
        assert len(ungrounded) == 0

        # Empty code_cells
        grounded, ungrounded = validate_hooks_grounded(
            [{"code_line": "x = 1"}], []
        )
        assert len(grounded) == 0
        assert len(ungrounded) == 1
