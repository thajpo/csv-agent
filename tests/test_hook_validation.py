"""
Test hook validation logic.

Tests:
- Hook grounding validation (code_line must exactly match a line in executed code)
- Whitespace normalization in matching
- Edge cases (empty hooks, missing code_line)
- False positive rejection (substring matching removed)
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

    def test_substring_no_longer_matches(self):
        """Substring matching removed - must match full line exactly."""
        # Hook is just "df['col']" which is a substring of the full line
        hooks = [{"code_line": "df['col']"}]
        code_cells = ["# Calculate mean\nresult = df['col'].mean()\nprint(result)"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        # Should be ungrounded because "df['col']" is only a substring, not a full line
        assert len(grounded) == 0
        assert len(ungrounded) == 1
        assert (
            ungrounded[0]["_ungrounded_reason"]
            == "code_line not found in executed code"
        )

    def test_whitespace_normalization(self):
        """Whitespace differences should not prevent matching."""
        hooks = [{"code_line": "result  =  df['col'].mean()"}]  # Extra spaces
        code_cells = ["result = df['col'].mean()"]  # Normal spacing

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 1
        assert len(ungrounded) == 0

    def test_multiline_normalization(self):
        """Multiline code_line must have all lines match individually after normalization."""
        # Hook spans two logical lines
        hooks = [{"code_line": "result = df.groupby('a')\n    .mean()"}]
        # Code cell has same two lines
        code_cells = ["result = df.groupby('a')\n    .mean()"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        # Both normalized lines should match individually
        assert len(grounded) == 1
        assert len(ungrounded) == 0

    def test_multiline_partial_match_fails(self):
        """Multiline hook where only some lines match should be ungrounded."""
        hooks = [{"code_line": "result = df.groupby('a')\n    .mean()"}]
        # Only first line matches
        code_cells = ["result = df.groupby('a')"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 0
        assert len(ungrounded) == 1

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
        assert ungrounded[0]["_ungrounded_reason"] == "missing code_line"

    def test_empty_code_line(self):
        """Hook with empty code_line should be ungrounded."""
        hooks = [{"code_line": ""}]
        code_cells = ["result = df['col'].mean()"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 0
        assert len(ungrounded) == 1
        assert ungrounded[0]["_ungrounded_reason"] == "missing code_line"

    def test_multiple_hooks_mixed(self):
        """Should correctly categorize mix of grounded and ungrounded hooks."""
        hooks = [
            {"code_line": "a = df['x'].mean()"},  # grounded
            {"code_line": "b = df['y'].sum()"},  # grounded
            {"code_line": "c = df['z'].max()"},  # NOT grounded
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
        grounded, ungrounded = validate_hooks_grounded([{"code_line": "x = 1"}], [])
        assert len(grounded) == 0
        assert len(ungrounded) == 1

    def test_false_positive_rejection(self):
        """x = 1 should NOT match x = 10 - reject false positives from substring matching."""
        hooks = [{"code_line": "x = 1"}]
        code_cells = ["x = 10"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 0, (
            "Should reject false positive: 'x = 1' is NOT in 'x = 10'"
        )
        assert len(ungrounded) == 1
        assert (
            ungrounded[0]["_ungrounded_reason"]
            == "code_line not found in executed code"
        )

    def test_exact_line_match_required(self):
        """Must match exact line, not just be a substring within a line."""
        # "df['col']" alone should not match a line containing "df['col'].mean()"
        hooks = [{"code_line": "df['col']"}]
        code_cells = ["result = df['col'].mean()"]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 0, "Should require exact line match, not substring"
        assert len(ungrounded) == 1

    def test_full_line_substring_still_matches(self):
        """Full line matches should still work even with extra content elsewhere."""
        hooks = [{"code_line": "result = df['col'].mean()"}]
        code_cells = [
            "# Step 1: Calculate mean",
            "result = df['col'].mean()",
            "print(result)",
        ]

        grounded, ungrounded = validate_hooks_grounded(hooks, code_cells)

        assert len(grounded) == 1
        assert len(ungrounded) == 0
