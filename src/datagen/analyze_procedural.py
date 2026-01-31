"""
CLI tool for analyzing procedural question pass rates.

Reads episodes from JSONL files, groups by various criteria, and calculates
pass rates. Outputs both JSON (for machines) and CLI table (for humans).

Usage:
    uv run python -m src.datagen.analyze_procedural --episodes data/episodes/procedural.jsonl
    uv run python -m src.datagen.analyze_procedural --episodes data/episodes/procedural.jsonl --json
    uv run python -m src.datagen.analyze_procedural --episodes data/episodes/procedural.jsonl --group-by operator
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_episodes(file_path: Path) -> list[dict]:
    """Load episodes from JSONL file.

    Args:
        file_path: Path to JSONL file containing episodes

    Returns:
        List of episode dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Episodes file not found: {file_path}")

    episodes = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    return episodes


def get_name_prefix(template_name: str | None) -> str:
    """Extract name prefix from template name.

    Args:
        template_name: Template name (e.g., "procedural_mean_age")

    Returns:
        Prefix before first underscore, or whole name if no underscore.
        Returns "unknown" for None input.
    """
    if template_name is None:
        return "unknown"

    parts = template_name.split("_")
    return parts[0] if parts else ""


# Known pandas/DataFrame operators
KNOWN_OPERATORS = {
    # Single ops
    "mean",
    "max",
    "min",
    "sum",
    "count",
    "std",
    "var",
    "median",
    # Multi-op sequences
    "filter",
    "groupby",
    "sort",
    "first",
    "last",
    "agg",
    "apply",
    "join",
    "merge",
    "pivot",
    "melt",
    "drop",
    "rename",
    "select",
}


def extract_operator_sequence(template_name: str | None) -> list[str]:
    """Extract operator sequence from template name.

    Args:
        template_name: Template name (e.g., "procedural_filter_sum")

    Returns:
        List of operators (consecutive known operators after prefix).
        Returns empty list for single-word templates or None.

    Patterns:
        - procedural_mean_age -> ["mean"] (single op + column "age")
        - procedural_max_salary -> ["max"] (single op + column "salary")
        - procedural_filter_sum -> ["filter", "sum"] (multi-op)
        - procedural_groupby_count -> ["groupby", "count"] (multi-op)
    """
    if template_name is None:
        return []

    parts = template_name.split("_")

    # Need at least prefix + 1 operator
    if len(parts) < 2:
        return []

    # Skip prefix, collect consecutive known operators
    remaining = parts[1:]
    operators = []

    for part in remaining:
        if part in KNOWN_OPERATORS:
            operators.append(part)
        else:
            # Stop at first non-operator (assumed to be column name)
            break

    return operators


class EpisodeAnalyzer:
    """Analyzer for episode pass rates."""

    def __init__(self, episodes: list[dict]):
        """Initialize with episodes.

        Args:
            episodes: List of episode dictionaries
        """
        self.episodes = episodes

    def filter_procedural(self) -> list[dict]:
        """Filter to procedural questions only.

        Returns:
            Episodes with template_name starting with "procedural_"
        """
        return [
            ep
            for ep in self.episodes
            if ep.get("question", {}).get("template_name", "").startswith("procedural_")
        ]

    def group_by_name_prefix(self) -> dict[str, list[dict]]:
        """Group episodes by name prefix.

        Returns:
            Dict mapping prefix to list of episodes
        """
        groups = defaultdict(list)
        for ep in self.episodes:
            template_name = ep.get("question", {}).get("template_name")
            prefix = get_name_prefix(template_name)
            groups[prefix].append(ep)
        return dict(groups)

    def group_by_operator_sequence(self) -> dict[tuple[str, ...], list[dict]]:
        """Group episodes by operator sequence.

        Returns:
            Dict mapping operator tuple to list of episodes
        """
        groups = defaultdict(list)
        for ep in self.episodes:
            template_name = ep.get("question", {}).get("template_name")
            operators = extract_operator_sequence(template_name)
            if operators:  # Only include if we found operators
                groups[tuple(operators)].append(ep)
        return dict(groups)

    def group_by_both(
        self,
    ) -> dict[tuple[str, tuple[str, ...]], list[dict]]:
        """Group episodes by both prefix and operator sequence.

        Returns:
            Dict mapping (prefix, operator_tuple) to list of episodes
        """
        groups = defaultdict(list)
        for ep in self.episodes:
            template_name = ep.get("question", {}).get("template_name")
            prefix = get_name_prefix(template_name)
            operators = extract_operator_sequence(template_name)
            key = (prefix, tuple(operators))
            groups[key].append(ep)
        return dict(groups)

    def _key_to_string(
        self, key: str | tuple[str, ...] | tuple[str, tuple[str, ...]]
    ) -> str:
        """Convert group key to string for JSON serialization.

        Args:
            key: Group key (string or tuple)

        Returns:
            String representation of key
        """
        if isinstance(key, str):
            return key

        # Handle tuple keys
        if len(key) == 2 and isinstance(key[1], tuple):
            # (prefix, (op1, op2, ...))
            prefix = key[0]
            ops = key[1]
            return f"{prefix}:{','.join(ops)}"
        else:
            # (op1, op2, ...) or (prefix, suffix)
            return ",".join(str(k) for k in key)

    def calculate_pass_rate(self, episodes: list[dict]) -> dict[str, Any]:
        """Calculate pass rate for a group of episodes.

        Args:
            episodes: List of episodes to analyze

        Returns:
            Dict with total, passed, failed, and pass_rate
        """
        total = len(episodes)
        passed = sum(1 for ep in episodes if ep.get("verified", False))
        failed = total - passed
        pass_rate = passed / total if total > 0 else 0.0

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 4),
        }

    def generate_report(
        self, group_by: str = "prefix", procedural_only: bool = True
    ) -> dict[str, Any]:
        """Generate analysis report.

        Args:
            group_by: How to group - "prefix", "operator", or "both"
            procedural_only: Whether to filter to procedural questions only

        Returns:
            Report dict with grouping, groups, and summary

        Raises:
            ValueError: If group_by is invalid
        """
        episodes = self.filter_procedural() if procedural_only else self.episodes

        if group_by == "prefix":
            groups = self.group_by_name_prefix()
            grouping_name = "name_prefix"
        elif group_by == "operator":
            groups = self.group_by_operator_sequence()
            grouping_name = "operator_sequence"
        elif group_by == "both":
            groups = self.group_by_both()
            grouping_name = "both"
        else:
            raise ValueError(
                f"Invalid group_by: {group_by}. Use 'prefix', 'operator', or 'both'"
            )

        # Calculate stats for each group
        group_stats = {}
        for key, group_episodes in groups.items():
            # Filter to procedural if needed
            if procedural_only:
                group_episodes = [
                    ep
                    for ep in group_episodes
                    if ep.get("question", {})
                    .get("template_name", "")
                    .startswith("procedural_")
                ]

            if group_episodes:  # Only include non-empty groups
                stats = self.calculate_pass_rate(group_episodes)
                # Convert tuple keys to strings for JSON serialization
                key_str = self._key_to_string(key)
                group_stats[key_str] = stats

        # Calculate overall summary
        all_procedural = self.filter_procedural()
        summary_stats = self.calculate_pass_rate(all_procedural)

        return {
            "grouping": grouping_name,
            "procedural_only": procedural_only,
            "groups": group_stats,
            "summary": {
                "total_episodes": len(self.episodes),
                "procedural_episodes": len(all_procedural),
                "total_passed": summary_stats["passed"],
                "total_failed": summary_stats["failed"],
                "overall_pass_rate": summary_stats["pass_rate"],
            },
        }


def format_table(report: dict) -> str:
    """Format report as CLI table.

    Args:
        report: Report dict from generate_report

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Procedural Questions Analysis")
    lines.append("=" * 70)
    lines.append(f"Grouping: {report['grouping']}")
    lines.append(f"Procedural only: {report['procedural_only']}")
    lines.append("")

    # Group stats table
    lines.append("-" * 70)
    lines.append(
        f"{'Group':<30} {'Total':>8} {'Passed':>8} {'Failed':>8} {'Pass %':>8}"
    )
    lines.append("-" * 70)

    for group_name, stats in sorted(report["groups"].items()):
        pass_pct = f"{stats['pass_rate'] * 100:.1f}%"
        lines.append(
            f"{group_name:<30} {stats['total']:>8} {stats['passed']:>8} "
            f"{stats['failed']:>8} {pass_pct:>8}"
        )

    lines.append("-" * 70)

    # Summary
    summary = report["summary"]
    lines.append("")
    lines.append("Summary:")
    lines.append(f"  Total episodes: {summary['total_episodes']}")
    lines.append(f"  Procedural episodes: {summary['procedural_episodes']}")
    lines.append(f"  Overall pass rate: {summary['overall_pass_rate'] * 100:.1f}%")
    lines.append(
        f"  ({summary['total_passed']} passed, {summary['total_failed']} failed)"
    )
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze procedural question pass rates from episode data"
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="Path to episodes JSONL file",
    )
    parser.add_argument(
        "--group-by",
        choices=["prefix", "operator", "both"],
        default="prefix",
        help="How to group episodes (default: prefix)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of table",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="include_all",
        help="Include non-procedural questions too",
    )

    args = parser.parse_args()

    try:
        episodes = load_episodes(args.episodes)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    analyzer = EpisodeAnalyzer(episodes)
    report = analyzer.generate_report(
        group_by=args.group_by,
        procedural_only=not args.include_all,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_table(report))


if __name__ == "__main__":
    main()
