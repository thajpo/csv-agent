"""
Tests for procedural question analysis pipeline.

This module tests the analyze_procedural CLI tool which:
1. Reads episodes from JSONL files
2. Groups by name prefix, operator sequence, or both
3. Calculates pass rates per group
4. Outputs JSON (machines) and CLI table (humans)

Run with: uv run pytest tests/test_analyze_procedural.py -v
"""

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from src.datagen.analyze_procedural import (
    EpisodeAnalyzer,
    extract_operator_sequence,
    get_name_prefix,
    load_episodes,
    main,
)


FIXTURE_PATH = Path("tests/fixtures/mock_episodes.jsonl")


class TestLoadEpisodes:
    """Test episode loading from JSONL."""

    def test_load_episodes_from_fixture(self):
        """Should load all episodes from fixture file."""
        episodes = load_episodes(FIXTURE_PATH)
        assert len(episodes) == 7

    def test_load_episodes_returns_list_of_dicts(self):
        """Should return list of dictionaries."""
        episodes = load_episodes(FIXTURE_PATH)
        assert isinstance(episodes, list)
        assert all(isinstance(ep, dict) for ep in episodes)

    def test_load_episodes_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_episodes(Path("nonexistent.jsonl"))


class TestNamePrefixExtraction:
    """Test extraction of name prefixes from template names."""

    def test_get_name_prefix_procedural(self):
        """Should extract 'procedural' from 'procedural_mean_age'."""
        assert get_name_prefix("procedural_mean_age") == "procedural"

    def test_get_name_prefix_correlation(self):
        """Should extract 'correlation' from 'correlation_analysis'."""
        assert get_name_prefix("correlation_analysis") == "correlation"

    def test_get_name_prefix_single_word(self):
        """Should return whole word if no underscore."""
        assert get_name_prefix("simple") == "simple"

    def test_get_name_prefix_empty(self):
        """Should return empty string for empty input."""
        assert get_name_prefix("") == ""

    def test_get_name_prefix_none(self):
        """Should return 'unknown' for None input."""
        assert get_name_prefix(None) == "unknown"


class TestOperatorSequenceExtraction:
    """Test extraction of operator sequences from template names."""

    def test_extract_operator_sequence_mean(self):
        """Should extract ['mean'] from 'procedural_mean_age'."""
        assert extract_operator_sequence("procedural_mean_age") == ["mean"]

    def test_extract_operator_sequence_max(self):
        """Should extract ['max'] from 'procedural_max_salary'."""
        assert extract_operator_sequence("procedural_max_salary") == ["max"]

    def test_extract_operator_sequence_filter_sum(self):
        """Should extract ['filter', 'sum'] from 'procedural_filter_sum'."""
        assert extract_operator_sequence("procedural_filter_sum") == ["filter", "sum"]

    def test_extract_operator_sequence_groupby_count(self):
        """Should extract ['groupby', 'count'] from 'procedural_groupby_count'."""
        assert extract_operator_sequence("procedural_groupby_count") == [
            "groupby",
            "count",
        ]

    def test_extract_operator_sequence_sort_first(self):
        """Should extract ['sort', 'first'] from 'procedural_sort_first'."""
        assert extract_operator_sequence("procedural_sort_first") == ["sort", "first"]

    def test_extract_operator_sequence_single_word(self):
        """Should return empty list for single word template."""
        assert extract_operator_sequence("simple") == []

    def test_extract_operator_sequence_empty(self):
        """Should return empty list for empty input."""
        assert extract_operator_sequence("") == []

    def test_extract_operator_sequence_none(self):
        """Should return empty list for None input."""
        assert extract_operator_sequence(None) == []


class TestEpisodeAnalyzer:
    """Test the EpisodeAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with fixture data."""
        episodes = load_episodes(FIXTURE_PATH)
        return EpisodeAnalyzer(episodes)

    def test_analyzer_initialization(self, analyzer):
        """Should initialize with episodes."""
        assert len(analyzer.episodes) == 7

    def test_filter_procedural_only(self, analyzer):
        """Should filter to procedural questions only."""
        filtered = analyzer.filter_procedural()
        assert len(filtered) == 6  # 7 total - 1 non-procedural

    def test_group_by_name_prefix(self, analyzer):
        """Should group episodes by name prefix."""
        groups = analyzer.group_by_name_prefix()
        assert "procedural" in groups
        assert "correlation" in groups
        assert len(groups["procedural"]) == 6
        assert len(groups["correlation"]) == 1

    def test_group_by_operator_sequence(self, analyzer):
        """Should group episodes by operator sequence."""
        groups = analyzer.group_by_operator_sequence()
        # Check that we have operator-based groups
        assert len(groups) > 0
        # Each group key should be a tuple of operators
        for key in groups:
            assert isinstance(key, tuple)

    def test_group_by_both(self, analyzer):
        """Should group by both prefix and operator sequence."""
        groups = analyzer.group_by_both()
        # Check structure: keys should be (prefix, operator_tuple)
        for key in groups:
            assert isinstance(key, tuple)
            assert len(key) == 2
            prefix, ops = key
            assert isinstance(prefix, str)
            assert isinstance(ops, tuple)

    def test_calculate_pass_rate_all_pass(self):
        """Should calculate 100% pass rate when all pass."""
        episodes = [
            {"verified": True},
            {"verified": True},
            {"verified": True},
        ]
        analyzer = EpisodeAnalyzer(episodes)
        stats = analyzer.calculate_pass_rate(episodes)
        assert stats["total"] == 3
        assert stats["passed"] == 3
        assert stats["failed"] == 0
        assert stats["pass_rate"] == 1.0

    def test_calculate_pass_rate_all_fail(self):
        """Should calculate 0% pass rate when all fail."""
        episodes = [
            {"verified": False},
            {"verified": False},
        ]
        analyzer = EpisodeAnalyzer(episodes)
        stats = analyzer.calculate_pass_rate(episodes)
        assert stats["total"] == 2
        assert stats["passed"] == 0
        assert stats["failed"] == 2
        assert stats["pass_rate"] == 0.0

    def test_calculate_pass_rate_mixed(self):
        """Should calculate correct pass rate for mixed results."""
        episodes = [
            {"verified": True},
            {"verified": True},
            {"verified": False},
            {"verified": False},
        ]
        analyzer = EpisodeAnalyzer(episodes)
        stats = analyzer.calculate_pass_rate(episodes)
        assert stats["total"] == 4
        assert stats["passed"] == 2
        assert stats["failed"] == 2
        assert stats["pass_rate"] == 0.5

    def test_generate_report_by_prefix(self, analyzer):
        """Should generate report grouped by prefix."""
        report = analyzer.generate_report(group_by="prefix")
        assert "grouping" in report
        assert report["grouping"] == "name_prefix"
        assert "groups" in report
        assert "summary" in report

        # Check procedural group stats
        proc_group = report["groups"]["procedural"]
        assert proc_group["total"] == 6
        assert proc_group["passed"] == 4  # 4 pass, 2 fail
        assert proc_group["failed"] == 2

    def test_generate_report_by_operator(self, analyzer):
        """Should generate report grouped by operator sequence."""
        report = analyzer.generate_report(group_by="operator")
        assert report["grouping"] == "operator_sequence"
        assert "groups" in report

    def test_generate_report_by_both(self, analyzer):
        """Should generate report grouped by both."""
        report = analyzer.generate_report(group_by="both")
        assert report["grouping"] == "both"
        assert "groups" in report

    def test_generate_report_invalid_group_by(self, analyzer):
        """Should raise ValueError for invalid group_by."""
        with pytest.raises(ValueError):
            analyzer.generate_report(group_by="invalid")


class TestCLI:
    """Test CLI interface."""

    def test_main_json_output(self, capsys, monkeypatch):
        """Should output JSON when --json flag is used."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["analyze_procedural", "--episodes", str(FIXTURE_PATH), "--json"],
        )
        main()
        captured = capsys.readouterr()
        # Should be valid JSON
        output = json.loads(captured.out)
        assert "grouping" in output
        assert "groups" in output

    def test_main_table_output(self, capsys, monkeypatch):
        """Should output table when no --json flag."""
        monkeypatch.setattr(
            sys, "argv", ["analyze_procedural", "--episodes", str(FIXTURE_PATH)]
        )
        main()
        captured = capsys.readouterr()
        # Should contain table-like output
        assert (
            "Procedural Questions Analysis" in captured.out or "Group" in captured.out
        )

    def test_main_with_group_by(self, capsys, monkeypatch):
        """Should respect --group-by argument."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "analyze_procedural",
                "--episodes",
                str(FIXTURE_PATH),
                "--group-by",
                "operator",
                "--json",
            ],
        )
        main()
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["grouping"] == "operator_sequence"

    def test_main_missing_file(self, monkeypatch):
        """Should exit with error for missing file."""
        monkeypatch.setattr(
            sys, "argv", ["analyze_procedural", "--episodes", "nonexistent.jsonl"]
        )
        with pytest.raises(SystemExit):
            main()


class TestIntegration:
    """Integration tests with mock data."""

    def test_end_to_end_analysis(self):
        """Full analysis pipeline on mock data."""
        episodes = load_episodes(FIXTURE_PATH)
        analyzer = EpisodeAnalyzer(episodes)

        # Test all grouping methods
        for group_by in ["prefix", "operator", "both"]:
            report = analyzer.generate_report(group_by=group_by)

            # Verify report structure
            assert "grouping" in report
            assert "groups" in report
            assert "summary" in report
            assert "total_episodes" in report["summary"]
            assert "overall_pass_rate" in report["summary"]

            # Verify all groups have stats
            for group_name, stats in report["groups"].items():
                assert "total" in stats
                assert "passed" in stats
                assert "failed" in stats
                assert "pass_rate" in stats
                assert stats["total"] == stats["passed"] + stats["failed"]
                assert 0.0 <= stats["pass_rate"] <= 1.0

    def test_procedural_filtering(self):
        """Should correctly identify procedural questions."""
        episodes = load_episodes(FIXTURE_PATH)
        analyzer = EpisodeAnalyzer(episodes)

        procedural = analyzer.filter_procedural()
        # All procedural templates start with "procedural_"
        for ep in procedural:
            template_name = ep.get("question", {}).get("template_name", "")
            assert template_name.startswith("procedural_")
