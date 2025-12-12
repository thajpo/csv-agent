#!/usr/bin/env python3
"""
Test script to validate prompt token optimization.

Compares different verbosity modes and configurations for the student prompt.

NOTE: This test file references old functions that may no longer exist:
  - build_student_prompt
  - extract_tools_from_trace
  - format_tool_docs (from src.tools)
This file needs to be updated to work with the refactored codebase.
"""

from src.authoring.prompts import (
    # build_student_prompt,  # TODO: Function doesn't exist in refactored code
    generate_data_overview,
    DEFAULT_DATASET_DESCRIPTION,
    # extract_tools_from_trace,  # TODO: Function doesn't exist in refactored code
)
# from src.tools import format_tool_docs  # TODO: src.tools module doesn't exist


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


def print_comparison():
    """Compare different prompt configurations."""

    # Generate test data
    try:
        data_overview = generate_data_overview("data.csv")
    except FileNotFoundError:
        # Fallback for testing without data file
        data_overview = """=== SHAPE ===
Rows: 100, Columns: 5

=== HEAD ===
   col1  col2  col3
0     1     2     3
1     4     5     6"""

    print("=" * 80)
    print("PROMPT TOKEN OPTIMIZATION COMPARISON")
    print("=" * 80)

    # Test 1: Current compact mode (baseline)
    print("\n1. BASELINE (compact verbosity, full data overview)")
    print("-" * 80)
    prompt_compact = build_student_prompt(
        dataset_description=DEFAULT_DATASET_DESCRIPTION,
        data_overview=data_overview,
        verbosity="compact",
        include_data_overview=True,
    )
    tokens_compact = estimate_tokens(prompt_compact)
    print(f"Token count: ~{tokens_compact:,}")
    print(f"Character count: {len(prompt_compact):,}")

    # Test 2: Minimal mode with full data
    print("\n2. MINIMAL VERBOSITY (function signatures, full data overview)")
    print("-" * 80)
    prompt_minimal = build_student_prompt(
        dataset_description=DEFAULT_DATASET_DESCRIPTION,
        data_overview=data_overview,
        verbosity="minimal",
        include_data_overview=True,
    )
    tokens_minimal = estimate_tokens(prompt_minimal)
    print(f"Token count: ~{tokens_minimal:,}")
    print(f"Character count: {len(prompt_minimal):,}")
    print(f"Savings vs baseline: {tokens_compact - tokens_minimal:,} tokens ({(1 - tokens_minimal/tokens_compact)*100:.1f}%)")

    # Test 3: Minimal mode without data (structural compression)
    print("\n3. MINIMAL + NO DATA OVERVIEW (assumes teacher trace has data)")
    print("-" * 80)
    prompt_minimal_no_data = build_student_prompt(
        dataset_description=DEFAULT_DATASET_DESCRIPTION,
        data_overview=data_overview,
        verbosity="minimal",
        include_data_overview=False,
    )
    tokens_minimal_no_data = estimate_tokens(prompt_minimal_no_data)
    print(f"Token count: ~{tokens_minimal_no_data:,}")
    print(f"Character count: {len(prompt_minimal_no_data):,}")
    print(f"Savings vs baseline: {tokens_compact - tokens_minimal_no_data:,} tokens ({(1 - tokens_minimal_no_data/tokens_compact)*100:.1f}%)")

    # Test 4: Lazy tool loading (simulated with 8 tools)
    print("\n4. MINIMAL + NO DATA + LAZY LOADING (8 tools only)")
    print("-" * 80)
    # Simulate teacher using these tools
    common_tools = {"group_stat", "group_extremum", "combine", "lookup",
                    "correlation", "filter_stat", "inspect", "describe"}
    prompt_lazy = build_student_prompt(
        dataset_description=DEFAULT_DATASET_DESCRIPTION,
        data_overview=data_overview,
        verbosity="minimal",
        include_data_overview=False,
        filter_tools=common_tools,
    )
    tokens_lazy = estimate_tokens(prompt_lazy)
    print(f"Token count: ~{tokens_lazy:,}")
    print(f"Character count: {len(prompt_lazy):,}")
    print(f"Savings vs baseline: {tokens_compact - tokens_lazy:,} tokens ({(1 - tokens_lazy/tokens_compact)*100:.1f}%)")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<45} {'Tokens':>10} {'Savings':>15}")
    print("-" * 80)
    print(f"{'1. Baseline (compact + full data)':<45} {tokens_compact:>10,} {'-':>15}")
    print(f"{'2. Minimal verbosity':<45} {tokens_minimal:>10,} {f'-{tokens_compact - tokens_minimal:,} ({(1 - tokens_minimal/tokens_compact)*100:.1f}%)':>15}")
    print(f"{'3. Minimal + no data overview':<45} {tokens_minimal_no_data:>10,} {f'-{tokens_compact - tokens_minimal_no_data:,} ({(1 - tokens_minimal_no_data/tokens_compact)*100:.1f}%)':>15}")
    print(f"{'4. Minimal + no data + lazy (8 tools)':<45} {tokens_lazy:>10,} {f'-{tokens_compact - tokens_lazy:,} ({(1 - tokens_lazy/tokens_compact)*100:.1f}%)':>15}")
    print("=" * 80)

    # Tool docs comparison
    print("\nTOOL DOCUMENTATION COMPARISON")
    print("=" * 80)

    tool_docs_full = format_tool_docs(verbosity="full")
    tool_docs_compact = format_tool_docs(verbosity="compact")
    tool_docs_minimal = format_tool_docs(verbosity="minimal")
    tool_docs_minimal_lazy = format_tool_docs(verbosity="minimal", filter_tools=common_tools)

    print(f"{'Mode':<20} {'Tokens':>10} {'Characters':>12}")
    print("-" * 80)
    print(f"{'Full':<20} {estimate_tokens(tool_docs_full):>10,} {len(tool_docs_full):>12,}")
    print(f"{'Compact':<20} {estimate_tokens(tool_docs_compact):>10,} {len(tool_docs_compact):>12,}")
    print(f"{'Minimal (28 tools)':<20} {estimate_tokens(tool_docs_minimal):>10,} {len(tool_docs_minimal):>12,}")
    print(f"{'Minimal (8 tools)':<20} {estimate_tokens(tool_docs_minimal_lazy):>10,} {len(tool_docs_minimal_lazy):>12,}")
    print("=" * 80)

    # Show sample of minimal mode output
    print("\nSAMPLE: MINIMAL MODE TOOL DOCS (first 500 chars)")
    print("-" * 80)
    print(tool_docs_minimal[:500] + "...")
    print("=" * 80)


if __name__ == "__main__":
    print_comparison()
