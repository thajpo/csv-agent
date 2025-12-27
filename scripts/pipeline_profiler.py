#!/usr/bin/env python
"""
Pipeline Profiler - Diagnostic tool for investigating pipeline issues.

Run with: uv run python scripts/pipeline_profiler.py [COMMAND]

Commands:
    containers    - Check Docker container health and leaks
    episodes      - Analyze episode JSONL files for data quality issues
    hooks         - Validate hook grounding in episodes
    timing        - Analyze timing distributions
    silent        - Detect silent failures in pipeline logs
"""

import argparse
import asyncio
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def cmd_containers(args):
    """Check Docker container health and detect leaks."""
    print("=" * 60)
    print("DOCKER CONTAINER ANALYSIS")
    print("=" * 60)

    # List all containers matching our naming patterns
    patterns = ["csv-sandbox-", "csv-mt-", "csv-analysis-"]

    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.CreatedAt}}"],
        capture_output=True, text=True
    )

    containers = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            name, status, created = parts[0], parts[1], parts[2]
            if any(p in name for p in patterns):
                containers.append({"name": name, "status": status, "created": created})

    print(f"\n[Container Count] {len(containers)} csv-agent containers found")

    if containers:
        # Group by status
        running = [c for c in containers if "Up" in c["status"]]
        exited = [c for c in containers if "Exited" in c["status"]]

        print(f"  Running: {len(running)}")
        print(f"  Exited:  {len(exited)}")

        if running:
            print("\n[Running Containers]")
            for c in running[:10]:
                print(f"  • {c['name']} - {c['status']}")
            if len(running) > 10:
                print(f"  ... and {len(running) - 10} more")

        if exited:
            print("\n[ISSUE: Orphaned Containers]")
            print("  Exited containers should be cleaned up.")
            for c in exited[:5]:
                print(f"  • {c['name']} - {c['status']}")
            if len(exited) > 5:
                print(f"  ... and {len(exited) - 5} more")
            print(f"\n  Fix: uv run python -c \"from src.utils.docker import cleanup_csv_sandbox_containers; cleanup_csv_sandbox_containers()\"")
    else:
        print("  No csv-agent containers found (good!)")

    # Check for zombie worker processes (containers with dead workers)
    print("\n[Worker Health Check]")
    for c in running[:3]:
        result = subprocess.run(
            ["docker", "exec", c["name"], "pgrep", "-c", "python"],
            capture_output=True, text=True
        )
        proc_count = result.stdout.strip() if result.returncode == 0 else "0"
        print(f"  • {c['name']}: {proc_count} python processes")


def cmd_episodes(args):
    """Analyze episode JSONL files for data quality issues."""
    print("=" * 60)
    print("EPISODE DATA QUALITY ANALYSIS")
    print("=" * 60)

    episodes_dir = Path("data/episodes")
    if not episodes_dir.exists():
        print(f"Episodes directory not found: {episodes_dir}")
        return 1

    jsonl_files = list(episodes_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found in data/episodes/")
        return 1

    for jsonl_path in jsonl_files:
        print(f"\n[Analyzing: {jsonl_path.name}]")

        episodes = []
        parse_errors = 0
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    parse_errors += 1

        if parse_errors:
            print(f"  [ISSUE] {parse_errors} lines failed to parse as JSON")

        if not episodes:
            print("  No episodes found")
            continue

        print(f"  Total episodes: {len(episodes)}")

        # Verification stats
        verified = sum(1 for e in episodes if e.get("verified", False))
        print(f"  Verified: {verified}/{len(episodes)} ({100*verified/len(episodes):.1f}%)")

        # Check for silent failures
        issues = defaultdict(list)

        for i, ep in enumerate(episodes):
            gold = ep.get("gold_trace", {})
            consistency = ep.get("consistency_traces", [])

            # Issue: Gold trace has no final answer
            if gold.get("final_answer") is None:
                issues["gold_no_answer"].append(i)

            # Issue: All consistency traces failed
            if consistency:
                succeeded = sum(1 for t in consistency if t.get("success", False))
                if succeeded == 0:
                    issues["all_consistency_failed"].append(i)

            # Issue: Verified=True but gold answer is None
            if ep.get("verified") and gold.get("final_answer") is None:
                issues["verified_but_no_gold_answer"].append(i)

            # Issue: No hooks in gold trace
            all_hooks = []
            for turn in gold.get("turns", []):
                exec_result = turn.get("execution", {})
                all_hooks.extend(exec_result.get("hooks", []))
            if not all_hooks:
                issues["gold_no_hooks"].append(i)

            # Issue: Mismatch between turn count and execution results
            turns = gold.get("turns", [])
            for j, turn in enumerate(turns):
                if turn.get("code") and not turn.get("execution", {}).get("stdout"):
                    issues["code_but_no_output"].append((i, j))

        if issues:
            print("\n  [DATA QUALITY ISSUES]")
            for issue_type, indices in issues.items():
                print(f"    • {issue_type}: {len(indices)} episodes")
                if len(indices) <= 3:
                    print(f"      Episode indices: {indices}")
        else:
            print("  No data quality issues detected!")

        # Difficulty distribution
        difficulties = Counter(ep.get("question", {}).get("difficulty") for ep in episodes)
        print(f"\n  Difficulty distribution: {dict(difficulties)}")


def cmd_hooks(args):
    """Validate hook grounding in episodes."""
    print("=" * 60)
    print("HOOK VALIDATION ANALYSIS")
    print("=" * 60)

    episodes_dir = Path("data/episodes")
    jsonl_files = list(episodes_dir.glob("*.jsonl"))

    if not jsonl_files:
        print("No episode files found")
        return 1

    for jsonl_path in jsonl_files:
        print(f"\n[{jsonl_path.name}]")

        episodes = []
        with open(jsonl_path) as f:
            for line in f:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        ungrounded_count = 0
        total_hooks = 0
        missing_code_line = 0

        for ep in episodes:
            gold = ep.get("gold_trace", {})
            all_code = []

            for turn in gold.get("turns", []):
                code = turn.get("code", "")
                if code:
                    all_code.append(code)

                for hook in turn.get("execution", {}).get("hooks", []):
                    total_hooks += 1
                    code_line = hook.get("code_line", "")

                    if not code_line:
                        missing_code_line += 1
                        continue

                    # Check if code_line is grounded
                    normalized_code = " ".join(" ".join(all_code).split())
                    normalized_line = " ".join(code_line.split())

                    if normalized_line not in normalized_code:
                        ungrounded_count += 1

        print(f"  Total hooks: {total_hooks}")
        if total_hooks > 0:
            print(f"  Missing code_line: {missing_code_line} ({100*missing_code_line/total_hooks:.1f}%)")
            print(f"  Ungrounded: {ungrounded_count} ({100*ungrounded_count/total_hooks:.1f}%)")

            if ungrounded_count > 0:
                print(f"  [ISSUE] {ungrounded_count} hooks have code_lines that don't match executed code")


def cmd_timing(args):
    """Analyze timing distributions in episodes."""
    print("=" * 60)
    print("TIMING ANALYSIS")
    print("=" * 60)

    episodes_dir = Path("data/episodes")
    jsonl_files = list(episodes_dir.glob("*.jsonl"))

    if not jsonl_files:
        print("No episode files found")
        return 1

    all_timings = {
        "gold": [],
        "consistency_avg": [],
        "total": [],
    }

    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    ep = json.loads(line)
                    timing = ep.get("timing", {})
                    if timing:
                        if "gold_elapsed" in timing:
                            all_timings["gold"].append(timing["gold_elapsed"])
                        if "avg_elapsed" in timing:
                            all_timings["consistency_avg"].append(timing["avg_elapsed"])
                        if "total_elapsed" in timing:
                            all_timings["total"].append(timing["total_elapsed"])
                except json.JSONDecodeError:
                    continue

    for name, times in all_timings.items():
        if times:
            times.sort()
            p50 = times[len(times)//2]
            p95 = times[int(len(times)*0.95)] if len(times) > 20 else times[-1]
            mean = sum(times) / len(times)

            print(f"\n[{name.upper()}]")
            print(f"  Count: {len(times)}")
            print(f"  Mean:  {mean:.2f}s")
            print(f"  P50:   {p50:.2f}s")
            print(f"  P95:   {p95:.2f}s")
            print(f"  Max:   {max(times):.2f}s")

            # Detect outliers (>3x median)
            outliers = [t for t in times if t > p50 * 3]
            if outliers:
                print(f"  [WARNING] {len(outliers)} outliers (>3x median)")


def cmd_silent(args):
    """Detect silent failures by scanning for common patterns."""
    print("=" * 60)
    print("SILENT FAILURE DETECTION")
    print("=" * 60)

    print("\n[Checking source code for silent failure patterns...]")

    patterns = [
        ("except.*:.*pass", "Silent exception swallowing"),
        ("except.*:.*continue", "Exception causes silent skip"),
        ("except.*JSONDecodeError.*continue", "JSON parse failure silently ignored"),
        (r"if .* is None:.*return", "Silent None return without logging"),
    ]

    src_dir = Path("src")
    issues_by_file = defaultdict(list)

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text()
        lines = content.split("\n")

        for i, line in enumerate(lines):
            for pattern, desc in patterns:
                import re
                if re.search(pattern, line, re.IGNORECASE):
                    issues_by_file[str(py_file)].append((i+1, desc, line.strip()[:60]))

    if issues_by_file:
        print("\n[Potential Silent Failure Locations]")
        for filepath, issues in sorted(issues_by_file.items()):
            print(f"\n  {filepath}:")
            for line_no, desc, snippet in issues[:5]:
                print(f"    L{line_no}: {desc}")
                print(f"           {snippet}...")
            if len(issues) > 5:
                print(f"    ... and {len(issues) - 5} more")
    else:
        print("  No obvious silent failure patterns detected")

    # Check for missing error handling in key functions
    print("\n[Key Functions Without Error Logging]")
    key_functions = [
        ("src/datagen/teacher.py", "parse_hooks_from_stdout"),
        ("src/datagen/teacher.py", "answers_match"),
        ("src/core/environment.py", "parse_submitted_answer"),
    ]

    for filepath, func_name in key_functions:
        path = Path(filepath)
        if path.exists():
            content = path.read_text()
            if f"def {func_name}" in content:
                # Check if function has logging
                func_start = content.index(f"def {func_name}")
                # Find next function or end
                next_def = content.find("\ndef ", func_start + 1)
                func_body = content[func_start:next_def] if next_def > 0 else content[func_start:]

                has_logging = "logging." in func_body or "print(" in func_body
                if not has_logging:
                    print(f"  • {filepath}:{func_name}() - no logging/print statements")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline profiling and diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("containers", help="Check Docker container health")
    subparsers.add_parser("episodes", help="Analyze episode data quality")
    subparsers.add_parser("hooks", help="Validate hook grounding")
    subparsers.add_parser("timing", help="Analyze timing distributions")
    subparsers.add_parser("silent", help="Detect silent failure patterns")
    subparsers.add_parser("all", help="Run all checks")

    args = parser.parse_args()

    if args.command == "containers":
        cmd_containers(args)
    elif args.command == "episodes":
        cmd_episodes(args)
    elif args.command == "hooks":
        cmd_hooks(args)
    elif args.command == "timing":
        cmd_timing(args)
    elif args.command == "silent":
        cmd_silent(args)
    elif args.command == "all":
        cmd_containers(args)
        cmd_episodes(args)
        cmd_hooks(args)
        cmd_timing(args)
        cmd_silent(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
