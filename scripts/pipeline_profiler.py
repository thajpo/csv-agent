#!/usr/bin/env python
"""
Pipeline Profiler - Diagnostic tool for investigating pipeline issues.

Run with: uv run python scripts/pipeline_profiler.py [COMMAND]

Commands:
    containers    - Check Docker container health and leaks
    episodes      - Analyze episode JSONL files for data quality issues
    hooks         - Validate hook grounding in episodes
    timing        - Analyze timing distributions
    taxonomy      - Failure taxonomy for episode traces
    perf          - Performance summary from episode timings
    triangulation - Profile verification vs consistency count
    silent        - Detect silent failures in pipeline logs
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _load_episodes(episodes_dir: Path) -> dict[str, list[dict]]:
    """Load all episodes from JSONL files in a directory."""
    files = list(episodes_dir.glob("*.jsonl"))
    episodes_by_file: dict[str, list[dict]] = {}
    for jsonl_path in files:
        episodes = []
        with open(jsonl_path) as f:
            for line in f:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        episodes_by_file[jsonl_path.name] = episodes
    return episodes_by_file


def _iter_episodes(episodes_dir: Path):
    """Yield (file_name, index, episode) tuples."""
    for jsonl_path in episodes_dir.glob("*.jsonl"):
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                try:
                    yield jsonl_path.name, i, json.loads(line)
                except json.JSONDecodeError:
                    continue


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
            print("\n  Fix: uv run python -c \"from src.utils.docker import cleanup_csv_sandbox_containers; cleanup_csv_sandbox_containers()\"")
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

    episodes_dir = Path(args.episodes_dir)
    if not episodes_dir.exists():
        print(f"Episodes directory not found: {episodes_dir}")
        return 1

    jsonl_files = list(episodes_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {episodes_dir}")
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

    episodes_dir = Path(args.episodes_dir)
    jsonl_files = list(episodes_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No episode files found in {episodes_dir}")
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

    episodes_dir = Path(args.episodes_dir)
    jsonl_files = list(episodes_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No episode files found in {episodes_dir}")
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


def cmd_taxonomy(args):
    """Classify failures into a simple taxonomy for triage."""
    print("=" * 60)
    print("FAILURE TAXONOMY")
    print("=" * 60)

    episodes_dir = Path(args.episodes_dir)
    if not episodes_dir.exists():
        print(f"Episodes directory not found: {episodes_dir}")
        return 1

    taxonomy_counts = Counter()
    totals = 0

    for _, _, ep in _iter_episodes(episodes_dir):
        totals += 1
        gold = ep.get("gold_trace", {})
        consistency = ep.get("consistency_traces", [])
        triangulation = ep.get("triangulation", {})

        if not gold.get("success", False):
            taxonomy_counts["gold_failed"] += 1
        if gold.get("final_answer") is None:
            taxonomy_counts["gold_no_answer"] += 1

        gold_turns = gold.get("turns", [])
        if any(not t.get("execution", {}).get("success", True) for t in gold_turns):
            taxonomy_counts["gold_exec_error"] += 1

        gold_hooks = []
        for turn in gold_turns:
            gold_hooks.extend(turn.get("execution", {}).get("hooks", []))
        if not gold_hooks:
            taxonomy_counts["gold_no_hooks"] += 1

        if consistency:
            all_failed = all(
                (not t.get("success", False)) or t.get("final_answer") is None
                for t in consistency
            )
            if all_failed:
                taxonomy_counts["consistency_all_failed"] += 1

            majority_count = triangulation.get("majority_count")
            n_runs = triangulation.get("n_consistency_runs")
            if majority_count is not None and n_runs:
                required = (n_runs // 2) + 1
                if majority_count < required:
                    taxonomy_counts["low_agreement"] += 1

        if not ep.get("verified", False):
            taxonomy_counts["unverified"] += 1

    if totals == 0:
        print("No episodes found")
        return 1

    print(f"\nTotal episodes: {totals}")
    for label, count in taxonomy_counts.most_common():
        print(f"  {label}: {count} ({100*count/max(totals,1):.1f}%)")

    return 0


def cmd_perf(args):
    """Summarize performance metrics from episode timings."""
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    episodes_dir = Path(args.episodes_dir)
    if not episodes_dir.exists():
        print(f"Episodes directory not found: {episodes_dir}")
        return 1

    episodes_by_file = _load_episodes(episodes_dir)
    if not episodes_by_file:
        print("No JSONL files found in data/episodes/")
        return 1

    for filename, episodes in episodes_by_file.items():
        if not episodes:
            continue

        gold_times = []
        avg_times = []
        total_times = []
        timestamps = []

        for ep in episodes:
            timing = ep.get("timing", {})
            if "gold_elapsed" in timing:
                gold_times.append(timing["gold_elapsed"])
            if "avg_elapsed" in timing:
                avg_times.append(timing["avg_elapsed"])
            if "total_elapsed" in timing:
                total_times.append(timing["total_elapsed"])
            ts = ep.get("timestamp")
            if ts:
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except ValueError:
                    pass

        print(f"\n[{filename}]")
        print(f"  Episodes: {len(episodes)}")

        def _print_stats(label: str, values: list[float]):
            if not values:
                return
            values = sorted(values)
            p50 = values[len(values) // 2]
            p95 = values[int(len(values) * 0.95)] if len(values) > 20 else values[-1]
            mean = sum(values) / len(values)
            print(f"  {label}: mean {mean:.2f}s | p50 {p50:.2f}s | p95 {p95:.2f}s | max {max(values):.2f}s")

        _print_stats("gold", gold_times)
        _print_stats("avg", avg_times)
        _print_stats("total", total_times)

        if len(timestamps) >= 2:
            timestamps.sort()
            elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
            if elapsed > 0:
                rate = len(timestamps) / elapsed * 3600
                print(f"  throughput: {rate:.1f} episodes/hour")

    return 0


def cmd_triangulation(args):
    """Profile verification rate vs number of consistency traces."""
    print("=" * 60)
    print("TRIANGULATION PROFILING")
    print("=" * 60)

    from src.core.config import config
    from src.datagen.teacher import answers_match, get_majority_answer

    episodes_dir = Path(args.episodes_dir)
    if not episodes_dir.exists():
        print(f"Episodes directory not found: {episodes_dir}")
        return 1

    max_k = args.max_k
    if max_k is None:
        max_k = config.n_consistency

    totals_by_k = Counter()
    verified_by_k = Counter()

    for _, _, ep in _iter_episodes(episodes_dir):
        consistency = ep.get("consistency_traces", [])
        if not consistency:
            continue
        gold_answer = ep.get("gold_trace", {}).get("final_answer")
        if gold_answer is None:
            continue

        for k in range(1, min(max_k, len(consistency)) + 1):
            subset = consistency[:k]
            submitted = [
                trace.get("final_answer")
                for trace in subset
                if trace.get("final_answer") is not None
            ]
            if not submitted:
                totals_by_k[k] += 1
                continue
            majority_value, _ = get_majority_answer(submitted, float_tol=config.float_tolerance)
            verified = answers_match(None, None, gold_answer, majority_value, float_tol=config.float_tolerance)
            totals_by_k[k] += 1
            if verified:
                verified_by_k[k] += 1

    if not totals_by_k:
        print("No triangulated episodes found")
        return 1

    for k in range(1, max(totals_by_k.keys()) + 1):
        total = totals_by_k.get(k, 0)
        verified = verified_by_k.get(k, 0)
        if total == 0:
            continue
        print(f"  k={k}: {verified}/{total} verified ({100*verified/total:.1f}%)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline profiling and diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--episodes-dir",
        default="data/episodes",
        help="Episodes directory (default: data/episodes)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("containers", help="Check Docker container health")
    subparsers.add_parser("episodes", help="Analyze episode data quality")
    subparsers.add_parser("hooks", help="Validate hook grounding")
    subparsers.add_parser("timing", help="Analyze timing distributions")
    subparsers.add_parser("taxonomy", help="Failure taxonomy summary")
    subparsers.add_parser("perf", help="Performance summary")
    triang_parser = subparsers.add_parser("triangulation", help="Triangulation profiling")
    triang_parser.add_argument("--max-k", type=int, default=None, help="Max consistency traces to evaluate")
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
    elif args.command == "taxonomy":
        cmd_taxonomy(args)
    elif args.command == "perf":
        cmd_perf(args)
    elif args.command == "triangulation":
        cmd_triangulation(args)
    elif args.command == "silent":
        cmd_silent(args)
    elif args.command == "all":
        cmd_containers(args)
        cmd_episodes(args)
        cmd_hooks(args)
        cmd_timing(args)
        cmd_taxonomy(args)
        cmd_perf(args)
        cmd_triangulation(args)
        cmd_silent(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
