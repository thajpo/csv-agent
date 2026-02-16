"""
Data generation statistics and coverage report.

Usage (via CLI):
    csvagent stats              # Full report
    csvagent stats --questions  # Questions only
    csvagent stats --episodes   # Episodes only
    csvagent stats --gaps       # Show gaps/missing data
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console()


def collect_questions_stats() -> dict:
    """Collect statistics about generated questions."""
    stats = {
        "synthetic": {"total": 0, "by_dataset": {}, "by_difficulty": Counter(), "by_template": Counter()},
        "llm": {"total": 0, "by_dataset": {}, "by_difficulty": Counter()},
    }

    # Synthetic questions
    synth_dir = Path("data/questions_synthetic")
    if synth_dir.exists():
        for qf in synth_dir.glob("*/questions.json"):
            dataset = qf.parent.name
            with open(qf) as f:
                data = json.load(f)
            questions = data.get("questions", data if isinstance(data, list) else [])

            stats["synthetic"]["total"] += len(questions)
            stats["synthetic"]["by_dataset"][dataset] = len(questions)

            for q in questions:
                diff = q.get("difficulty", "UNKNOWN")
                stats["synthetic"]["by_difficulty"][diff] += 1
                template = q.get("template_name", "unknown")
                stats["synthetic"]["by_template"][template] += 1

    # LLM questions
    llm_dir = Path("data/questions_llm")
    if llm_dir.exists():
        for qf in llm_dir.glob("*/questions.json"):
            dataset = qf.parent.name
            with open(qf) as f:
                data = json.load(f)
            questions = data.get("questions", data if isinstance(data, list) else [])

            stats["llm"]["total"] += len(questions)
            stats["llm"]["by_dataset"][dataset] = len(questions)

            for q in questions:
                diff = q.get("difficulty", "UNKNOWN")
                stats["llm"]["by_difficulty"][diff] += 1

    return stats


def collect_episodes_stats() -> dict:
    """Collect statistics about generated episodes."""
    stats = {
        "synthetic": {"total": 0, "verified": 0, "by_dataset": {}, "by_difficulty": Counter()},
        "llm": {"total": 0, "verified": 0, "by_dataset": {}, "by_difficulty": Counter()},
    }

    # Synthetic episodes
    synth_file = Path("data/episodes/episodes_synthetic.jsonl")
    if synth_file.exists():
        with open(synth_file) as f:
            for line in f:
                ep = json.loads(line)
                stats["synthetic"]["total"] += 1
                if ep.get("verified"):
                    stats["synthetic"]["verified"] += 1

                # Extract dataset from csv_source
                csv_source = ep.get("csv_source", "")
                dataset = Path(csv_source).parent.name if csv_source else "unknown"
                stats["synthetic"]["by_dataset"][dataset] = stats["synthetic"]["by_dataset"].get(dataset, 0) + 1

                diff = ep.get("question", {}).get("difficulty", "UNKNOWN")
                stats["synthetic"]["by_difficulty"][diff] += 1

    # LLM episodes
    llm_file = Path("data/episodes/episodes_llm.jsonl")
    if llm_file.exists():
        with open(llm_file) as f:
            for line in f:
                ep = json.loads(line)
                stats["llm"]["total"] += 1
                if ep.get("verified"):
                    stats["llm"]["verified"] += 1

                csv_source = ep.get("csv_source", "")
                dataset = Path(csv_source).parent.name if csv_source else "unknown"
                stats["llm"]["by_dataset"][dataset] = stats["llm"]["by_dataset"].get(dataset, 0) + 1

                diff = ep.get("question", {}).get("difficulty", "UNKNOWN")
                stats["llm"]["by_difficulty"][diff] += 1

    return stats


def collect_datasets() -> list[str]:
    """Get list of available datasets."""
    datasets = set()

    # From kaggle
    kaggle_dir = Path("data/kaggle")
    if kaggle_dir.exists():
        for d in kaggle_dir.iterdir():
            if d.is_dir() and (d / "data.csv").exists():
                datasets.add(d.name)

    # From csv dir
    csv_dir = Path("data/csv")
    if csv_dir.exists():
        for f in csv_dir.glob("*.csv"):
            datasets.add(f.stem)
        for d in csv_dir.iterdir():
            if d.is_dir() and (d / "data.csv").exists():
                datasets.add(d.name)

    return sorted(datasets)


def show_summary(q_stats: dict, e_stats: dict):
    """Show high-level summary."""
    console.print(Panel(
        f"[bold]Questions:[/bold]\n"
        f"  Synthetic: {q_stats['synthetic']['total']:,} ({len(q_stats['synthetic']['by_dataset'])} datasets)\n"
        f"  LLM: {q_stats['llm']['total']:,} ({len(q_stats['llm']['by_dataset'])} datasets)\n\n"
        f"[bold]Episodes:[/bold]\n"
        f"  Synthetic: {e_stats['synthetic']['total']:,} ({e_stats['synthetic']['verified']:,} verified, "
        f"{e_stats['synthetic']['verified']/max(e_stats['synthetic']['total'],1)*100:.0f}%)\n"
        f"  LLM: {e_stats['llm']['total']:,} ({e_stats['llm']['verified']:,} verified, "
        f"{e_stats['llm']['verified']/max(e_stats['llm']['total'],1)*100:.0f}%)",
        title="Data Generation Summary",
    ))


def show_questions_detail(q_stats: dict):
    """Show detailed questions breakdown."""
    console.print("\n[bold]Questions by Difficulty[/bold]\n")

    table = Table()
    table.add_column("Difficulty")
    table.add_column("Synthetic", justify="right")
    table.add_column("LLM", justify="right")

    difficulties = ["EASY", "MEDIUM", "HARD", "VERY_HARD", "UNKNOWN"]
    for diff in difficulties:
        s_count = q_stats["synthetic"]["by_difficulty"].get(diff, 0)
        l_count = q_stats["llm"]["by_difficulty"].get(diff, 0)
        if s_count or l_count:
            table.add_row(diff, str(s_count), str(l_count))

    console.print(table)

    # Top templates
    if q_stats["synthetic"]["by_template"]:
        console.print("\n[bold]Top Templates (Synthetic)[/bold]\n")
        table = Table()
        table.add_column("Template")
        table.add_column("Count", justify="right")

        for template, count in q_stats["synthetic"]["by_template"].most_common(10):
            table.add_row(template, str(count))

        console.print(table)


def show_episodes_detail(e_stats: dict):
    """Show detailed episodes breakdown."""
    console.print("\n[bold]Episodes by Difficulty[/bold]\n")

    table = Table()
    table.add_column("Difficulty")
    table.add_column("Synthetic", justify="right")
    table.add_column("LLM", justify="right")

    difficulties = ["EASY", "MEDIUM", "HARD", "VERY_HARD", "UNKNOWN"]
    for diff in difficulties:
        s_count = e_stats["synthetic"]["by_difficulty"].get(diff, 0)
        l_count = e_stats["llm"]["by_difficulty"].get(diff, 0)
        if s_count or l_count:
            table.add_row(diff, str(s_count), str(l_count))

    console.print(table)

    # By dataset
    console.print("\n[bold]Episodes by Dataset (Top 10)[/bold]\n")
    table = Table()
    table.add_column("Dataset")
    table.add_column("Synthetic", justify="right")
    table.add_column("LLM", justify="right")

    all_datasets = set(e_stats["synthetic"]["by_dataset"].keys()) | set(e_stats["llm"]["by_dataset"].keys())
    sorted_datasets = sorted(
        all_datasets,
        key=lambda d: e_stats["synthetic"]["by_dataset"].get(d, 0) + e_stats["llm"]["by_dataset"].get(d, 0),
        reverse=True
    )[:10]

    for dataset in sorted_datasets:
        s_count = e_stats["synthetic"]["by_dataset"].get(dataset, 0)
        l_count = e_stats["llm"]["by_dataset"].get(dataset, 0)
        table.add_row(dataset[:30], str(s_count), str(l_count))

    console.print(table)


def show_gaps(q_stats: dict, e_stats: dict):
    """Show datasets with missing data."""
    all_datasets = collect_datasets()

    console.print("\n[bold]Coverage Gaps[/bold]\n")

    # Datasets without questions
    datasets_with_synth_q = set(q_stats["synthetic"]["by_dataset"].keys())
    missing_synth_q = [d for d in all_datasets if d not in datasets_with_synth_q]

    if missing_synth_q:
        console.print(f"[yellow]Datasets without synthetic questions ({len(missing_synth_q)}):[/yellow]")
        for d in missing_synth_q[:10]:
            console.print(f"  • {d}")
        if len(missing_synth_q) > 10:
            console.print(f"  ... and {len(missing_synth_q) - 10} more")
        console.print()

    # Datasets with questions but no episodes
    datasets_with_synth_e = set(e_stats["synthetic"]["by_dataset"].keys())
    have_q_no_e = [d for d in datasets_with_synth_q if d not in datasets_with_synth_e]

    if have_q_no_e:
        console.print(f"[yellow]Datasets with questions but no episodes ({len(have_q_no_e)}):[/yellow]")
        for d in have_q_no_e[:10]:
            q_count = q_stats["synthetic"]["by_dataset"].get(d, 0)
            console.print(f"  • {d} ({q_count} questions)")
        if len(have_q_no_e) > 10:
            console.print(f"  ... and {len(have_q_no_e) - 10} more")
        console.print()

    # Low verification rate
    console.print("[bold]Low Verification Datasets[/bold]")
    # Would need per-dataset verified counts to show this


def main():
    parser = argparse.ArgumentParser(description="Data generation statistics")
    parser.add_argument("--questions", action="store_true", help="Show questions detail")
    parser.add_argument("--episodes", action="store_true", help="Show episodes detail")
    parser.add_argument("--gaps", action="store_true", help="Show coverage gaps")
    args = parser.parse_args()

    q_stats = collect_questions_stats()
    e_stats = collect_episodes_stats()

    show_summary(q_stats, e_stats)

    if args.questions or not (args.episodes or args.gaps):
        show_questions_detail(q_stats)

    if args.episodes or not (args.questions or args.gaps):
        show_episodes_detail(e_stats)

    if args.gaps:
        show_gaps(q_stats, e_stats)


