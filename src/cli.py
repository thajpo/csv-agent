"""
Unified CLI for csv-agent data generation pipeline.

Usage:
    csvagent              # Interactive menu (default)
    csvagent status       # View data inventory
    csvagent generate questions --template|--procedural|--llm-gen|--all
    csvagent generate episodes --template|--procedural|--llm-gen|--all
    csvagent run          # Full pipeline
    csvagent inspect      # Preview data
    csvagent validate     # Debug single question
    csvagent stats        # Coverage report
    csvagent profiler     # Diagnostics toolbox
"""

import argparse
import subprocess
import sys
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============= Status Command =============


def cmd_status():
    """Show data inventory - the most important visibility command."""
    from src.utils.stats import (
        collect_questions_stats,
        collect_episodes_stats,
        collect_datasets,
    )

    datasets = collect_datasets()
    q_stats = collect_questions_stats()
    e_stats = collect_episodes_stats()

    # Compact status display
    console.print()
    console.print(
        Panel.fit("[bold]csv-agent[/bold] Data Generation Pipeline", style="cyan")
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="dim")
    table.add_column("Value")

    # Datasets
    table.add_row("Datasets", f"[bold]{len(datasets)}[/bold] available")
    table.add_row("", "")

    # Questions
    synth_q = q_stats["synthetic"]["total"]
    synth_q_datasets = len(q_stats["synthetic"]["by_dataset"])
    llm_q = q_stats["llm"]["total"]
    llm_q_datasets = len(q_stats["llm"]["by_dataset"])

    table.add_row(
        "Questions",
        f"[green]synthetic[/green] {synth_q:,} ({synth_q_datasets} datasets) | "
        f"[blue]llm[/blue] {llm_q:,} ({llm_q_datasets} datasets)",
    )

    # Episodes
    synth_e = e_stats["synthetic"]["total"]
    synth_e_verified = e_stats["synthetic"]["verified"]
    llm_e = e_stats["llm"]["total"]
    llm_e_verified = e_stats["llm"]["verified"]

    synth_pct = f"{synth_e_verified / max(synth_e, 1) * 100:.0f}%" if synth_e else "0%"
    llm_pct = f"{llm_e_verified / max(llm_e, 1) * 100:.0f}%" if llm_e else "0%"

    table.add_row(
        "Episodes",
        f"[green]synthetic[/green] {synth_e_verified}/{synth_e} verified ({synth_pct}) | "
        f"[blue]llm[/blue] {llm_e_verified}/{llm_e} verified ({llm_pct})",
    )

    console.print(table)
    console.print()

    # Suggest next action
    if synth_q == 0:
        console.print("[dim]Next:[/dim] csvagent generate questions --template")
    elif synth_e == 0:
        console.print("[dim]Next:[/dim] csvagent generate episodes --template")
    elif llm_q == 0:
        console.print("[dim]Next:[/dim] csvagent generate questions --llm-gen")
    elif llm_e == 0:
        console.print("[dim]Next:[/dim] csvagent generate episodes --llm-gen")
    else:
        console.print("[dim]Pipeline complete! Run 'csvagent stats' for details.[/dim]")

    console.print()
    return 0


def cmd_progress():
    """Show detailed pipeline progress with estimates."""
    import json
    from datetime import datetime

    from src.core.config import config
    from src.utils.stats import collect_questions_stats, collect_episodes_stats

    q_stats = collect_questions_stats()
    e_stats = collect_episodes_stats()

    # Calculate totals
    total_synth_q = q_stats["synthetic"]["total"]
    total_llm_q = q_stats["llm"]["total"]
    total_synth_e = e_stats["synthetic"]["total"]
    total_llm_e = e_stats["llm"]["total"]

    console.print()
    console.print(Panel.fit("[bold]Pipeline Progress Report[/bold]", style="cyan"))

    # Stage breakdown
    table = Table(title="Stage Completion", show_header=True)
    table.add_column("Stage", style="bold")
    table.add_column("Synthetic", justify="right")
    table.add_column("LLM", justify="right")
    table.add_column("Progress", justify="center")

    def progress_bar(done, total, width=20):
        if total == 0:
            return "[dim]no data[/dim]"
        pct = done / total
        filled = int(pct * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {pct * 100:.0f}%"

    # Questions (consider 100% if we have any)
    synth_q_pct = 100 if total_synth_q > 0 else 0
    llm_q_pct = 100 if total_llm_q > 0 else 0
    table.add_row(
        "1. Questions",
        f"{total_synth_q:,}",
        f"{total_llm_q:,}",
        "[green]Complete[/green]"
        if total_synth_q + total_llm_q > 0
        else "[red]Not started[/red]",
    )

    # Episodes
    table.add_row(
        "2. Episodes",
        f"{total_synth_e:,} / {total_synth_q:,}",
        f"{total_llm_e:,} / {total_llm_q:,}",
        progress_bar(total_synth_e + total_llm_e, total_synth_q + total_llm_q),
    )

    console.print(table)

    def load_per_trace_seconds(path: Path) -> float | None:
        if not path.exists():
            return None

        per_trace = []
        with open(path) as f:
            for line in f:
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    continue
                timing = ep.get("timing", {})
                if not timing:
                    continue
                total_elapsed = timing.get("total_elapsed")
                consistency_elapsed = timing.get("consistency_elapsed", [])
                if total_elapsed is None:
                    total_elapsed = timing.get("gold_elapsed")
                if total_elapsed is None:
                    continue
                traces = 1 + len(consistency_elapsed)
                per_trace.append(total_elapsed / traces)
        if not per_trace:
            return None
        return sum(per_trace) / len(per_trace)

    def estimate_llm_traces_remaining() -> int:
        if not config.dynamic_triangulation:
            return max(total_llm_q - total_llm_e, 0) * (1 + config.n_consistency)

        total = 0
        by_diff_q = q_stats["llm"]["by_difficulty"]
        by_diff_e = e_stats["llm"]["by_difficulty"]
        for diff, q_count in by_diff_q.items():
            remaining = max(q_count - by_diff_e.get(diff, 0), 0)
            n_cons = config.triangulation_by_difficulty.get(
                diff.upper(), config.n_consistency
            )
            total += remaining * (1 + n_cons)
        return total

    # Bottleneck analysis
    console.print()
    remaining_synth = total_synth_q - total_synth_e
    remaining_llm = total_llm_q - total_llm_e
    total_remaining = remaining_synth + remaining_llm

    if total_remaining > 0:
        console.print(f"[bold yellow]Bottleneck:[/bold yellow] Episode generation")
        console.print(
            f"  Remaining: {remaining_synth:,} synthetic + {remaining_llm:,} LLM = [bold]{total_remaining:,}[/bold] episodes"
        )

        # Estimate based on episode file timestamps
        synth_file = Path("data/episodes/episodes_synthetic.jsonl")
        llm_file = Path("data/episodes/episodes_llm.jsonl")
        synth_per_trace = load_per_trace_seconds(synth_file)
        llm_per_trace = load_per_trace_seconds(llm_file)
        llm_traces_remaining = estimate_llm_traces_remaining()

        if synth_per_trace is not None:
            synth_hours = (remaining_synth * synth_per_trace) / 3600
            console.print(f"  Estimated synthetic time: ~{synth_hours:.1f} hours")

        if llm_per_trace is not None:
            llm_hours = (llm_traces_remaining * llm_per_trace) / 3600
            console.print(f"  Estimated LLM time: ~{llm_hours:.1f} hours")

        if (
            synth_per_trace is None
            and llm_per_trace is None
            and synth_file.exists()
            and total_synth_e > 0
        ):
            # Fallback to timestamp-based throughput
            with open(synth_file) as f:
                lines = f.readlines()
            if len(lines) >= 2:
                first = json.loads(lines[0])
                last = json.loads(lines[-1])
                first_ts = datetime.fromisoformat(first.get("timestamp", "2025-01-01"))
                last_ts = datetime.fromisoformat(last.get("timestamp", "2025-01-01"))
                elapsed = (last_ts - first_ts).total_seconds()
                if elapsed > 0:
                    rate = (len(lines) - 1) / elapsed * 3600  # episodes per hour
                    if rate > 0:
                        hours_remaining = total_remaining / rate
                        console.print(f"  Estimated rate: {rate:.1f} episodes/hour")
                        console.print(
                            f"  Time to complete: ~{hours_remaining:.1f} hours"
                        )

        # Show per-type estimates
        console.print()
        console.print("[bold]Time Estimates by Type:[/bold]")
        console.print("  Synthetic: ~1 trace/question (fast, uses ground truth hash)")
        if config.dynamic_triangulation and config.triangulation_by_difficulty:
            diff_items = sorted(config.triangulation_by_difficulty.items())
            diff_summary = ", ".join(f"{k}:{v}" for k, v in diff_items)
            console.print(
                f"  LLM:       {diff_summary} (consistency traces by difficulty)"
            )
        else:
            console.print(f"  LLM:       ~{1 + config.n_consistency} traces/question")

        console.print()
        console.print("[bold]Speed Strategy:[/bold]")
        console.print(
            "  1. [green]Run synthetic first[/green] - 8x faster, has ground truth"
        )
        console.print("  2. Run LLM in parallel (separate terminal) if resources allow")
        console.print(
            "  3. Adjust triangulation_by_difficulty (config) for faster LLM validation"
        )

    console.print()

    # Recommendations
    console.print("[bold]Recommended Next Steps:[/bold]")
    if total_synth_e < total_synth_q:
        console.print(
            f"  [green]→[/green] csvagent generate episodes --template  # {remaining_synth:,} remaining"
        )
    if total_llm_e < total_llm_q:
        console.print(
            f"  [blue]→[/blue] csvagent generate episodes --llm-gen   # {remaining_llm:,} remaining"
        )

    console.print()
    return 0


# ============= Generate Commands =============


def _modes_from_flag(mode: str) -> tuple[bool, bool, bool]:
    """Convert mode string to template/procedural/llm_gen booleans."""
    template = mode in ("template", "all")
    procedural = mode in ("procedural", "all")
    llm_gen = mode in ("llm_gen", "all")
    return template, procedural, llm_gen


def cmd_generate_questions(
    mode: str,
    max_datasets: int | None,
    dry_run: bool,
    regenerate: bool = False,
):
    """Generate questions by mode."""
    template, procedural, llm_gen = _modes_from_flag(mode)

    if dry_run:
        console.print("[bold]Dry Run - Generate Questions[/bold]\n")
        if template:
            console.print("  [green]template[/green]: Will generate template questions")
            console.print(f"    Max datasets: {max_datasets or 'all'}")
            console.print(f"    Output: data/questions_synthetic/")
        if procedural:
            console.print(
                "  [magenta]procedural[/magenta]: Will generate program-based questions"
            )
            console.print(f"    Max datasets: {max_datasets or 'all'}")
            console.print("    Output: data/questions_synthetic/")
        if llm_gen:
            console.print("  [blue]llm_gen[/blue]: Will run LLM exploration")
            console.print(f"    Max datasets: {max_datasets or 'all'}")
            console.print(f"    Output: data/questions_llm/")
            if regenerate:
                console.print(f"    [yellow]Mode: Regenerate (will overwrite)[/yellow]")
        console.print("\n[dim]No changes made (dry run)[/dim]")
        return 0

    exit_code = 0

    if template:
        console.print("[bold]Generating template questions...[/bold]")
        from src.datagen.synthetic.generator import main as synth_gen_main

        result = synth_gen_main(max_datasets=max_datasets)
        if result != 0:
            exit_code = result

    if procedural:
        console.print("[bold]Generating procedural questions...[/bold]")
        import asyncio
        from src.datagen.synthetic.programs.runner import run_all as procedural_run_all

        result = asyncio.run(procedural_run_all(max_datasets=max_datasets))
        if result != 0:
            exit_code = result

    if llm_gen:
        console.print("[bold]Generating LLM questions...[/bold]")
        from src.datagen.question_gen import main as llm_gen_main

        result = llm_gen_main(max_datasets=max_datasets, regenerate=regenerate)
        if result != 0:
            exit_code = result

    return exit_code


def _show_episode_preflight(
    source: str, questions_dir: Path, episodes_file: Path
) -> tuple[int, int, set]:
    """
    Show pre-flight summary before episode generation.
    Returns: (total_questions, existing_count, existing_question_ids)
    """
    import json

    def _matches_source(question: dict, selected_source: str) -> bool:
        if selected_source == "template":
            return question.get("source") == "template"
        if selected_source == "procedural":
            return question.get("source") == "procedural"
        if selected_source == "llm_gen":
            return question.get("source") == "llm"
        return True

    # Count total questions available for selected source
    total_questions = 0
    for qf in questions_dir.glob("*/questions.json"):
        try:
            with open(qf) as f:
                data = json.load(f)
            questions = data.get("questions", data if isinstance(data, list) else [])
            questions = [q for q in questions if _matches_source(q, source)]
            total_questions += len(questions)
        except Exception:
            pass

    # Load existing episode question IDs for selected source
    existing_ids = set()
    if episodes_file.exists():
        try:
            with open(episodes_file) as f:
                for line in f:
                    ep = json.loads(line)
                    question = ep.get("question", {})
                    if not _matches_source(question, source):
                        continue
                    qid = question.get("id", "")
                    if qid:
                        existing_ids.add(qid)
        except Exception:
            pass

    existing_count = len(existing_ids)
    remaining = total_questions - existing_count
    pct = (existing_count / max(total_questions, 1)) * 100

    # Progress bar
    bar_width = 30
    filled = int(pct / 100 * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Display
    color_map = {"template": "green", "procedural": "magenta", "llm_gen": "blue"}
    color = color_map.get(source, "blue")
    console.print()
    console.print(
        Panel(
            f"[bold]Questions available:[/bold]    {total_questions:,}\n"
            f"[bold]Already processed:[/bold]      {existing_count:,} ({pct:.0f}%)\n"
            f"[bold]Remaining to generate:[/bold]  {remaining:,}\n\n"
            f"Progress: [{color}]{bar}[/{color}] {pct:.0f}%",
            title=f"Episode Generation - {source}",
            border_style=color,
        )
    )

    return total_questions, existing_count, existing_ids


def cmd_generate_episodes(
    mode: str,
    max_questions: int | None,
    dry_run: bool,
    fresh: bool = False,
):
    """Generate verified episodes via teacher triangulation."""
    template, procedural, llm_gen = _modes_from_flag(mode)

    exit_code = 0

    if template:
        questions_dir = Path("data/questions_synthetic")
        episodes_file = Path("data/episodes/episodes_synthetic.jsonl")

        total_q, existing, existing_ids = _show_episode_preflight(
            "template", questions_dir, episodes_file
        )
        remaining = total_q - existing

        if remaining == 0:
            console.print("\n[green]All template questions already processed![/green]")
        elif dry_run:
            console.print(
                f"\n[dim]Dry run: Would generate up to {remaining:,} episodes[/dim]"
            )
        else:
            if not fresh and existing > 0:
                console.print(
                    f"\n[dim]Mode: Append (skipping {existing:,} existing)[/dim]"
                )
            elif fresh:
                console.print(
                    f"\n[yellow]Mode: Fresh start (will overwrite {existing:,} existing)[/yellow]"
                )

            console.print()
            import asyncio
            from src.datagen.validate_synthetic import main as synth_ep_main

            result = asyncio.run(
                synth_ep_main(
                    questions_dir=str(questions_dir),
                    output_path=str(episodes_file),
                    max_questions=max_questions,
                    skip_existing=existing_ids if not fresh else set(),
                    source="template",
                )
            )
            if result != 0:
                exit_code = result

    if procedural:
        questions_dir = Path("data/questions_synthetic")
        episodes_file = Path("data/episodes/episodes_synthetic.jsonl")

        total_q, existing, existing_ids = _show_episode_preflight(
            "procedural", questions_dir, episodes_file
        )
        remaining = total_q - existing

        if remaining == 0:
            console.print("\n[green]All procedural questions already processed![/green]")
        elif dry_run:
            console.print(
                f"\n[dim]Dry run: Would generate up to {remaining:,} episodes[/dim]"
            )
        else:
            import asyncio
            from src.datagen.validate_synthetic import main as synth_ep_main

            result = asyncio.run(
                synth_ep_main(
                    questions_dir=str(questions_dir),
                    output_path=str(episodes_file),
                    max_questions=max_questions,
                    skip_existing=existing_ids if not fresh else set(),
                    source="procedural",
                )
            )
            if result != 0:
                exit_code = result

    if llm_gen:
        questions_dir = Path("data/questions_llm")
        episodes_file = Path("data/episodes/episodes_llm.jsonl")

        total_q, existing, existing_ids = _show_episode_preflight(
            "llm_gen", questions_dir, episodes_file
        )
        remaining = total_q - existing

        if remaining == 0:
            console.print("\n[green]All LLM questions already processed![/green]")
        elif dry_run:
            console.print(
                f"\n[dim]Dry run: Would generate up to {remaining:,} episodes[/dim]"
            )
        else:
            if not fresh and existing > 0:
                console.print(
                    f"\n[dim]Mode: Append (skipping {existing:,} existing)[/dim]"
                )
            elif fresh:
                console.print(
                    f"\n[yellow]Mode: Fresh start (will overwrite {existing:,} existing)[/yellow]"
                )

            console.print()
            import asyncio
            from src.datagen.episode_gen import main as llm_ep_main

            result = asyncio.run(
                llm_ep_main(
                    questions_dir=str(questions_dir),
                    output_path=str(episodes_file),
                    max_questions=max_questions,
                    skip_existing=existing_ids if not fresh else set(),
                )
            )
            if result != 0:
                exit_code = result

    return exit_code


# ============= Run Command =============


def cmd_run(mode: str, test: bool, dry_run: bool):
    """Run full pipeline."""
    template, procedural, llm_gen = _modes_from_flag(mode)
    if dry_run:
        console.print("[bold]Dry Run - Full Pipeline[/bold]\n")
        console.print(f"  Mode: {mode}")
        console.print(f"  Test mode: {test}")

        if template:
            console.print("\n  Stage 1a: Generate template questions")
            console.print("  Stage 2a: Generate template episodes")
        if procedural:
            console.print("  Stage 1b: Generate procedural questions")
            console.print("  Stage 2b: Generate procedural episodes")
        if llm_gen:
            console.print("  Stage 1c: Generate LLM questions")
            console.print("  Stage 2c: Generate LLM episodes")

        console.print("\n[dim]No changes made (dry run)[/dim]")
        return 0

    from src.datagen.pipeline import main as run_all_main

    return run_all_main(mode=mode, test=test)


# ============= Inspect Commands =============


def cmd_inspect_questions(
    dataset: str | None, sample: int, source: str, show_hint: bool, show_answer: bool
):
    """Preview generated questions."""
    from src.utils.inspect import inspect_questions

    inspect_questions(
        dataset=dataset,
        sample=sample,
        source=source,
        show_hint=show_hint,
        show_answer=show_answer,
    )
    return 0


def cmd_inspect_episodes(
    output: str | None, count: int, verified: bool, show_hooks: bool
):
    """Preview generated episodes."""
    from src.utils.inspect import inspect_episodes

    inspect_episodes(
        output=output,
        count=count,
        verified=verified,
        show_hooks=show_hooks,
    )
    return 0


def cmd_inspect_trace(episode_id: str, output: str | None):
    """Deep inspect a single episode trace."""
    from src.utils.inspect import inspect_trace

    inspect_trace(episode_id=episode_id, output=output)
    return 0


# ============= Validate Command =============


def cmd_validate(
    csv: str,
    questions_file: str | None,
    question: str | None,
    index: int,
    hint: str | None,
):
    """Validate a single question for debugging."""
    import asyncio
    from src.datagen.validate_question import validate_question

    return asyncio.run(
        validate_question(
            csv_path=csv,
            questions_file=questions_file,
            question_text=question,
            index=index,
            hint=hint,
        )
    )


# ============= Stats Command =============


def cmd_stats(questions: bool, episodes: bool, gaps: bool):
    """Show coverage statistics."""
    from src.utils.stats import (
        collect_questions_stats,
        collect_episodes_stats,
        show_summary,
        show_questions_detail,
        show_episodes_detail,
        show_gaps,
    )

    q_stats = collect_questions_stats()
    e_stats = collect_episodes_stats()

    show_summary(q_stats, e_stats)

    if questions or not (episodes or gaps):
        show_questions_detail(q_stats)

    if episodes or not (questions or gaps):
        show_episodes_detail(e_stats)

    if gaps:
        show_gaps(q_stats, e_stats)

    return 0


# ============= Analyze Command =============


def cmd_analyze(
    episodes_file: str, group_by: str, output_json: bool, include_all: bool
):
    """Analyze procedural question pass rates."""
    from pathlib import Path
    from src.datagen.analyze_procedural import (
        load_episodes,
        EpisodeAnalyzer,
        format_table,
    )

    try:
        episodes = load_episodes(Path(episodes_file))
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1

    analyzer = EpisodeAnalyzer(episodes)
    report = analyzer.generate_report(
        group_by=group_by,
        procedural_only=not include_all,
    )

    if output_json:
        import json

        console.print(json.dumps(report, indent=2))
    else:
        console.print(format_table(report))

    return 0


# ============= Manifest Command =============


def cmd_manifest_summary():
    """Show manifest summary - cached questions and template changes."""
    from src.datagen.manifest import DatagenManifest, compute_dataset_hash
    from src.core.config import config

    manifest = DatagenManifest()
    manifest.load()

    if not manifest.path.exists():
        console.print(
            "[dim]No manifest found. Run data generation to create one.[/dim]"
        )
        return 0

    stats = manifest.stats()
    total = stats["synthetic_total"] + stats["llm_total"]

    console.print()
    console.print(Panel.fit("[bold]Manifest Summary[/bold]", style="cyan"))
    console.print(f"[dim]Location: {manifest.path}[/dim]")
    console.print(f"[dim]Total entries: {total}[/dim]")
    console.print()

    # Overall stats
    table = Table(title="Overall Statistics", show_header=True, box=None)
    table.add_column("Type", style="cyan")
    table.add_column("Success", style="green", justify="right")
    table.add_column("Failure", style="red", justify="right")
    table.add_column("Total", justify="right")

    table.add_row(
        "Synthetic",
        str(stats["synthetic_success"]),
        str(stats["synthetic_failure"]),
        str(stats["synthetic_total"]),
    )
    table.add_row(
        "LLM",
        str(stats["llm_success"]),
        str(stats["llm_failure"]),
        str(stats["llm_total"]),
    )
    console.print(table)
    console.print()

    # Dataset summary
    ds_summary = manifest.dataset_summary()
    if ds_summary:
        ds_table = Table(title="By Dataset", show_header=True, box=None)
        ds_table.add_column("Dataset", style="cyan")
        ds_table.add_column("Synth OK", style="green", justify="right")
        ds_table.add_column("Synth Fail", style="red", justify="right")
        ds_table.add_column("LLM OK", style="green", justify="right")
        ds_table.add_column("LLM Fail", style="red", justify="right")
        ds_table.add_column("Models", style="dim")

        for ds, data in sorted(ds_summary.items()):
            models = ", ".join(data["models"][:2])  # Show first 2 models
            if len(data["models"]) > 2:
                models += f" +{len(data['models']) - 2}"
            ds_table.add_row(
                ds[:30],  # Truncate long names
                str(data["synthetic_success"]),
                str(data["synthetic_failure"]),
                str(data["llm_success"]),
                str(data["llm_failure"]),
                models or "-",
            )

        console.print(ds_table)
        console.print()

    # Template summary
    tmpl_summary = manifest.template_summary()
    if tmpl_summary:
        tmpl_table = Table(title="By Template (Synthetic)", show_header=True, box=None)
        tmpl_table.add_column("Template", style="cyan")
        tmpl_table.add_column("Success", style="green", justify="right")
        tmpl_table.add_column("Failure", style="red", justify="right")

        for tmpl, data in sorted(tmpl_summary.items(), key=lambda x: -x[1]["success"]):
            tmpl_table.add_row(
                tmpl[:40],
                str(data["success"]),
                str(data["failure"]),
            )

        console.print(tmpl_table)
        console.print()

    # Model summary
    model_summary = manifest.model_summary()
    if model_summary and any(m != "unknown" for m in model_summary.keys()):
        model_table = Table(title="By Model", show_header=True, box=None)
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Synthetic", justify="right")
        model_table.add_column("LLM", justify="right")
        model_table.add_column("Total", justify="right")

        for model, data in sorted(model_summary.items(), key=lambda x: -x[1]["total"]):
            if model == "unknown":
                continue
            model_table.add_row(
                model,
                str(data["synthetic"]),
                str(data["llm"]),
                str(data["total"]),
            )

        console.print(model_table)
        console.print()

    # Template change detection (if we have CSV sources)
    csv_sources = config.csv_sources
    if isinstance(csv_sources, str):
        csv_sources = [csv_sources]

    if csv_sources and stats["synthetic_total"] > 0:
        try:
            # Use first dataset for change detection
            first_csv = csv_sources[0]
            dataset_hash = compute_dataset_hash(first_csv)
            changes = manifest.detect_template_changes(dataset_hash)

            if changes:
                console.print("[yellow]Template Changes Detected:[/yellow]")
                for tmpl, reason in sorted(changes.items()):
                    console.print(f"  [yellow]•[/yellow] {tmpl}: {reason}")
                console.print()
        except Exception:
            pass  # Silently skip if we can't compute hash

    return 0


# ============= Profiler Command =============


def cmd_profiler(tool: str, episodes_dir: str | None, max_k: int | None):
    """Run pipeline profiler tools."""
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "pipeline_profiler.py"
    )
    if not script_path.exists():
        console.print(f"[red]Profiler script not found: {script_path}[/red]")
        return 1

    cmd = [sys.executable, str(script_path), tool]
    if episodes_dir:
        cmd.extend(["--episodes-dir", episodes_dir])
    if max_k is not None and tool == "triangulation":
        cmd.extend(["--max-k", str(max_k)])

    result = subprocess.run(cmd)
    return result.returncode


# ============= Interactive Menu =============


def interactive_menu():
    """Main interactive menu with arrow key navigation."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]csv-agent[/bold cyan] Data Generation Pipeline\n"
            "[dim]Use arrow keys to navigate, Enter to select[/dim]",
        )
    )

    while True:
        choice = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("View Status", value="status"),
                questionary.Choice("Generate Questions", value="gen_q"),
                questionary.Choice("Generate Episodes", value="gen_e"),
                questionary.Choice("Run Full Pipeline", value="run"),
                questionary.Separator(),
                questionary.Choice("Inspect Questions", value="insp_q"),
                questionary.Choice("Inspect Episodes", value="insp_e"),
                questionary.Choice("Inspect Trace", value="insp_t"),
                questionary.Separator(),
                questionary.Choice("Debug/Validate Question", value="validate"),
                questionary.Choice("Coverage Stats", value="stats"),
                questionary.Separator(),
                questionary.Choice("Exit", value="exit"),
            ],
            use_shortcuts=True,
        ).ask()

        if choice is None or choice == "exit":
            console.print("[dim]Goodbye![/dim]")
            break

        console.print()

        if choice == "status":
            cmd_status()

        elif choice == "gen_q":
            _interactive_generate_questions()

        elif choice == "gen_e":
            _interactive_generate_episodes()

        elif choice == "run":
            _interactive_run()

        elif choice == "insp_q":
            _interactive_inspect_questions()

        elif choice == "insp_e":
            cmd_inspect_episodes(output=None, count=10, verified=False, show_hooks=True)

        elif choice == "insp_t":
            _interactive_inspect_trace()

        elif choice == "validate":
            console.print(
                "[yellow]Use CLI for validate: csvagent validate --csv PATH[/yellow]"
            )

        elif choice == "stats":
            cmd_stats(questions=False, episodes=False, gaps=True)

        console.print()


def _interactive_generate_questions():
    """Interactive question generation flow."""
    source = questionary.select(
        "Question generation method:",
        choices=[
            questionary.Choice("Template", value="template"),
            questionary.Choice("Procedural", value="procedural"),
            questionary.Choice("LLM", value="llm_gen"),
            questionary.Choice("All", value="all"),
        ],
    ).ask()

    if source is None:
        return

    max_datasets_str = questionary.text(
        "Limit datasets? (Enter for all):", default=""
    ).ask()

    max_datasets = int(max_datasets_str) if max_datasets_str else None

    dry_run = questionary.confirm("Dry run (preview only)?", default=False).ask()

    # Only ask about regenerate if LLM mode is selected
    regenerate = False
    if source in ("llm_gen", "all"):
        regenerate = questionary.confirm(
            "Regenerate questions even if episodes exist?", default=False
        ).ask()

    cmd_generate_questions(
        mode=source,
        max_datasets=max_datasets,
        dry_run=dry_run,
        regenerate=regenerate,
    )


def _interactive_generate_episodes():
    """Interactive episode generation flow."""
    source = questionary.select(
        "Episode source:",
        choices=[
            questionary.Choice("Template questions", value="template"),
            questionary.Choice("Procedural questions", value="procedural"),
            questionary.Choice("LLM questions", value="llm_gen"),
            questionary.Choice("All", value="all"),
        ],
    ).ask()

    if source is None:
        return

    max_q_str = questionary.text(
        "Max questions per dataset? (Enter for all):", default=""
    ).ask()

    max_questions = int(max_q_str) if max_q_str else None

    dry_run = questionary.confirm("Dry run (preview only)?", default=False).ask()

    cmd_generate_episodes(
        mode=source, max_questions=max_questions, dry_run=dry_run
    )


def _interactive_run():
    """Interactive full pipeline flow."""
    mode = questionary.select(
        "Pipeline mode:",
        choices=[
            questionary.Choice("All", value="all"),
            questionary.Choice("Template only", value="template"),
            questionary.Choice("Procedural only", value="procedural"),
            questionary.Choice("LLM only", value="llm_gen"),
        ],
    ).ask()

    if mode is None:
        return

    test = questionary.confirm(
        "Test mode (1 dataset, 1 question)?", default=False
    ).ask()

    dry_run = questionary.confirm("Dry run (preview only)?", default=False).ask()

    cmd_run(mode=mode, test=test, dry_run=dry_run)


def _interactive_inspect_questions():
    """Interactive question inspection."""
    source = questionary.select(
        "Question source:", choices=["template", "procedural", "llm_gen", "all"]
    ).ask()

    if source is None:
        return

    show_hint = questionary.confirm("Show hints?", default=False).ask()
    show_answer = questionary.confirm("Show answers?", default=False).ask()

    cmd_inspect_questions(
        dataset=None,
        sample=10,
        source=source,
        show_hint=show_hint,
        show_answer=show_answer,
    )


def _interactive_inspect_trace():
    """Interactive trace inspection."""
    episode_id = questionary.text(
        "Episode ID (prefix):",
    ).ask()

    if episode_id:
        cmd_inspect_trace(episode_id=episode_id, output=None)


# ============= Argument Parsing =============


def build_parser():
    parser = argparse.ArgumentParser(
        prog="csvagent",
        description="csv-agent data generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  csvagent                          # Interactive menu
  csvagent status                   # View data inventory
  csvagent generate questions --template
  csvagent generate episodes --llm-gen --dry-run
  csvagent run --all --test
  csvagent inspect questions --source template
  csvagent inspect trace abc123
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # status
    subparsers.add_parser("status", help="View data inventory")

    # progress
    subparsers.add_parser("progress", help="Detailed progress with estimates")

    # generate
    gen_parser = subparsers.add_parser(
        "generate", help="Generate questions or episodes"
    )
    gen_sub = gen_parser.add_subparsers(dest="gen_type", required=True)

    q_parser = gen_sub.add_parser("questions", help="Generate questions")
    q_mode = q_parser.add_mutually_exclusive_group(required=True)
    q_mode.add_argument("--template", action="store_const", dest="mode", const="template")
    q_mode.add_argument("--procedural", action="store_const", dest="mode", const="procedural")
    q_mode.add_argument("--llm-gen", action="store_const", dest="mode", const="llm_gen")
    q_mode.add_argument("--all", action="store_const", dest="mode", const="all")
    q_parser.add_argument("--max-datasets", type=int, help="Limit datasets")
    q_parser.add_argument("--dry-run", action="store_true", help="Preview only")
    q_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate questions even if episodes exist (LLM only)",
    )

    e_parser = gen_sub.add_parser("episodes", help="Generate episodes")
    e_mode = e_parser.add_mutually_exclusive_group(required=True)
    e_mode.add_argument("--template", action="store_const", dest="mode", const="template")
    e_mode.add_argument("--procedural", action="store_const", dest="mode", const="procedural")
    e_mode.add_argument("--llm-gen", action="store_const", dest="mode", const="llm_gen")
    e_mode.add_argument("--all", action="store_const", dest="mode", const="all")
    e_parser.add_argument("--max-questions", type=int, help="Max per dataset")
    e_parser.add_argument("--dry-run", action="store_true", help="Preview only")
    e_parser.add_argument(
        "--fresh", action="store_true", help="Start fresh (overwrite existing)"
    )

    # run
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_mode = run_parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument("--template", action="store_const", dest="mode", const="template")
    run_mode.add_argument("--procedural", action="store_const", dest="mode", const="procedural")
    run_mode.add_argument("--llm-gen", action="store_const", dest="mode", const="llm_gen")
    run_mode.add_argument("--all", action="store_const", dest="mode", const="all")
    run_parser.add_argument("--test", action="store_true", help="Quick test mode")
    run_parser.add_argument("--dry-run", action="store_true", help="Preview only")

    # inspect
    insp_parser = subparsers.add_parser("inspect", help="Inspect data")
    insp_sub = insp_parser.add_subparsers(dest="insp_type", required=True)

    iq_parser = insp_sub.add_parser("questions", help="Preview questions")
    iq_parser.add_argument("--dataset", help="Specific dataset")
    iq_parser.add_argument("--sample", type=int, default=5, help="Number to show")
    iq_parser.add_argument(
        "--source",
        choices=["template", "procedural", "llm_gen", "all"],
        required=True,
    )
    iq_parser.add_argument("--show-hint", action="store_true")
    iq_parser.add_argument("--show-answer", action="store_true")

    ie_parser = insp_sub.add_parser("episodes", help="Preview episodes")
    ie_parser.add_argument("--output", help="Episodes file path")
    ie_parser.add_argument("--count", type=int, default=5)
    ie_parser.add_argument("--verified", action="store_true")
    ie_parser.add_argument("--show-hooks", action="store_true")

    it_parser = insp_sub.add_parser("trace", help="Deep inspect trace")
    it_parser.add_argument("episode_id", help="Episode ID prefix")
    it_parser.add_argument("--output", help="Episodes file path")

    # validate
    val_parser = subparsers.add_parser("validate", help="Debug single question")
    val_parser.add_argument("--csv", required=True, help="CSV file path")
    val_parser.add_argument("--questions-file", help="Questions JSON file")
    val_parser.add_argument("--question", help="Question text directly")
    val_parser.add_argument("--index", type=int, default=0, help="Question index")
    val_parser.add_argument("--hint", help="Hint text")

    # stats
    stats_parser = subparsers.add_parser("stats", help="Coverage report")
    stats_parser.add_argument("--questions", action="store_true")
    stats_parser.add_argument("--episodes", action="store_true")
    stats_parser.add_argument("--gaps", action="store_true")

    # profiler
    prof_parser = subparsers.add_parser("profiler", help="Pipeline profiler toolbox")
    prof_parser.add_argument(
        "tool",
        choices=[
            "containers",
            "episodes",
            "hooks",
            "timing",
            "taxonomy",
            "perf",
            "triangulation",
            "silent",
            "all",
        ],
        help="Profiler tool to run",
    )
    prof_parser.add_argument("--episodes-dir", help="Episodes directory")
    prof_parser.add_argument(
        "--max-k", type=int, default=None, help="Max k for triangulation profiling"
    )

    # manifest
    subparsers.add_parser("manifest", help="Show manifest summary and template changes")

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze procedural question pass rates"
    )
    analyze_sub = analyze_parser.add_subparsers(dest="analyze_type", required=True)

    proc_parser = analyze_sub.add_parser(
        "procedural", help="Analyze procedural questions"
    )
    proc_parser.add_argument(
        "--episodes", required=True, help="Path to episodes JSONL file"
    )
    proc_parser.add_argument(
        "--group-by",
        choices=["prefix", "operator", "both"],
        default="prefix",
        help="How to group episodes (default: prefix)",
    )
    proc_parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of table"
    )
    proc_parser.add_argument(
        "--all",
        action="store_true",
        dest="include_all",
        help="Include non-procedural questions too",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # No command = interactive mode
    if args.command is None:
        try:
            interactive_menu()
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted[/dim]")
        return 0

    # Dispatch to commands
    if args.command == "status":
        return cmd_status()

    elif args.command == "progress":
        return cmd_progress()

    elif args.command == "generate":
        if args.gen_type == "questions":
            return cmd_generate_questions(
                mode=args.mode,
                max_datasets=args.max_datasets,
                dry_run=args.dry_run,
                regenerate=args.regenerate,
            )
        elif args.gen_type == "episodes":
            return cmd_generate_episodes(
                mode=args.mode,
                max_questions=args.max_questions,
                dry_run=args.dry_run,
                fresh=args.fresh,
            )

    elif args.command == "run":
        return cmd_run(
            mode=args.mode,
            test=args.test,
            dry_run=args.dry_run,
        )

    elif args.command == "inspect":
        if args.insp_type == "questions":
            return cmd_inspect_questions(
                dataset=args.dataset,
                sample=args.sample,
                source=args.source,
                show_hint=args.show_hint,
                show_answer=args.show_answer,
            )
        elif args.insp_type == "episodes":
            return cmd_inspect_episodes(
                output=args.output,
                count=args.count,
                verified=args.verified,
                show_hooks=args.show_hooks,
            )
        elif args.insp_type == "trace":
            return cmd_inspect_trace(
                episode_id=args.episode_id,
                output=args.output,
            )

    elif args.command == "validate":
        return cmd_validate(
            csv=args.csv,
            questions_file=args.questions_file,
            question=args.question,
            index=args.index,
            hint=args.hint,
        )

    elif args.command == "stats":
        return cmd_stats(
            questions=args.questions,
            episodes=args.episodes,
            gaps=args.gaps,
        )
    elif args.command == "profiler":
        return cmd_profiler(
            tool=args.tool,
            episodes_dir=args.episodes_dir,
            max_k=args.max_k,
        )

    elif args.command == "manifest":
        return cmd_manifest_summary()

    elif args.command == "analyze":
        if args.analyze_type == "procedural":
            return cmd_analyze(
                episodes_file=args.episodes,
                group_by=args.group_by,
                output_json=args.json,
                include_all=args.include_all,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
