"""
Unified CLI for csv-agent data generation pipeline.

Usage:
    csvagent              # Interactive menu (default)
    csvagent status       # View data inventory
    csvagent generate questions --synth|--llm
    csvagent generate episodes --synth|--llm
    csvagent run          # Full pipeline
    csvagent inspect      # Preview data
    csvagent validate     # Debug single question
    csvagent stats        # Coverage report
"""

import argparse
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
    from src.utils.stats import collect_questions_stats, collect_episodes_stats, collect_datasets

    datasets = collect_datasets()
    q_stats = collect_questions_stats()
    e_stats = collect_episodes_stats()

    # Compact status display
    console.print()
    console.print(Panel.fit(
        "[bold]csv-agent[/bold] Data Generation Pipeline",
        style="cyan"
    ))

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
        f"[blue]llm[/blue] {llm_q:,} ({llm_q_datasets} datasets)"
    )

    # Episodes
    synth_e = e_stats["synthetic"]["total"]
    synth_e_verified = e_stats["synthetic"]["verified"]
    llm_e = e_stats["llm"]["total"]
    llm_e_verified = e_stats["llm"]["verified"]

    synth_pct = f"{synth_e_verified/max(synth_e,1)*100:.0f}%" if synth_e else "0%"
    llm_pct = f"{llm_e_verified/max(llm_e,1)*100:.0f}%" if llm_e else "0%"

    table.add_row(
        "Episodes",
        f"[green]synthetic[/green] {synth_e_verified}/{synth_e} verified ({synth_pct}) | "
        f"[blue]llm[/blue] {llm_e_verified}/{llm_e} verified ({llm_pct})"
    )

    console.print(table)
    console.print()

    # Suggest next action
    if synth_q == 0:
        console.print("[dim]Next:[/dim] csvagent generate questions --synth")
    elif synth_e == 0:
        console.print("[dim]Next:[/dim] csvagent generate episodes --synth")
    elif llm_q == 0:
        console.print("[dim]Next:[/dim] csvagent generate questions --llm")
    elif llm_e == 0:
        console.print("[dim]Next:[/dim] csvagent generate episodes --llm")
    else:
        console.print("[dim]Pipeline complete! Run 'csvagent stats' for details.[/dim]")

    console.print()
    return 0


# ============= Generate Commands =============


def cmd_generate_questions(synth: bool, llm: bool, max_datasets: int | None, dry_run: bool):
    """Generate questions using synthetic templates or LLM exploration."""
    if not synth and not llm:
        console.print("[red]Specify --synth or --llm (or both)[/red]")
        return 1

    if dry_run:
        console.print("[bold]Dry Run - Generate Questions[/bold]\n")
        if synth:
            console.print(f"  [green]synthetic[/green]: Will generate template-based questions")
            console.print(f"    Max datasets: {max_datasets or 'all'}")
            console.print(f"    Output: data/questions_synthetic/")
        if llm:
            console.print(f"  [blue]llm[/blue]: Will run LLM exploration")
            console.print(f"    Max datasets: {max_datasets or 'all'}")
            console.print(f"    Output: data/questions/")
        console.print("\n[dim]No changes made (dry run)[/dim]")
        return 0

    exit_code = 0

    if synth:
        console.print("[bold]Generating synthetic questions...[/bold]")
        from src.datagen.synthetic.generator import main as synth_gen_main
        result = synth_gen_main(max_datasets=max_datasets)
        if result != 0:
            exit_code = result

    if llm:
        console.print("[bold]Generating LLM questions...[/bold]")
        from src.datagen.question_gen import main as llm_gen_main
        result = llm_gen_main(max_datasets=max_datasets)
        if result != 0:
            exit_code = result

    return exit_code


def cmd_generate_episodes(synth: bool, llm: bool, max_questions: int | None, dry_run: bool):
    """Generate verified episodes via teacher triangulation."""
    if not synth and not llm:
        console.print("[red]Specify --synth or --llm (or both)[/red]")
        return 1

    if dry_run:
        console.print("[bold]Dry Run - Generate Episodes[/bold]\n")
        if synth:
            console.print(f"  [green]synthetic[/green]: Will validate synthetic questions")
            console.print(f"    Max questions: {max_questions or 'all'}")
            console.print(f"    Output: data/episodes/episodes_synthetic.jsonl")
        if llm:
            console.print(f"  [blue]llm[/blue]: Will triangulate LLM questions")
            console.print(f"    Max questions: {max_questions or 'all'}")
            console.print(f"    Output: data/episodes/episodes_llm.jsonl")
        console.print("\n[dim]No changes made (dry run)[/dim]")
        return 0

    exit_code = 0

    if synth:
        console.print("[bold]Generating synthetic episodes...[/bold]")
        from src.datagen.synthetic_episodes import main as synth_ep_main
        result = synth_ep_main(
            questions_dir="data/questions_synthetic",
            output="data/episodes/episodes_synthetic.jsonl",
            max_questions=max_questions,
        )
        if result != 0:
            exit_code = result

    if llm:
        console.print("[bold]Generating LLM episodes...[/bold]")
        from src.datagen.episode_gen import main as llm_ep_main
        result = llm_ep_main(
            questions_dir="data/questions",
            output="data/episodes/episodes_llm.jsonl",
            max_questions=max_questions,
        )
        if result != 0:
            exit_code = result

    return exit_code


# ============= Run Command =============


def cmd_run(mode: str, triangulate: bool, test: bool, dry_run: bool):
    """Run full pipeline."""
    if dry_run:
        console.print("[bold]Dry Run - Full Pipeline[/bold]\n")
        console.print(f"  Mode: {mode}")
        console.print(f"  Triangulate only: {triangulate}")
        console.print(f"  Test mode: {test}")

        if mode in ("synth", "both") and not triangulate:
            console.print("\n  Stage 1a: Generate synthetic questions")
        if mode in ("synth", "both"):
            console.print("  Stage 2a: Generate synthetic episodes")
        if mode in ("llm", "both") and not triangulate:
            console.print("  Stage 1b: Generate LLM questions")
        if mode in ("llm", "both"):
            console.print("  Stage 2b: Generate LLM episodes")

        console.print("\n[dim]No changes made (dry run)[/dim]")
        return 0

    from src.datagen.run_all import main as run_all_main
    return run_all_main(mode=mode, triangulate=triangulate, test=test)


# ============= Inspect Commands =============


def cmd_inspect_questions(dataset: str | None, sample: int, source: str, show_hint: bool, show_answer: bool):
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


def cmd_inspect_episodes(output: str | None, count: int, verified: bool, show_hooks: bool):
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


def cmd_validate(csv: str, questions_file: str | None, question: str | None, index: int, hint: str | None):
    """Validate a single question for debugging."""
    import asyncio
    from src.datagen.validate_question import validate_question

    return asyncio.run(validate_question(
        csv_path=csv,
        questions_file=questions_file,
        question_text=question,
        index=index,
        hint=hint,
    ))


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


# ============= Interactive Menu =============


def interactive_menu():
    """Main interactive menu with arrow key navigation."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]csv-agent[/bold cyan] Data Generation Pipeline\n"
        "[dim]Use arrow keys to navigate, Enter to select[/dim]",
    ))

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
            console.print("[yellow]Use CLI for validate: csvagent validate --csv PATH[/yellow]")

        elif choice == "stats":
            cmd_stats(questions=False, episodes=False, gaps=True)

        console.print()


def _interactive_generate_questions():
    """Interactive question generation flow."""
    source = questionary.select(
        "Question generation method:",
        choices=[
            questionary.Choice("Synthetic (fast, deterministic)", value="synth"),
            questionary.Choice("LLM-based (slow, exploratory)", value="llm"),
            questionary.Choice("Both", value="both"),
        ]
    ).ask()

    if source is None:
        return

    max_datasets_str = questionary.text(
        "Limit datasets? (Enter for all):",
        default=""
    ).ask()

    max_datasets = int(max_datasets_str) if max_datasets_str else None

    dry_run = questionary.confirm(
        "Dry run (preview only)?",
        default=False
    ).ask()

    synth = source in ("synth", "both")
    llm = source in ("llm", "both")

    cmd_generate_questions(synth=synth, llm=llm, max_datasets=max_datasets, dry_run=dry_run)


def _interactive_generate_episodes():
    """Interactive episode generation flow."""
    source = questionary.select(
        "Episode source:",
        choices=[
            questionary.Choice("Synthetic questions", value="synth"),
            questionary.Choice("LLM questions", value="llm"),
            questionary.Choice("Both", value="both"),
        ]
    ).ask()

    if source is None:
        return

    max_q_str = questionary.text(
        "Max questions per dataset? (Enter for all):",
        default=""
    ).ask()

    max_questions = int(max_q_str) if max_q_str else None

    dry_run = questionary.confirm(
        "Dry run (preview only)?",
        default=False
    ).ask()

    synth = source in ("synth", "both")
    llm = source in ("llm", "both")

    cmd_generate_episodes(synth=synth, llm=llm, max_questions=max_questions, dry_run=dry_run)


def _interactive_run():
    """Interactive full pipeline flow."""
    mode = questionary.select(
        "Pipeline mode:",
        choices=[
            questionary.Choice("Both (synthetic + LLM)", value="both"),
            questionary.Choice("Synthetic only", value="synth"),
            questionary.Choice("LLM only", value="llm"),
        ]
    ).ask()

    if mode is None:
        return

    triangulate = questionary.confirm(
        "Skip question generation (episodes only)?",
        default=False
    ).ask()

    test = questionary.confirm(
        "Test mode (1 dataset, 1 question)?",
        default=False
    ).ask()

    dry_run = questionary.confirm(
        "Dry run (preview only)?",
        default=False
    ).ask()

    cmd_run(mode=mode, triangulate=triangulate, test=test, dry_run=dry_run)


def _interactive_inspect_questions():
    """Interactive question inspection."""
    source = questionary.select(
        "Question source:",
        choices=["synthetic", "llm"]
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
  csvagent generate questions --synth
  csvagent generate episodes --llm --dry-run
  csvagent run --both --test
  csvagent inspect questions --source synthetic
  csvagent inspect trace abc123
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # status
    subparsers.add_parser("status", help="View data inventory")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate questions or episodes")
    gen_sub = gen_parser.add_subparsers(dest="gen_type", required=True)

    q_parser = gen_sub.add_parser("questions", help="Generate questions")
    q_parser.add_argument("--synth", action="store_true", help="Synthetic (template-based)")
    q_parser.add_argument("--llm", action="store_true", help="LLM exploration")
    q_parser.add_argument("--max-datasets", type=int, help="Limit datasets")
    q_parser.add_argument("--dry-run", action="store_true", help="Preview only")

    e_parser = gen_sub.add_parser("episodes", help="Generate episodes")
    e_parser.add_argument("--synth", action="store_true", help="From synthetic questions")
    e_parser.add_argument("--llm", action="store_true", help="From LLM questions")
    e_parser.add_argument("--max-questions", type=int, help="Max per dataset")
    e_parser.add_argument("--dry-run", action="store_true", help="Preview only")

    # run
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_mode = run_parser.add_mutually_exclusive_group()
    run_mode.add_argument("--synth", action="store_const", dest="mode", const="synth")
    run_mode.add_argument("--llm", action="store_const", dest="mode", const="llm")
    run_mode.add_argument("--both", action="store_const", dest="mode", const="both")
    run_parser.set_defaults(mode="both")
    run_parser.add_argument("--triangulate", action="store_true", help="Episodes only")
    run_parser.add_argument("--test", action="store_true", help="Quick test mode")
    run_parser.add_argument("--dry-run", action="store_true", help="Preview only")

    # inspect
    insp_parser = subparsers.add_parser("inspect", help="Inspect data")
    insp_sub = insp_parser.add_subparsers(dest="insp_type", required=True)

    iq_parser = insp_sub.add_parser("questions", help="Preview questions")
    iq_parser.add_argument("--dataset", help="Specific dataset")
    iq_parser.add_argument("--sample", type=int, default=5, help="Number to show")
    iq_parser.add_argument("--source", choices=["synthetic", "llm"], default="synthetic")
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

    elif args.command == "generate":
        if args.gen_type == "questions":
            return cmd_generate_questions(
                synth=args.synth,
                llm=args.llm,
                max_datasets=args.max_datasets,
                dry_run=args.dry_run,
            )
        elif args.gen_type == "episodes":
            return cmd_generate_episodes(
                synth=args.synth,
                llm=args.llm,
                max_questions=args.max_questions,
                dry_run=args.dry_run,
            )

    elif args.command == "run":
        return cmd_run(
            mode=args.mode,
            triangulate=args.triangulate,
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
