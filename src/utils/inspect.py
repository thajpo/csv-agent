"""
Inspect and preview generated outputs.

Usage (via CLI):
    csvagent inspect questions                    # Preview questions
    csvagent inspect questions --dataset titanic  # Specific dataset
    csvagent inspect episodes                     # Preview episodes
    csvagent inspect episodes --verified          # Only verified
    csvagent inspect trace EPISODE_ID             # Deep inspect trace
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax


console = Console()


def inspect_questions(
    dataset: str | None = None,
    sample: int = 5,
    source: str = "template",
    show_hint: bool = False,
    show_answer: bool = False,
):
    """Preview generated questions."""
    if source in ("template", "procedural"):
        questions_dirs = [Path("data/questions_synthetic")]
    elif source == "llm_gen":
        questions_dirs = [Path("data/questions_llm")]
    else:  # all
        questions_dirs = [Path("data/questions_synthetic"), Path("data/questions_llm")]

    files = []
    for questions_dir in questions_dirs:
        if not questions_dir.exists():
            continue
        if dataset:
            questions_file = questions_dir / dataset / "questions.json"
            if questions_file.exists():
                files.append(questions_file)
        else:
            files.extend(questions_dir.glob("*/questions.json"))

    source_filters = {
        "template": lambda q: q.get("source") == "template",
        "procedural": lambda q: q.get("source") == "procedural",
        "llm_gen": lambda q: q.get("source") == "llm_gen",
        "all": lambda q: True,
    }

    if not files:
        console.print("[yellow]No questions found for selected source[/yellow]")
        return

    total_questions = 0
    sample_questions: list[dict] = []
    dataset_name = files[0].parent.name
    for qf in files:
        with open(qf) as f:
            data = json.load(f)
        questions = data.get("questions", data if isinstance(data, list) else [])
        questions = [q for q in questions if source_filters[source](q)]
        total_questions += len(questions)
        if not sample_questions and questions:
            sample_questions = questions[:sample]
            dataset_name = qf.parent.name

    console.print(
        f"\n[bold]Found {total_questions} questions across {len(files)} datasets[/bold]\n"
    )

    table = Table(title=f"Questions from {dataset_name}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Difficulty", width=10)
    table.add_column("Question", max_width=60)
    if show_hint:
        table.add_column("Hint", max_width=40, style="dim")
    if show_answer:
        table.add_column("Answer", max_width=20)

    for i, q in enumerate(sample_questions, 1):
        row = [
            str(i),
            str(q.get("difficulty", "?")),
            (q.get("question_text") or q.get("question_mechanical") or "")[:60],
        ]
        if show_hint:
            row.append((q.get("hint") or "")[:40])
        if show_answer:
            ans = q.get("ground_truth")
            row.append(str(ans)[:20] if ans is not None else "?")
        table.add_row(*row)

    console.print(table)

    # Show one full question
    if sample_questions:
        q = sample_questions[0]
        console.print(Panel(
            f"[bold]{q.get('question_text') or q.get('question_mechanical') or ''}[/bold]\n\n"
            f"[dim]Template:[/dim] {q.get('template_name', 'N/A')}\n"
            f"[dim]Difficulty:[/dim] {q.get('difficulty', 'N/A')}\n"
            f"[dim]Steps:[/dim] {q.get('n_steps', 'N/A')}\n"
            f"[dim]Hint:[/dim] {q.get('hint', 'None')[:100]}",
            title="Full Question #1",
            expand=False
        ))


def inspect_episodes(
    output: str | None = None,
    count: int = 5,
    verified: bool = False,
    show_hooks: bool = False,
):
    """Preview generated episodes."""
    # Find episodes file
    if output:
        episodes_file = Path(output)
    else:
        # Try synthetic first, then LLM
        candidates = [
            Path("data/episodes/episodes_synthetic.jsonl"),
            Path("data/episodes/episodes_llm.jsonl"),
        ]
        episodes_file = None
        for c in candidates:
            if c.exists():
                episodes_file = c
                break

    if not episodes_file or not episodes_file.exists():
        console.print("[red]No episodes file found[/red]")
        console.print("Try: --output data/episodes/episodes_synthetic.jsonl")
        return

    # Count total and load sample
    total = 0
    verified_count = 0
    episodes = []

    with open(episodes_file) as f:
        for line in f:
            ep = json.loads(line)
            total += 1
            if ep.get("verified"):
                verified_count += 1

            # Collect sample
            if verified and not ep.get("verified"):
                continue
            if len(episodes) < count:
                episodes.append(ep)

    console.print(f"\n[bold]{episodes_file.name}[/bold]: {total} episodes ({verified_count} verified)\n")

    if not episodes:
        console.print("[yellow]No episodes match filter[/yellow]")
        return

    table = Table(title=f"Episodes (showing {len(episodes)})")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("V", width=2)
    table.add_column("Question", max_width=50)
    table.add_column("Answer", max_width=25)
    table.add_column("Turns", width=5)
    if show_hooks:
        table.add_column("Hooks", width=5)

    for ep in episodes:
        # Extract answer
        gold_trace = ep.get("gold_trace", {})
        answer = gold_trace.get("final_answer", "?")
        if isinstance(answer, (dict, list)):
            answer = json.dumps(answer)[:25]
        else:
            answer = str(answer)[:25]

        # Count turns
        turns = len(gold_trace.get("turns", []))

        # Count hooks
        hooks = sum(
            len(t.get("execution", {}).get("hooks", []))
            for t in gold_trace.get("turns", [])
        )

        question = ep.get("question", {})
        q_text = question.get("question_text", "?")[:50]

        row = [
            ep.get("episode_id", "?")[:10],
            "[green]✓[/green]" if ep.get("verified") else "[red]✗[/red]",
            q_text,
            answer,
            str(turns),
        ]
        if show_hooks:
            row.append(str(hooks))

        table.add_row(*row)

    console.print(table)


def inspect_trace(episode_id: str, output: str | None = None):
    """Deep inspect a single episode trace."""
    # Find episodes file
    if output:
        episodes_file = Path(output)
    else:
        candidates = [
            Path("data/episodes/episodes_synthetic.jsonl"),
            Path("data/episodes/episodes_llm.jsonl"),
        ]
        episodes_file = None
        for c in candidates:
            if c.exists():
                episodes_file = c
                break

    if not episodes_file or not episodes_file.exists():
        console.print("[red]No episodes file found[/red]")
        return

    # Find episode by ID
    episode = None
    with open(episodes_file) as f:
        for line in f:
            ep = json.loads(line)
            if ep.get("episode_id", "").startswith(episode_id):
                episode = ep
                break

    if not episode:
        console.print(f"[red]Episode not found: {episode_id}[/red]")
        return

    # Display episode details
    question = episode.get("question", {})
    gold_trace = episode.get("gold_trace", {})

    console.print(Panel(
        f"[bold]{question.get('question_text', '')}[/bold]\n\n"
        f"[dim]Verified:[/dim] {'Yes' if episode.get('verified') else 'No'}\n"
        f"[dim]Difficulty:[/dim] {question.get('difficulty', 'N/A')}\n"
        f"[dim]Template:[/dim] {question.get('template_name', 'N/A')}\n"
        f"[dim]Hint:[/dim] {question.get('hint', 'None')}",
        title=f"Episode {episode.get('episode_id', '')[:10]}",
    ))

    # Show each turn
    for i, turn in enumerate(gold_trace.get("turns", []), 1):
        code = turn.get("code", "")
        execution = turn.get("execution", {})

        console.print(f"\n[bold]Turn {i}[/bold]")

        if turn.get("reasoning"):
            console.print(f"[dim]Reasoning:[/dim] {turn['reasoning'][:200]}...")

        if code:
            console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

        if execution.get("stdout"):
            console.print(Panel(execution["stdout"][:500], title="stdout", style="green"))

        if execution.get("stderr"):
            console.print(Panel(execution["stderr"][:300], title="stderr", style="red"))

        hooks = execution.get("hooks", [])
        if hooks:
            console.print(f"[dim]Hooks ({len(hooks)}):[/dim]")
            for h in hooks[:3]:
                console.print(f"  {h.get('variable_name', '?')}: {str(h.get('value', '?'))[:50]}")

    # Final answer
    console.print(Panel(
        f"[bold]{gold_trace.get('final_answer', 'None')}[/bold]\n"
        f"[dim]Hash:[/dim] {gold_trace.get('final_answer_hash', 'N/A')[:20]}",
        title="Final Answer",
        style="green" if episode.get("verified") else "red"
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Inspect generated outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m src.utils.inspect questions
  uv run python -m src.utils.inspect questions --dataset titanic --show-hint
  uv run python -m src.utils.inspect episodes --verified --count 10
  uv run python -m src.utils.inspect trace abc123
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Questions subcommand
    q_parser = subparsers.add_parser("questions", help="Inspect generated questions")
    q_parser.add_argument("--dataset", help="Specific dataset name")
    q_parser.add_argument("--sample", type=int, default=5, help="Number to show")
    q_parser.add_argument(
        "--source",
        choices=["template", "procedural", "llm_gen", "all"],
        required=True,
    )
    q_parser.add_argument("--show-hint", action="store_true", help="Show hints")
    q_parser.add_argument("--show-answer", action="store_true", help="Show ground truth")

    # Episodes subcommand
    e_parser = subparsers.add_parser("episodes", help="Inspect generated episodes")
    e_parser.add_argument("--output", help="Episodes JSONL file")
    e_parser.add_argument("--count", type=int, default=5, help="Number to show")
    e_parser.add_argument("--verified", action="store_true", help="Only verified")
    e_parser.add_argument("--show-hooks", action="store_true", help="Show hook counts")

    # Trace subcommand
    t_parser = subparsers.add_parser("trace", help="Deep inspect a single trace")
    t_parser.add_argument("episode_id", help="Episode ID (prefix match)")
    t_parser.add_argument("--output", help="Episodes JSONL file")

    args = parser.parse_args()

    if args.command == "questions":
        inspect_questions(
            dataset=args.dataset,
            sample=args.sample,
            source=args.source,
            show_hint=args.show_hint,
            show_answer=args.show_answer,
        )
    elif args.command == "episodes":
        inspect_episodes(
            output=args.output,
            count=args.count,
            verified=args.verified,
            show_hooks=args.show_hooks,
        )
    elif args.command == "trace":
        inspect_trace(episode_id=args.episode_id, output=args.output)
