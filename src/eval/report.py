"""
Report generation for evaluation results.

Supports both markdown and JSON output formats.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from src.eval.metrics import EvalResult, EvalMetrics


def generate_report(
    metrics: EvalMetrics,
    results: list[EvalResult],
    output_path: str,
    format: Literal["markdown", "json"] = "markdown",
    model: str = "",
    episodes_path: str = "",
) -> None:
    """
    Generate evaluation report in markdown or JSON format.

    Args:
        metrics: Aggregate metrics
        results: Individual episode results
        output_path: Path to write report
        format: Output format ('markdown' or 'json')
        model: Model identifier (for report header)
        episodes_path: Path to episodes file (for report header)
    """
    if format == "markdown":
        _generate_markdown_report(
            metrics, results, output_path, model, episodes_path
        )
    elif format == "json":
        _generate_json_report(metrics, results, output_path, model, episodes_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def _generate_markdown_report(
    metrics: EvalMetrics,
    results: list[EvalResult],
    output_path: str,
    model: str,
    episodes_path: str,
) -> None:
    """Generate markdown report."""

    # Build report sections
    lines = []

    # Header
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model**: `{model}`")
    lines.append(f"**Episodes**: `{episodes_path}`")
    lines.append(f"**Total Episodes**: {metrics.total_episodes}")
    lines.append("")

    # Overall Metrics
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(f"- **Accuracy**: {metrics.accuracy:.1%} ({metrics.total_correct}/{metrics.total_episodes})")
    lines.append(f"- **Execution Success Rate**: {metrics.execution_success_rate:.1%} ({metrics.total_executed}/{metrics.total_episodes})")
    lines.append(f"- **Average Turns**: {metrics.avg_turns:.1f}")
    lines.append(f"- **Average Time**: {metrics.avg_elapsed_seconds:.1f}s")
    lines.append("")

    # Accuracy by Difficulty
    if metrics.accuracy_by_difficulty:
        lines.append("## Accuracy by Difficulty")
        lines.append("")
        for difficulty in sorted(metrics.accuracy_by_difficulty.keys()):
            accuracy = metrics.accuracy_by_difficulty[difficulty]
            total = metrics.episodes_by_difficulty[difficulty]
            correct = metrics.correct_by_difficulty.get(difficulty, 0)
            lines.append(f"- **{difficulty}**: {accuracy:.1%} ({correct}/{total})")
        lines.append("")

    # Per-Episode Results
    lines.append("## Per-Episode Results")
    lines.append("")
    lines.append("| Episode ID | Question | Difficulty | Correct | Executed | Turns | Time (s) |")
    lines.append("|------------|----------|------------|---------|----------|-------|----------|")

    for result in results:
        episode_id = result.episode_id
        # Truncate question for table display
        question = result.question_text[:50] + "..." if len(result.question_text) > 50 else result.question_text
        difficulty = result.difficulty or "N/A"
        correct = "✓" if result.final_answer_correct else "✗"
        executed = "✓" if result.execution_success else "✗"
        turns = result.num_turns
        elapsed = f"{result.elapsed_seconds:.1f}"

        lines.append(f"| {episode_id} | {question} | {difficulty} | {correct} | {executed} | {turns} | {elapsed} |")

    lines.append("")

    # Failed Episodes (if any)
    failed_results = [r for r in results if not r.execution_success]
    if failed_results:
        lines.append("## Failed Episodes")
        lines.append("")
        lines.append("Episodes that failed to execute or submit an answer:")
        lines.append("")
        for result in failed_results:
            lines.append(f"### {result.episode_id}")
            lines.append(f"- **Question**: {result.question_text}")
            if result.error_message:
                lines.append(f"- **Error**: {result.error_message}")
            lines.append("")

    # Incorrect Episodes (executed but wrong answer)
    incorrect_results = [
        r for r in results
        if r.execution_success and not r.final_answer_correct
    ]
    if incorrect_results:
        lines.append("## Incorrect Episodes")
        lines.append("")
        lines.append("Episodes that executed successfully but produced wrong answers:")
        lines.append("")
        for result in incorrect_results:
            lines.append(f"### {result.episode_id}")
            lines.append(f"- **Question**: {result.question_text}")
            lines.append(f"- **Expected**: `{result.expected_answer}`")
            lines.append(f"- **Actual**: `{result.actual_answer}`")
            lines.append("")

    # Write to file
    output = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(output)


def _generate_json_report(
    metrics: EvalMetrics,
    results: list[EvalResult],
    output_path: str,
    model: str,
    episodes_path: str,
) -> None:
    """Generate JSON report."""

    report = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "episodes_path": episodes_path,
        "metrics": metrics.to_dict(),
        "results": [
            {
                "episode_id": r.episode_id,
                "question_text": r.question_text,
                "difficulty": r.difficulty,
                "final_answer_correct": r.final_answer_correct,
                "execution_success": r.execution_success,
                "num_turns": r.num_turns,
                "elapsed_seconds": r.elapsed_seconds,
                "expected_answer": r.expected_answer,
                "actual_answer": r.actual_answer,
                "error_message": r.error_message,
            }
            for r in results
        ],
    }

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
