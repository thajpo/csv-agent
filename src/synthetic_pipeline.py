"""
Rich terminal entrypoint for the CSV exploration agent.

Usage:
    python -m src.rich_pipeline --mode explore --output questions.jsonl
    python -m src.rich_pipeline --mode episodes --output episodes.jsonl
    python -m src.rich_pipeline --mode tool-feedback --output tool_requests.jsonl
    
    python -m src.rich_pipeline --csv data.csv --max-turns 10
    python -m src.rich_pipeline --output questions.jsonl --target-questions 12
    
    # Tool feedback mode - identify missing tools
    python -m src.rich_pipeline --tool-feedback --output tool_requests.jsonl
"""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from src.model import APILLM
from src.tools import parse_tool_call, run_tool
from src.prompts import (
    DEFAULT_DATASET_DESCRIPTION,
    build_prompt,
    build_question_generation_prompt,
    build_tool_feedback_prompt,
    generate_bootstrap_output,
)
from src.text_extraction import (
    extract_code_blocks,
    extract_json_array,
    extract_json_episodes,
    extract_question_plans,
)
from rich.console import Console
from rich.panel import Panel

console = Console()


@dataclass
class ModeConfig:
    """Mode-specific configuration for the pipeline."""
    system_prompt: str
    extractor: Callable[[str], list[dict] | None]
    success_label: str
    parse_error_msg: str
    continue_msg: str
    final_msg: str


def get_mode_config(
    mode: str,
    dataset_description: str,
    bootstrap_output: str,
    target_questions: int,
) -> ModeConfig:
    """Build mode-specific config. Call this once at pipeline start."""
    
    if mode == "explore":
        return ModeConfig(
            system_prompt=build_question_generation_prompt(dataset_description, bootstrap_output, target_questions),
            extractor=extract_question_plans,
            success_label="question plans",
            parse_error_msg="[red]✗ Failed to parse question plans[/red]",
            continue_msg="\n\nContinue exploring the dataset. When done, write DONE and output your question plans as JSON.",
            final_msg="You've reached the turn limit. Please output your question plans now as a JSON array. Write DONE then the JSON.",
        )
    
    elif mode == "episodes":
        return ModeConfig(
            system_prompt=build_prompt(dataset_description, bootstrap_output),
            extractor=extract_json_episodes,
            success_label="episodes",
            parse_error_msg="[red]✗ Failed to parse episodes[/red]",
            continue_msg="\n\nAbove are the actual tool results from your calls. Use these real values to inform your next exploration steps. Continue exploring the dataset - make 3-8 parallel tool calls per turn to explore broadly (different treatments, columns, relationships simultaneously). Observe patterns, brainstorm questions. Do NOT output episodes yet - you need multiple turns of exploration first. When you have thoroughly explored (typically 5-8 turns), then write DONE and output your final 10 episodes as JSON.",
            final_msg="You've reached the turn limit. Please output your final 10 episodes now as a JSON array. Write DONE then the JSON.",
        )
    
    elif mode == "tool-feedback":
        return ModeConfig(
            system_prompt=build_tool_feedback_prompt(dataset_description, bootstrap_output),
            extractor=extract_json_array,
            success_label="tool recommendations",
            parse_error_msg="[red]✗ Failed to parse tool recommendations (check for invalid JSON like {...} placeholders)[/red]",
            continue_msg="\n\nContinue exploring. Note any tool friction with <TOOL_WISH>...</TOOL_WISH> tags. When done, write DONE and output your tool recommendations as JSON.",
            final_msg="You've reached the turn limit. Please output your tool recommendations now as a JSON array. Write DONE then the JSON.",
        )
    
    else:
        valid_modes = "explore, episodes, tool-feedback"
        raise ValueError(f"Unknown mode '{mode}' (expected one of: {valid_modes})")

def execute_tool_call(code: str, df: pd.DataFrame) -> tuple[str, str, bool]:
    """Parse and execute a tool call. Returns (tool_name, output, success)."""
    result = parse_tool_call(code)
    
    if isinstance(result, str):
        return ("error", result, False)
    
    tool_name, params = result
    output = run_tool(tool_name, df, params)
    return (tool_name, output, True)

def summarize_results(data: list[dict] | None, pipeline_mode: str, parse_error_msg: str, success_label: str):
    """Pretty-print parsed results depending on pipeline mode."""
    if not data:
        console.print(parse_error_msg)
        return
    
    console.print(f"\n[green]✓ {len(data)} {success_label}[/green]")
    
    pipeline_mode = pipeline_mode.lower()
    
    if pipeline_mode == "tool-feedback":
        for i, rec in enumerate(data, 1):
            name = rec.get("name", "?")
            priority = rec.get("priority", "?")
            why = rec.get("why", "?")[:60]
            console.print(f"  [dim]{i}.[/dim] [{priority}] [bold]{name}[/bold]: {why}")
    elif pipeline_mode == "explore":
        for i, plan in enumerate(data, 1):
            diff = plan.get("difficulty", "?")
            steps = plan.get("expected_steps", "?")
            q = plan.get("question_text", "?")
            console.print(f"  [dim]{i}.[/dim] [{diff}|steps={steps}] {q}")
    else:
        for i, ep in enumerate(data, 1):
            if isinstance(ep, dict):
                diff = ep.get("difficulty", "?")
                q = ep.get("question_text", "?")
                n_hooks = len(ep.get("hooks", []))
                console.print(f"  [dim]{i}.[/dim] [{diff}] ({n_hooks}h) {q}")


def run_pipeline(
    csv_path: str = "data.csv",
    dataset_description: str = DEFAULT_DATASET_DESCRIPTION,
    max_turns: int = 10,
    output_path: str | None = None,
    target_questions: int = 10,
    pipeline_mode: str = "explore",  # "explore" (question plans), "episodes", or "tool-feedback"
    teacher_model: str = "grok-4.1-fast",
):
    """Run the full exploration pipeline with rich output."""
    
    pipeline_mode = pipeline_mode.lower()
    console.print(f"\n[bold blue]CSV Agent[/bold blue] → {csv_path} [dim]({pipeline_mode})[/dim]\n")
    console.print(f"[dim]Loading {csv_path}...[/dim]", end=" ")
    df = pd.read_csv(csv_path)
    console.print(f"[green]✓[/green] {len(df):,} × {len(df.columns)}")
    
    results_data = []
    n_turns = 0
    
    # Bootstrap exploration to get initial context
    bootstrap_output = generate_bootstrap_output(csv_path)
    console.print(Panel(bootstrap_output, title="[cyan]Bootstrap[/cyan]", border_style="dim"))
    
    # Get mode-specific config (prompt, extractor, messages)
    config = get_mode_config(pipeline_mode, dataset_description, bootstrap_output, target_questions)
    
    llm = APILLM(model=teacher_model)
    
    try:
        conversation = [{"role": "user", "content": config.system_prompt}]
        
        for turn in range(1, max_turns + 1):
            n_turns = turn
            console.rule(f"[bold]Turn {turn}/{max_turns}[/bold]", style="blue")
            
            with console.status("[magenta]Thinking...[/magenta]", spinner="dots"):
                response = llm(conversation)
            
            # Check for tool calls FIRST - if present, truncate response after them
            code_blocks = extract_code_blocks(response)
            
            if code_blocks:
                # Find the last </code> tag and truncate everything after it
                last_code_end = response.rfind('</code>')
                if last_code_end != -1:
                    truncated = response[:last_code_end + len('</code>')]
                    if len(truncated) < len(response):
                        response = truncated
                        console.print("[dim](truncated after tool calls)[/dim]")
            
            console.print(Panel(response, title="[magenta]Assistant[/magenta]", border_style="magenta"))
            conversation.append({"role": "assistant", "content": response})
            
            # Check for done signal (only valid if NO tool calls in response)
            if re.search(r'^DONE\b', response, re.MULTILINE) and not code_blocks:
                console.print("\n[bold green]✓ DONE[/bold green]")
                # Re-extract from original response in case DONE was after truncation
                results_data = config.extractor(response)
                summarize_results(results_data, pipeline_mode, config.parse_error_msg, config.success_label)
                break
            
            if not code_blocks:
                console.print("[yellow]⚠ No tool call[/yellow]")
                feedback = "No tool call found. Use <code>{\"tool\": \"...\", ...}</code> to explore the data."
            else:
                tool_results = []
                for i, code in enumerate(code_blocks, 1):
                    tool_name, output, success = execute_tool_call(code.strip(), df)
                    style = "green" if success else "red"
                    console.print(f"[{style}]{'✓' if success else '✗'}[/{style}] [yellow]{tool_name}[/yellow]")
                    console.print(f"[dim]{output}[/dim]")
                    tool_results.append(f"[Call {i}]\n[{tool_name}]\n{output}")
                
                feedback = "\n\n".join(tool_results)
            
            feedback += config.continue_msg
            conversation.append({"role": "user", "content": feedback})
        
        else:
            # Reached max turns without DONE
            console.print(f"[yellow]Max turns ({max_turns})[/yellow]")
            
            conversation.append({"role": "user", "content": config.final_msg})
            
            with console.status("[magenta]Generating...[/magenta]", spinner="dots"):
                response = llm(conversation)
            
            console.print(Panel(response, title="[magenta]Assistant[/magenta]", border_style="magenta"))
            results_data = config.extractor(response)
            summarize_results(results_data, pipeline_mode, config.parse_error_msg, config.success_label)
    
    finally:
        del llm
        torch.cuda.empty_cache()
    
    # Save if output path provided
    if output_path and results_data:
        metadata = {
            "dataset_id": Path(csv_path).stem,
            "generation_timestamp": datetime.now().isoformat(),
            "teacher_model": teacher_model,
            "n_turns": n_turns,
            "mode": pipeline_mode,
        }
        if pipeline_mode == "explore":
            metadata["target_questions"] = target_questions
        with open(output_path, "w") as f:
            for item in results_data:
                f.write(json.dumps({**metadata, **item}) + "\n")
        console.print(f"[green]✓[/green] Saved {len(results_data)} → [bold]{output_path}[/bold]")
    
    console.print("[dim]Done.[/dim]")
    return results_data


def main():
    parser = argparse.ArgumentParser(description="CSV Exploration Agent with Rich Terminal UI")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV file")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum conversation turns")
    parser.add_argument("--description", default=None, help="Dataset description (uses default if not provided)")
    parser.add_argument("--output", "-o", default=None, help="Output path for JSONL")
    parser.add_argument("--mode", choices=["explore", "episodes", "tool-feedback"], default="explore", help="Pipeline mode: explore=question plans, episodes=full hook episodes, tool-feedback=tool gap analysis")
    parser.add_argument("--target-questions", type=int, default=10, help="Number of question blueprints to generate in explore mode")
    parser.add_argument("--tool-feedback", action="store_true", help="Run in tool feedback mode to identify missing tools (alias for --mode tool-feedback)")
    parser.add_argument("--teacher-model", default="grok-4.1-fast", help="Teacher model identifier for metadata")
    args = parser.parse_args()
    
    description = args.description or DEFAULT_DATASET_DESCRIPTION
    pipeline_mode = "tool-feedback" if args.tool_feedback else args.mode
    
    try:
        run_pipeline(
            csv_path=args.csv,
            dataset_description=description,
            max_turns=args.max_turns,
            output_path=args.output,
            pipeline_mode=pipeline_mode,
            target_questions=args.target_questions,
            teacher_model=args.teacher_model,
        )
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
    except Exception:
        console.print_exception()


if __name__ == "__main__":
    main()
