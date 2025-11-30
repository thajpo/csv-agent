"""
Rich terminal entrypoint for the CSV exploration agent.

Usage:
    python -m src.rich_pipeline
    python -m src.rich_pipeline --csv data.csv --max-turns 10
    python -m src.rich_pipeline --output episodes.jsonl
    
    # Tool feedback mode - identify missing tools
    python -m src.rich_pipeline --tool-feedback --output tool_requests.jsonl
"""

import argparse
import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.kernel import JupyterKernel
from src.llm import APILLM
from src.tools import parse_tool_call, run_tool
from src.prompts import BOOTSTRAP_CODE, build_prompt, build_tool_feedback_prompt, DEFAULT_DATASET_DESCRIPTION
from src.text_extraction import extract_code_blocks, extract_json_episodes, extract_json_array
from src import terminal_display as ui
from rich.console import Console

console = Console()

def execute_tool_call(code: str, df: pd.DataFrame) -> tuple[str, str, bool]:
    """Parse and execute a tool call. Returns (tool_name, output, success)."""
    result = parse_tool_call(code)
    
    if isinstance(result, str):
        return ("error", result, False)
    
    tool_name, params = result
    output = run_tool(tool_name, df, params)
    return (tool_name, output, True)


def save_episodes(episodes: list[dict], output_path: str, metadata: dict):
    """Save episodes to JSONL file."""
    path = Path(output_path)
    
    with open(path, "w") as f:
        for ep in episodes:
            full_episode = {**metadata, **ep}
            f.write(json.dumps(full_episode) + "\n")
    
    ui.saved(str(path), len(episodes))


def run_pipeline(
    csv_path: str = "data.csv",
    dataset_description: str = DEFAULT_DATASET_DESCRIPTION,
    max_turns: int = 10,
    output_path: str | None = None,
    mode: str = "episodes",  # "episodes" or "tool-feedback"
):
    """Run the full exploration pipeline with rich output."""
    
    ui.header(csv_path)
    ui.loading(csv_path)
    df = pd.read_csv(csv_path)
    ui.loaded(len(df), len(df.columns))
    
    # Setup temp workdir
    workdir = tempfile.mkdtemp()
    shutil.copy(csv_path, workdir)
    
    results_data = []
    n_turns = 0
    
    # Mode-specific configuration
    if mode == "tool-feedback":
        prompt_builder = build_tool_feedback_prompt
        continue_msg = "\n\nContinue exploring. Note any tool friction with {TOOL_WISH} tags. When done, write DONE and output your tool recommendations as JSON."
        final_msg = "You've reached the turn limit. Please output your tool recommendations now as a JSON array. Write DONE then the JSON."
    else:
        prompt_builder = build_prompt
        continue_msg = "\n\nContinue exploring. Note candidate questions as you go. When ready, write DONE and output your final 10 episodes as JSON."
        final_msg = "You've reached the turn limit. Please output your final 10 episodes now as a JSON array. Write DONE then the JSON."
    
    with JupyterKernel(workdir=workdir) as kernel:
        llm = APILLM()
        
        try:
            # Bootstrap exploration
            ui.bootstrap_start()
            bootstrap_result = kernel.execute(BOOTSTRAP_CODE)
            bootstrap_output = bootstrap_result.stdout or "[no output]"
            ui.bootstrap_output(bootstrap_output)
            
            # Build prompt and start conversation
            system_prompt = prompt_builder(dataset_description, bootstrap_output)
            conversation = [{"role": "user", "content": system_prompt}]
            
            for turn in range(1, max_turns + 1):
                n_turns = turn
                ui.turn_start(turn, max_turns)
                
                with ui.thinking():
                    response = llm(conversation)
                
                ui.assistant(response, turn)
                conversation.append({"role": "assistant", "content": response})
                
                # Check for done signal
                if re.search(r'^DONE\b', response, re.MULTILINE):
                    ui.done_signal()
                    if mode == "tool-feedback":
                        results_data = extract_json_array(response)
                        if results_data:
                            console.print(f"\n[green]✓ {len(results_data)} tool recommendations[/green]")
                            for i, rec in enumerate(results_data, 1):
                                name = rec.get("name", "?")
                                priority = rec.get("priority", "?")
                                why = rec.get("why", "?")[:60]
                                console.print(f"  [dim]{i}.[/dim] [{priority}] [bold]{name}[/bold]: {why}")
                        else:
                            ui.parse_failed(response)
                    else:
                        results_data = extract_json_episodes(response)
                        if results_data:
                            ui.episodes_summary(results_data)
                            for i, ep in enumerate(results_data, 1):
                                ui.episode(ep, i)
                        else:
                            ui.parse_failed(response)
                    break
                
                # Execute tool calls
                code_blocks = extract_code_blocks(response)
                
                if not code_blocks:
                    ui.no_tool_call()
                    feedback = "No tool call found. Use <code>{\"tool\": \"...\", ...}</code> to explore the data."
                else:
                    tool_results = []
                    for i, code in enumerate(code_blocks, 1):
                        tool_name, output, success = execute_tool_call(code.strip(), df)
                        ui.tool_result(code, tool_name, output, success, i)
                        tool_results.append(f"[Call {i}]\n[{tool_name}]\n{output}")
                    
                    feedback = "\n\n".join(tool_results)
                
                feedback += continue_msg
                conversation.append({"role": "user", "content": feedback})
            
            else:
                # Reached max turns without DONE
                ui.max_turns_reached(max_turns)
                
                conversation.append({"role": "user", "content": final_msg})
                
                with ui.generating_final():
                    response = llm(conversation)
                
                ui.assistant(response, max_turns + 1)
                if mode == "tool-feedback":
                    results_data = extract_json_array(response)
                else:
                    results_data = extract_json_episodes(response)
                
                if results_data:
                    if mode == "tool-feedback":
                        console.print(f"\n[green]✓ {len(results_data)} tool recommendations[/green]")
                    else:
                        ui.episodes_summary(results_data)
                        for i, ep in enumerate(results_data, 1):
                            ui.episode(ep, i)
                else:
                    ui.parse_failed(response)
        
        finally:
            del llm
            torch.cuda.empty_cache()
    
    # Save if output path provided
    if output_path and results_data:
        metadata = {
            "dataset_id": Path(csv_path).stem,
            "generation_timestamp": datetime.now().isoformat(),
            "teacher_model": "grok-4.1-fast",
            "n_turns": n_turns,
            "mode": mode,
        }
        with open(output_path, "w") as f:
            for item in results_data:
                f.write(json.dumps({**metadata, **item}) + "\n")
        ui.saved(output_path, len(results_data))
    
    ui.cleanup()
    return results_data


def main():
    parser = argparse.ArgumentParser(description="CSV Exploration Agent with Rich Terminal UI")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV file")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum conversation turns")
    parser.add_argument("--description", default=None, help="Dataset description (uses default if not provided)")
    parser.add_argument("--output", "-o", default=None, help="Output path for JSONL")
    parser.add_argument("--tool-feedback", action="store_true", help="Run in tool feedback mode to identify missing tools")
    args = parser.parse_args()
    
    description = args.description or DEFAULT_DATASET_DESCRIPTION
    mode = "tool-feedback" if args.tool_feedback else "episodes"
    
    try:
        run_pipeline(
            csv_path=args.csv,
            dataset_description=description,
            max_turns=args.max_turns,
            output_path=args.output,
            mode=mode,
        )
    except KeyboardInterrupt:
        ui.interrupted()
    except Exception:
        ui.exception()


if __name__ == "__main__":
    main()
