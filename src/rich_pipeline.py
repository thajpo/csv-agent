"""
Rich terminal entrypoint for the CSV exploration agent.

Usage:
    python -m src.rich_pipeline
    python -m src.rich_pipeline --csv data.csv --max-turns 10
    python -m src.rich_pipeline --output episodes.jsonl
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
from src.prompts import BOOTSTRAP_CODE, build_prompt, DEFAULT_DATASET_DESCRIPTION
from src.text_extraction import extract_code_blocks, extract_json_episodes
from src import terminal_display as ui

MAX_OUTPUT_CHARS = 10000


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
):
    """Run the full exploration pipeline with rich output."""
    
    ui.header(csv_path)
    ui.loading(csv_path)
    df = pd.read_csv(csv_path)
    ui.loaded(len(df), len(df.columns))
    
    # Setup temp workdir
    workdir = tempfile.mkdtemp()
    shutil.copy(csv_path, workdir)
    
    episodes = []
    n_turns = 0
    
    with JupyterKernel(workdir=workdir) as kernel:
        llm = APILLM()
        
        try:
            # Bootstrap exploration
            ui.bootstrap_start()
            bootstrap_result = kernel.execute(BOOTSTRAP_CODE)
            bootstrap_output = bootstrap_result.stdout or "[no output]"
            ui.bootstrap_output(bootstrap_output)
            
            # Build prompt and start conversation
            system_prompt = build_prompt(dataset_description, bootstrap_output)
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
                    episodes = extract_json_episodes(response)
                    
                    if episodes:
                        ui.episodes_summary(episodes)
                        for i, ep in enumerate(episodes, 1):
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
                    results = []
                    for i, code in enumerate(code_blocks, 1):
                        tool_name, output, success = execute_tool_call(code.strip(), df)
                        ui.tool_result(code, tool_name, output, success, i)
                        results.append(f"[Call {i}]\n[{tool_name}]\n{output}")
                    
                    feedback = "\n\n".join(results)
                
                # Truncate if needed
                if len(feedback) > MAX_OUTPUT_CHARS:
                    feedback = feedback[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
                
                feedback += "\n\nContinue exploring. Note candidate questions as you go. When ready, write DONE and output your final 10 episodes as JSON."
                conversation.append({"role": "user", "content": feedback})
            
            else:
                # Reached max turns without DONE
                ui.max_turns_reached(max_turns)
                
                conversation.append({
                    "role": "user",
                    "content": "You've reached the turn limit. Please output your final 10 episodes now as a JSON array. Write DONE then the JSON."
                })
                
                with ui.generating_final():
                    response = llm(conversation)
                
                ui.assistant(response, max_turns + 1)
                episodes = extract_json_episodes(response)
                
                if episodes:
                    ui.episodes_summary(episodes)
                    for i, ep in enumerate(episodes, 1):
                        ui.episode(ep, i)
                else:
                    ui.parse_failed(response)
        
        finally:
            del llm
            torch.cuda.empty_cache()
    
    # Save if output path provided
    if output_path and episodes:
        metadata = {
            "dataset_id": Path(csv_path).stem,
            "generation_timestamp": datetime.now().isoformat(),
            "teacher_model": "grok-4.1-fast",
            "n_turns": n_turns,
        }
        save_episodes(episodes, output_path, metadata)
    
    ui.cleanup()
    return episodes


def main():
    parser = argparse.ArgumentParser(description="CSV Exploration Agent with Rich Terminal UI")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV file")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum conversation turns")
    parser.add_argument("--description", default=None, help="Dataset description (uses default if not provided)")
    parser.add_argument("--output", "-o", default=None, help="Output path for episodes JSONL")
    args = parser.parse_args()
    
    description = args.description or DEFAULT_DATASET_DESCRIPTION
    
    try:
        run_pipeline(
            csv_path=args.csv,
            dataset_description=description,
            max_turns=args.max_turns,
            output_path=args.output,
        )
    except KeyboardInterrupt:
        ui.interrupted()
    except Exception:
        ui.exception()


if __name__ == "__main__":
    main()
