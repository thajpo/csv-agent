import re
import shutil
import tempfile

import pandas as pd
import torch

from src.kernel import JupyterKernel, ExecutionResult
from src.llm import APILLM
from src.tools import parse_tool_call, run_tool
from src.prompts import BOOTSTRAP_CODE, build_prompt, DEFAULT_DATASET_DESCRIPTION

MAX_OUTPUT_CHARS = 10000
MAX_TURNS = 10


def extract_code_blocks(text: str) -> list[str]:
    """Extract all code between <code> and </code> tags."""
    pattern = r'<code>(.*?)</code>'
    return re.findall(pattern, text, re.DOTALL)


def extract_questions(text: str) -> list[str]:
    """Extract all questions between {question} and {/question} tags."""
    pattern = r'\{question\}(.*?)\{/question\}'
    return [q.strip() for q in re.findall(pattern, text, re.DOTALL)]


def execute_tool_call(code: str, df: pd.DataFrame) -> str:
    """
    Parse and execute a tool call from a code block.
    
    Args:
        code: Raw content from <code>...</code> block (should be JSON)
        df: The dataframe to operate on
        
    Returns:
        Tool output or error message
    """
    result = parse_tool_call(code)
    
    if isinstance(result, str):
        # Parse error
        return f"[error]\n{result}"
    
    tool_name, params = result
    output = run_tool(tool_name, df, params)
    return f"[{tool_name}]\n{output}"


def format_result(result: ExecutionResult) -> str:
    """Format execution result for the LLM, truncating if needed."""
    parts = []
    
    if result.stdout:
        out = result.stdout
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(result.stdout)} chars total)"
        parts.append(f"[stdout]\n{out}")
    
    if result.result:
        out = result.result
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(result.result)} chars total)"
        parts.append(f"[result]\n{out}")
    
    if not result.success:
        parts.append(f"[error: {result.error_type}]\n{result.error_message}")
    
    return "\n".join(parts) if parts else "[no output]"


# --- Configuration ---
csv_path = "data.csv"
dataset_description = DEFAULT_DATASET_DESCRIPTION

# --- Load dataframe for tool execution ---
df = pd.read_csv(csv_path)

# --- Run exploration ---
workdir = tempfile.mkdtemp()
shutil.copy(csv_path, workdir)

with JupyterKernel(workdir=workdir) as kernel:
    llm = APILLM()
    try:
        # Run bootstrap exploration so the model sees real data first
        print("="*60)
        print("BOOTSTRAP: Running initial exploration...")
        print("="*60)
        bootstrap_result = kernel.execute(BOOTSTRAP_CODE)
        bootstrap_output = bootstrap_result.stdout or "[no output]"
        print(bootstrap_output[:2000] + "..." if len(bootstrap_output) > 2000 else bootstrap_output)
        
        system_prompt = build_prompt(dataset_description, bootstrap_output)
        conversation = [{"role": "user", "content": system_prompt}]
        
        for turn in range(MAX_TURNS):
            print(f"\n{'='*60}\nTURN {turn + 1}\n{'='*60}")
            
            # Get LLM response
            response = llm(conversation)
            print(f"\n[Assistant]\n{response}")
            conversation.append({"role": "assistant", "content": response})
            
            # Check for done signal
            if re.search(r'^DONE\b', response, re.MULTILINE):
                print("\n[Agent finished exploring]")
                # Extract final questions from conversation
                all_text = "\n".join(m["content"] for m in conversation if m["role"] == "assistant")
                questions = extract_questions(all_text)
                print(f"\n{'='*60}\nEXTRACTED QUESTIONS ({len(questions)})\n{'='*60}")
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q}")
                break
            
            # Extract and execute tool calls
            code_blocks = extract_code_blocks(response)
            
            if not code_blocks:
                feedback = "No tool call found. Use <code>{\"tool\": \"...\", ...}</code> to run a tool."
            else:
                results = []
                for i, code in enumerate(code_blocks):
                    output = execute_tool_call(code.strip(), df)
                    results.append(f"[Call {i+1}]\n{output}")
                    print(f"\n[Executed Call {i+1}]\n{output}")
                
                feedback = "\n\n".join(results)
            
            # Truncate feedback if too long
            if len(feedback) > MAX_OUTPUT_CHARS:
                feedback = feedback[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
            
            # Add nudge to continue
            feedback += "\n\nShare your interpretation, add {question}[DIFFICULTY]...{/question} tags, then <code> to continue. Write DONE when you have 8+ questions."
            
            conversation.append({"role": "user", "content": feedback})
        else:
            print(f"\n[Reached max turns ({MAX_TURNS})]")
    finally:
        # Clear GPU memory
        del llm
        torch.cuda.empty_cache()
