import tempfile, shutil, re
import torch
from src.environment.kernel import JupyterKernel, ExecutionResult
from src.llm import LLM

MAX_OUTPUT_CHARS = 10000
MAX_TURNS = 10

def strip_think_blocks(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def extract_code_blocks(text: str) -> list[str]:
    """Extract all code between <code> and </code> tags."""
    text = strip_think_blocks(text)
    pattern = r'<code>(.*?)</code>'
    return re.findall(pattern, text, re.DOTALL)

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

SYSTEM_PROMPT = """You have a Python notebook. Run code with <code> tags:

<code>
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
</code>

STATE PERSISTS between turns.

TASK: Explore data.csv and generate exam questions.

FORMAT each response as:
1. <code> block to explore
2. After seeing output, update your question list:

QUESTIONS:
1. [question] → [answer type, e.g. "number", "category name", "yes/no"]
2. ...

GOOD questions require computation:
- "Which treatment has the highest mean TL?" → category name
- "What % of rows have missing INTERNODE values?" → percentage
- "Is TL correlated with IN (r > 0.5)?" → yes/no

BAD questions are vague:
- "What is the relationship between X and Y?"
- "How does X affect Y?"

When you have 3+ good questions with verified answers, say DONE.

Start now."""

workdir = tempfile.mkdtemp()
csv_path = "data.csv"
shutil.copy(csv_path, workdir)

with JupyterKernel(workdir=workdir) as kernel:
    llm = LLM()
    try:
        conversation = [{"role": "user", "content": SYSTEM_PROMPT}]
        
        for turn in range(MAX_TURNS):
            print(f"\n{'='*60}\nTURN {turn + 1}\n{'='*60}")
            
            # Get LLM response
            raw_response = llm(conversation)
            response = strip_think_blocks(raw_response)
            print(f"\n[Assistant]\n{response}")
            conversation.append({"role": "assistant", "content": response})
            
            # Check for done signal
            if "DONE" in response:
                print("\n[Agent finished exploring]")
                break
            
            # Extract and execute code blocks
            code_blocks = extract_code_blocks(response)
            
            if not code_blocks:
                feedback = "No code block found. Use <code>...</code> to run Python code."
            else:
                results = []
                for i, code in enumerate(code_blocks):
                    result = kernel.execute(code.strip())
                    formatted = format_result(result)
                    results.append(f"[Cell {i+1}]\n{formatted}")
                    print(f"\n[Executed Cell {i+1}]\n{formatted}")
                
                feedback = "\n\n".join(results)
            
            # Add nudge to continue
            feedback += "\n\nUpdate your QUESTIONS list, then continue with <code> or say DONE."
            
            conversation.append({"role": "user", "content": feedback})
        else:
            print(f"\n[Reached max turns ({MAX_TURNS})]")
    finally:
        # Clear GPU memory
        del llm
        torch.cuda.empty_cache()