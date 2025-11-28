import tempfile, shutil, re
import torch
from src.environment.kernel import JupyterKernel, ExecutionResult
from src.llm import LLM

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

TASK: Explore data.csv and craft exam-style questions that REQUIRE computation.

FORMATTING:
- Wrap runnable Python in <code> ... </code>.
- Wrap each question as {question}Question text → answer type{/question}. Use concise answer types (number, percentage, category name, yes/no, count, etc.).
- When you have 3+ strong questions, write DONE on its own line.

TURN-BY-TURN PROTOCOL (follow in order):
1) Explore before asking: run code to inspect shape, columns, missingness, distributions, and relationships. Start by loading the CSV and printing columns/summary.
2) Write 1-3 bullets under "What I saw:" summarizing NEW findings this turn. Do not skip this note.
3) Propose your best 3-5 non-redundant questions based on those findings. Prefer variety and depth.
4) Add a code cell that advances exploration (e.g., compute stats that answer or inspire the next questions).

QUESTION QUALITY BAR:
- Must be answerable from the data via computation; no vague or subjective prompts.
- Cover multiple columns over time; mix aggregates (mean/median/std/range), comparisons (by treatment/tree/stage), missingness, extremes/outliers, and relationships (ratios, differences between internode stages, correlations with TL).
- Avoid repeating prior questions and avoid trivial restatements. Do not fixate on a single column (e.g., TL); include INTERNODE_* and other fields.
- If a column seems empty/constant, ask a verification question about it. If unsure what to ask, run a quick probe to surface anomalies first.

EXAMPLE PATTERNS (illustrative—do NOT reuse verbatim):
{question}Which treatment has the widest TL range? → category name{/question}
{question}What percent of rows have any INTERNODE_* missing? → percentage{/question}
{question}Which tree shows the largest drop between INTERNODE_3 and INTERNODE_4 means? → category name{/question}

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
            response = llm(conversation)
            print(f"\n[Assistant]\n{response}")
            conversation.append({"role": "assistant", "content": response})
            
            # Check for done signal - must be standalone, not part of nudge
            if re.search(r'^DONE\b', response, re.MULTILINE):
                print("\n[Agent finished exploring]")
                # Extract final questions from conversation
                all_text = "\n".join(m["content"] for m in conversation if m["role"] == "assistant")
                questions = extract_questions(all_text)
                print(f"\n{'='*60}\nEXTRACTED QUESTIONS ({len(questions)})\n{'='*60}")
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q}")
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
            feedback += "\n\nList your {question}...{/question} tags, then <code> to continue or write DONE."
            
            conversation.append({"role": "user", "content": feedback})
        else:
            print(f"\n[Reached max turns ({MAX_TURNS})]")
    finally:
        # Clear GPU memory
        del llm
        torch.cuda.empty_cache()
