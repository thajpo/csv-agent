import tempfile, shutil, re
import torch
from src.environment.kernel import JupyterKernel, ExecutionResult
from src.llm import LLM, APILLM

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

def build_prompt(dataset_description: str) -> str:
    """Build system prompt with dataset context."""
    return f"""You have a Python notebook. Run code with <code> tags:

<code>
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
</code>

STATE PERSISTS between turns.

DATASET CONTEXT:
{dataset_description}

TASK: Explore data.csv and craft exam-style questions that REQUIRE computation. Use your domain knowledge above to ask meaningful questions.

FORMATTING RULES:
- Wrap runnable Python in <code> ... </code>.
- Wrap each question as {{question}}Question text → answer type{{/question}}. Use tight answer types (number, percentage, category name, yes/no, count, etc.).
- When you have 6+ strong, non-duplicate questions overall, write DONE on its own line and stop.

TURN TEMPLATE (must follow in this order every turn):
What I saw:
- 1–3 concise bullets of NEW observations from code you ran this turn (shape, types, missingness, distributions, notable values, relationships). Do NOT omit this section.
Questions:
- 3–5 varied, non-redundant questions that require computation and reference the observations above.
Next code:
<code>
# runnable Python that advances exploration (cleaning, stats, comparisons)
</code>

MANDATORY BEHAVIOR:
- If you ever miss the template or a section, start the next reply with "RETRY" and then follow the template correctly.
- Always run code each turn. If a code cell errors, fix it next turn (e.g., convert INTERNODE columns from "?" strings to numeric).
- Track what you already asked; do NOT repeat or lightly restate prior questions.
- Cover multiple columns over time: TL, IN, TR, TREE, BR, and several INTERNODE_* columns. Include differences/ratios between internodes, missingness checks, extremes, and group comparisons.
- Prefer computations that test understanding (means/medians/std/ranges, top/bottom groups, deltas between stages, correlations after cleaning).

HELPFUL STARTING CODE (you may adapt):
# Clean INTERNODE_* columns: turn '?' to NaN and cast numeric
int_cols = [c for c in df.columns if c.startswith("INTERNODE_")]
df[int_cols] = df[int_cols].replace("?", pd.NA)
df[int_cols] = df[int_cols].apply(pd.to_numeric, errors="coerce")

EXAMPLE PATTERNS (do NOT reuse verbatim):
{{question}}Which treatment has the widest TL range? → category name{{/question}}
{{question}}What percent of rows have any INTERNODE_* missing? → percentage{{/question}}
{{question}}Which tree shows the largest drop between INTERNODE_3 and INTERNODE_4 means? → category name{{/question}}

Start now."""

# --- Configuration ---
csv_path = "data.csv"

# Dataset description (manual for now, could come from Kaggle API later)
dataset_description = """
Tree branch growth measurements from an agricultural experiment.
- TR: Treatment (control, methanol_control, PP_333_4g/L, PP_333_20g/L, EL_500_4g/L, EL_500_20g/L)
- TREE: Tree identifier (e.g., G28, M33)
- BR: Branch label (A-J)
- TL: Total branch length (cm)
- IN: Number of internodes
- INTERNODE_1 to INTERNODE_29: Length of each internode (cm), '?' = missing

Goal: Understand how treatments affect branch growth patterns.
""".strip()

# --- Run exploration ---
workdir = tempfile.mkdtemp()
shutil.copy(csv_path, workdir)

with JupyterKernel(workdir=workdir) as kernel:
    llm = APILLM()
    try:
        system_prompt = build_prompt(dataset_description)
        conversation = [{"role": "user", "content": system_prompt}]
        
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
