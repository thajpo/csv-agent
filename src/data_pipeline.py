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

# Initial exploration code - runs automatically before Turn 1 so the model sees real data
BOOTSTRAP_CODE = """
import pandas as pd
import numpy as np

# Load and get first impressions
df = pd.read_csv("data.csv")
print("=== SHAPE ===")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")

print("\\n=== FIRST 5 ROWS ===")
print(df.head().to_string())

print("\\n=== DATA TYPES ===")
print(df.dtypes.to_string())

print("\\n=== NUMERIC SUMMARY ===")
print(df.describe().to_string())

print("\\n=== CATEGORICAL COLUMNS ===")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].nunique()} unique → {df[col].value_counts().head(3).to_dict()}")

print("\\n=== MISSING VALUES ===")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0].to_string())
else:
    print("No null values (but check for placeholder strings like '?' or 'NA')")
""".strip()

def build_prompt(dataset_description: str, bootstrap_output: str) -> str:
    """Build system prompt with dataset context and initial exploration results."""
    return f"""You are a senior data scientist creating exam questions. Your goal is to demonstrate expert-level thinking: understanding what the data represents, forming hypotheses, and crafting questions that test deep comprehension.

DATASET CONTEXT (provided by user):
{dataset_description}

I've already run initial exploration. Here's what the data looks like:

```
{bootstrap_output}
```

YOUR TASK:
1. First, explain what you think this dataset is really about. What story does it tell? What would a domain expert care about?
2. Explore further to understand patterns, relationships, and edge cases.
3. Craft exam questions at THREE difficulty levels:
   - MEDIUM: Direct computations (means, counts, percentages)
   - HARD: Multi-step analysis (group comparisons, correlations, filtering + aggregation)
   - VERY HARD: Insight questions requiring domain reasoning (anomalies, relationships between variables, what-if scenarios)

FORMATTING:
- Run code with <code>...</code> tags. State persists between turns.
- Tag each question with difficulty: {{question}}[MEDIUM] Question → answer type{{/question}}
- Answer types: number, percentage, category name, yes/no, count, list

TURN STRUCTURE:
1. **My interpretation**: What does this data/output tell us? (1-2 sentences of insight)
2. **Questions**: 3-5 new questions at varied difficulties, grounded in what you just saw
3. **Next exploration**: <code> block that digs deeper (clean data, compute stats, find patterns)

THINK LIKE AN EXPERT:
- Why would someone collect this data? What decisions would it inform?
- What relationships SHOULD exist if the domain logic holds?
- Where might the data be messy, surprising, or misleading?
- What would separate a student who memorized formulas from one who understands the domain?

When you have 8+ strong questions across all difficulty levels, write DONE and list your final questions grouped by difficulty.

Begin by sharing your interpretation of what this dataset is about, then continue exploring."""

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
            feedback += "\n\nShare your interpretation, add {question}[DIFFICULTY]...{/question} tags, then <code> to continue. Write DONE when you have 8+ questions."
            
            conversation.append({"role": "user", "content": feedback})
        else:
            print(f"\n[Reached max turns ({MAX_TURNS})]")
    finally:
        # Clear GPU memory
        del llm
        torch.cuda.empty_cache()
