the old code uses tools

we are no longer going to use tools

the teacher now is told to solve the problem however it wants. at the end of its turn, it will give a final answer trace with *only* required tool calls

our verifier will use the output dataframe objects etc. as the checkpoints for more dense verification.

as you can imagine, this is a huge change.

we are also going back to the stateful jupyter environment

the use of rich logger will remain

i think most of the types and key contracts will remain 

my question: what remains, what is removed, what is added. i think that the following will 'go':
- tools
- prompts (mostly)

the other things.. should remain or be refactored.


# AI overview (gemini)
Gemini helped me plan. this is its take. my view (above) is the true answer if there is conflicting information

Here is the architectural summary of the **Stateful "Scavenger Hunt" Agent**.

### 1. System Overview
The objective is to train a local LLM to solve complex data analysis tasks (Pandas/CSV) via Reinforcement Learning (GRPO). The core philosophy is **"Code-as-Policy"** combined with **"Execution-as-Verification."** We reject custom tools and separate verifier models in favor of a stateful environment where valid intermediate data states serve as the reward signal.

### 2. The Environment: A Stateful Notebook MDP
We model the problem as a Markov Decision Process (MDP) that mimics a Jupyter Notebook.
* **State:** The cumulative context of the session (Question + History of Code Cells + Execution Outputs).
* **Action:** Generating a block of raw Python code (a "Cell").
* **Transition:** The code is executed in a persistent kernel. The resulting `STDOUT`, `STDERR`, and local variable state are appended to the context.
* **Termination:** The agent calls a `submit()` function or reaches a turn limit.
This persistence allows the agent to inspect data (`df.head()`), debug errors (`KeyError`), and iterate, creating a "Flow Engineering" loop rather than a brittle one-shot attempt.

### 3. The Teacher: "Tutor Mode" & Triangulation
Data generation relies on a strong model (Teacher) creating synthetic training episodes. To ensure quality without human labeling, we use a **Triangulation Protocol**:
* **Generation (The "Gold" Path):** The Teacher is prompted in **"Tutor Mode"**—explicitly instructed to solve the problem using verbose, step-by-step intermediate variables (e.g., `df_filtered`, `df_grouped`) rather than complex one-liners. This trace is generated *with* a hint.
* **Validation (The Consistency Path):** The Teacher solves the same problem $N$ times *without* the hint.
* **The Filter:** An episode is kept only if:
    1.  The Gold Trace executes without error.
    2.  The Gold Result matches the majority of the Consistency Results (Best-of-N).
    3.  The result is non-trivial (not empty/null).

### 4. The Reward Model: The "Scavenger Hunt"
Instead of rigid tool-based verification, we use a **State-Matching Reward** mechanism.
* **The Target:** The Teacher’s Gold Trace produces a "Bag of Artifacts"—a set of hashes representing every meaningful DataFrame created during the solution.
* **The Hunt:** As the Student generates code, we snapshot and hash its local variables after every cell execution.
* **The Reward:**
    * **Dense Reward (+1):** The Student creates a variable that matches a hash in the Teacher's bag. This confirms the Student found a correct intermediate step (e.g., correctly filtering the data), regardless of what they named the variable or how they wrote the code.
    * **Sparse Reward (+5):** The Student's final answer matches the Teacher's final hash.
    * **Penalty (-0.1):** Code crashes or produces no state change (discourages hallucination).

### 5. The Research Harness: Future-Proofing
While we are not training a verifier today, the system is designed to enable "Verifier Scaling Laws" research tomorrow.
* **The `EpisodeLog`:** We serialize the full state of every generation attempt—including "Hard Negatives" (traces that executed but failed the hash check) and "Near Misses" (traces that matched intermediate hashes but failed the final one).
* **Strategic Value:** This log file creates the perfect dataset to eventually train a discriminator model to detect subtle logic bugs that traditional unit tests miss.

### Summary
You are building an agent that learns to code by exploring a stateful environment. It is guided not by rigid syntax rules (Tools), but by reaching specific **Data Checkpoints** (The Scavenger Hunt) set by a Teacher who has proven their competence via self-consistency.