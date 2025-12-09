# 12/8/25
I had claude do a major refactor, after realizing that the tool-based approach was slowing me down, and probably not effective.

I now need to parse through the new logic. I guided claude through the changes, but there are new features that I need to look at more deeply.

# 12/6/25
I have been relying on not writing things down. it does not work

Talking to LLMs has a limited scope

My current project needs to be split up into these two projects:
1. Question generation from CSVs
2. Answering questions about CSVs

But, since I already have tools, I'll also be using the question generation as training data, with my tools.


So, how can I make good progress?

My goal is to make an agent that navigates a CSV, and can answer multi-step questions about it

I first need a dataset of:
- question
- hint
- dag (loose verification)
- answer(s) (tight verification)
- conversation 

The conversation is used for SFT, the hooks are used for RL

I need this structure on huggingface.

We have made question generation a seperate step from answering

The question generation will still be under the same environmental constraints. The only difference will be the prompt


# 12/5/25
- Reoganizing focus
The pipeline currently has a couple of operating modes:
1. dataset exploration and question generation with solution trace (in text)
2. tool use exploration
3. (planned) question solving with hint
4. (planned) best of n queston solving with no hint

We also need to integrate setps 3 and 4.
Before doing this, we need to make the output of step 1 most useful to the LLM

I'm also fighting context issues

For the teacher, providing the full summary of all tools in the sysprompt is useful

The student will have this information in its weights, so it will not need to be specified in the future

I also am thinking about compresssion

## More clearly
GRPO -> Step 4
SFT -> Step 1,3,4

Important tasks
- Make the output of step 1 clear for the development of step 3
- Make step 3 output a full hook trace with the provided hint
- Make step 4 use a best-of-n solution per problem from step 1
- Make a comparison verifier between steps 3 and 4 to determine which traces from step 4 are good training data
- Add compacting of context (?) - maybe
- Add smarter truncation of tool calls
    I think that we can have the model say what it is looking for in a tool call, and summarize the result of the tool call clearly. In this way, we dont have context explode, and the model learns to summarize what it finds as it goes. This is totally unverifiable, and ... 

So.. what *features* do we want?
We need smart context management
This should be as dumb as possible. We truncate outputs. 


## Datasets
Everything should be a .jsonl
We need to make both SFT and RL in the same pass

SFT is just tokens. The more tokens the better
RL is focused on verifiable hooks. We need to ensure that the produced hooks make sense etc.

For question generation, this is useful for SFT

Ultimately, we need to add context comprssion (per tool call) before even saving data



# 12/4/25
- We need to trim down the prompts (not yet)
- 

# 12/2/25
I think the best thing I can do is navigate the codebase, and report my observations here.
- work on context compression
    - summaries of specific tool calls
        - need to look at how tools work

## how do tools work
we use the openai tool calling api


# 12/1/25
- revise the tools: 
    - are they single function
    - are we missing any necessary tools (good to run the 'get tool' prompt for different datasets)
- improve the prompt structure
    - currently we do questions + answers in the same prompt. this gives hallucination
    - we should do exploration + questions 
    - then we should run best of n on the questions, and save the common DAG
- spend more time thinking about DAG and edge cases
- think about the proper model output
- what is the final data structure for the training data?
    - should this be seperate for SFT and RL?
- ..

Tomorrow: expand on these questions. Pick one, work on it.

# old
What is the goal of this project?

Well, I spent a lot of time trying to get LLMs to help me spec things out. It was helpful to understand what direction to move in. It was not helpful in when it comes to "how do I actually make this agent harness". I think solving this question on my own, or making a good attempt with minimal LLM usage is very important as a *learning experience*

1. Load csv
2. Expose tools to teacher (how?)
    what tools to give it?
3. Have teacher come up with questions and answer pairs 
4. Give the question to the teacher again, and see if it can solve the question
5. If it does solve the question, then we save that data (not clear what this data *is*)


So we have python tools. This means that we cannot allow the LLM to import anything.
The LLM is not allowed to use python.
I think the way is that we have the LLM make <code> block calls, but we intercept before passing into the interpreter.

What else is confusing

Well, how do we define the different datatypes and classes. I think this will reveal itself as I code this up.

I am also confused on what tools we want to expose. We already have the pandas tool file. I might just keep that

I am very confused on the hook and contracts we use to store 'states' for the problem solution.

There are many ways to cook an egg... should we be so strict in the hook outputs? Maybe


Okay

It is a new day


