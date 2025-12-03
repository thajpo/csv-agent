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


