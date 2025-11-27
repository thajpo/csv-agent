- Get a sample csv
- Test pipeline on small LLM
- Integrate vLLM

Goals:
1. Get a csv loaded
2. Have an LLM navigate the CSV in the environment and generate Q&A pairs for classification or regression.
3. Have an LLM solve Q&A pairs. 
4. Have an LLM corrupt the CSV with a set of functions (possibly dependent on the problem type)
4. Have the LLM continue solving the problem with increased corruption
5. Save the trajectories and upload the dataset to huggingface.