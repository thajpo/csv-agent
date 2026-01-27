# questions.md

This file is for our back-and-forth questions and decisions. Add your notes or answers under the relevant section. I will respond in this file and in chat as needed.

---

## 1) Goals and scope
- What is the single most important outcome right now? (quality / quantity / correctness / speed)
Most important thing is difficult and data aware questions that make sense. This probably needs clearer definitions

- Is program-based generation first-class (CLI) or experimental?
It will become first-class

- Should program-based questions be labeled as "synthetic" or a new source?
It is synthetic. Perhaps we should make distinction between hardcoded and procedural questions

## 2) Keep separate vs consolidate
- Which separations are non-negotiable? (template vs program vs LLM)
Not clear wym

- Should we share one execution layer across all sources?
If you mean execution layer as-in the python REPL, i see no reason not to share

- Is having both `synthetic/generator.py` and `validate_synthetic.py` intentional?
Not clear... seems like generator.py is doing too much.

## 3) Outputs and artifacts
- One unified questions file or separate per source?
We should have a unified schema, but seperate outputs by question types

- Where should program-based questions be saved?
Not sure yet.

- Should we always emit both mechanical + verbalized questions?
No, verbalization is an optional flag

## 4) Evaluation settings
- Should evaluation include hints for verbalized questions?
I think a hint flag is the way here. So verbalization always generates hints? Then we just decide whether to provide at eval time. Should define what hint is

- Evaluate mechanical only, verbalized only, or both?
Both

- Should evaluation be mandatory or optional for program-based output?
Evaluation is a seperate process.

## 5) Question quality filters (minimal)
- Confirm obvious filters: ID-like columns, near-zero variance, tiny groups, heavy missingness.
i think our data profiler might be a good way to centrlaize

- Should filters be strict or lenient?
?

## 6) Verbalization details
- Should verbalized questions include real column names?
Yes

- Should hints be allowed to include column names too?
Yes

- If code excludes ID-like columns, can the verbalizer mention it?
Yes

## 7) Manifest and caching
- Should program-based questions be in the manifest?
manifest? i think ... yes - questions are saved to avoid re-gen? ... we want to avoid re-gen if our templates dont change. generating quetions is cheap, validation is expensive.

- Fingerprint by program spec or by question text?
... program spec . i think we should hash the code? what do we do elsewhere?

## 8) Difficulty / metadata
- How to assign difficulty for program-based questions? (chain length / op types / other)
this is tbd... maybe we could decide how hard stuff is after doing some tests

- Should `n_steps` map directly to chain length?
?

## 9) Integration into CLI/pipeline
- Add a CLI flag for program-based generation?
yes

- Should program-based questions go through the same episode pipeline?
yes. only if validates. this is where we should think about better abstractions (question gen, validated, ready for episode, episode)

## 10) Testing expectations
- Minimal acceptance test for consolidation?
- Any regressions we must avoid?

i am really bad w/ tests.

---

# Follow-up questions (from assistant)

## A) “Data‑aware, difficult questions”
1) When you say “data‑aware,” do you mean: use real column names, reject ID‑like columns, check distributions, or something else?
yes. data aware meaning: the question somehow encodes real world knowledge - it is not pure abstaction.

2) What makes a question “difficult”? (more steps, group comparisons, statistical tests, multi‑step filters?)
all of the above.

3) Examples of **bad questions** we should explicitly filter?
... a bad question is procedural. but... i presume you are talking about how we communicate to the verbalizer on how this is done?

4) Do you want an explicit difficulty label, or just diversity?
for now, no label



## B) Mechanical vs verbalized questions
5) Should the mechanical question always be stored alongside the verbalized one?
yes, inside the same jsonl or whatever we use

6) Which question should the episode pipeline use by default?
default to no verbalizer?

7) Should hints always exist for verbalized questions? (you said yes)
yes

8) For evaluation, should the solver see hints or not?
add flag

## C) Filters (centralized in profiler)
9) Do you want the same filters applied to templates and programs?
what filters. i think both have some indepdnent filters.

10) Which filters are required? (ID‑like, near‑zero variance, tiny groups, heavy missingness, other?)
all of the above. but, should discuss specifics

11) What’s “strict” vs “lenient” for you?
?



## D) Consolidation boundaries
12) What must remain separate (template vs program vs LLM), and why?
maybe good to identify true overlap first.

13) Should we share one execution layer (LocalCSVAnalysisEnv) across templates + programs?
is there a reason not to? i lean towards yes

14) `synthetic/generator.py` executes + verbalizes + (optionally) validates; `validate_synthetic.py` validates again. Should validation live in one place only?
correct. it should be modular.

## E) Outputs and storage
15) Where should program questions live? (`data/questions_synthetic/` with a source field vs new folder)
?

16) Should schema be unified across all sources with a `source` field?
yes... but im not sure what this mean. we want unification and simplicfcation.

17) Should program be labeled as synthetic with a sub‑type (`procedural` vs `template`)?
yes, metadata tracking is very important. identify missing metadata project wide/discrepancies (spawn subagents)


## F) Manifest / caching
18) Should program‑based questions be stored in the manifest? (you said yes)
Manifest is there to prevent computational waste. This is super important for verbalization, less so for the raw code we generate. What do you think? I think centralizing this cache is very importaht

19) Fingerprint based on program spec, compiled code, or question text?
I think the code makes most sense. What do you think?

20) Should failures be recorded to?
What do you think?


## G) Difficulty metadata
21) For program‑based questions, should difficulty be chain length, op types, or empirical pass rate?
We are not there yet. Need to see failure modes?

22) Should `n_steps` map to chain length?
Yes, what do you think? How might they differ?

## H) CLI / pipeline integration
23) Should program generation be a new CLI flag or new subcommand?
cli flag

24) Should program questions flow through the same episode pipeline as templates?
yes, why would they not? curious

## I) Testing (simple)
25) Minimum acceptable test? (generate 1 program question + validate 1 episode?)
I think test should do for each new addition. I think our tests are generally weak. This probably requires an entire discussion? We need to test each part of pipeline, then full versions of the pipe.ien?

