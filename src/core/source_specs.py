from typing import TypedDict


class SourceSpec(TypedDict):
    mode: str
    color: str
    question_output_dir_attr: str
    episode_output_file_attr: str
    question_stage: str
    episode_stage: str
    question_module: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    {
        "mode": "template",
        "color": "green",
        "question_output_dir_attr": "questions_template_dir",
        "episode_output_file_attr": "episodes_template_jsonl",
        "question_stage": "Stage 1a: Generate Template Questions",
        "episode_stage": "Stage 2a: Generate Template Episodes",
        "question_module": "src.datagen.synthetic.generator",
    },
    {
        "mode": "procedural",
        "color": "magenta",
        "question_output_dir_attr": "questions_procedural_dir",
        "episode_output_file_attr": "episodes_procedural_jsonl",
        "question_stage": "Stage 1b: Generate Procedural Questions",
        "episode_stage": "Stage 2b: Generate Procedural Episodes",
        "question_module": "src.datagen.synthetic.programs.runner",
    },
    {
        "mode": "llm_gen",
        "color": "blue",
        "question_output_dir_attr": "questions_llm_gen_dir",
        "episode_output_file_attr": "episodes_llm_gen_jsonl",
        "question_stage": "Stage 1c: Generate LLM Questions",
        "episode_stage": "Stage 2c: Generate LLM Episodes",
        "question_module": "src.datagen.question_gen",
    },
)


def source_specs_for_mode(mode: str) -> list[SourceSpec]:
    if mode == "all":
        return list(SOURCE_SPECS)
    return [spec for spec in SOURCE_SPECS if spec["mode"] == mode]
