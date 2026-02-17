from pathlib import Path
from typing import Literal

from src.core.config import Config, config

ArtifactSource = Literal["template", "procedural", "llm_gen"]


def _normalize_source(source: str) -> ArtifactSource:
    if source == "template":
        return "template"
    if source == "procedural":
        return "procedural"
    if source == "llm_gen":
        return "llm_gen"
    raise ValueError(
        f"Invalid artifact source: {source!r}. Expected one of: template, procedural, llm_gen"
    )


def resolve_questions_dir(source: str, cfg: Config = config) -> Path:
    normalized = _normalize_source(source)
    if normalized == "template":
        return Path(cfg.questions_template_dir)
    if normalized == "procedural":
        return Path(cfg.questions_procedural_dir)
    return Path(cfg.questions_llm_gen_dir)


def resolve_episodes_file(source: str, cfg: Config = config) -> Path:
    normalized = _normalize_source(source)
    if normalized == "template":
        return Path(cfg.episodes_template_jsonl)
    if normalized == "procedural":
        return Path(cfg.episodes_procedural_jsonl)
    return Path(cfg.episodes_llm_gen_jsonl)


def resolve_all_question_dirs(cfg: Config = config) -> list[Path]:
    return [
        resolve_questions_dir("template", cfg),
        resolve_questions_dir("procedural", cfg),
        resolve_questions_dir("llm_gen", cfg),
    ]


def resolve_all_episode_files(cfg: Config = config) -> list[Path]:
    return [
        resolve_episodes_file("template", cfg),
        resolve_episodes_file("procedural", cfg),
        resolve_episodes_file("llm_gen", cfg),
    ]
