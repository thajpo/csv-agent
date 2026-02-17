import json

import pytest

from src.core.paths import (
    resolve_all_episode_files,
    resolve_episodes_file,
    resolve_questions_dir,
)
from src.datagen import pipeline as pipeline_module
from src.utils.inspect import inspect_episodes
from src.utils.stats import collect_episodes_stats, collect_questions_stats


@pytest.mark.parametrize(
    "source, expected_questions_dir, expected_episodes_file",
    [
        ("template", "data/questions/template", "data/episodes/template.jsonl"),
        ("procedural", "data/questions/procedural", "data/episodes/procedural.jsonl"),
        ("llm_gen", "data/questions/llm_gen", "data/episodes/llm_gen.jsonl"),
    ],
)
def test_resolver_returns_source_scoped_paths(
    source: str, expected_questions_dir: str, expected_episodes_file: str
):
    assert str(resolve_questions_dir(source)) == expected_questions_dir
    assert str(resolve_episodes_file(source)) == expected_episodes_file


@pytest.mark.parametrize("invalid_source", ["all", "synthetic", "", "LLM"])
def test_resolver_fail_fast_on_invalid_source(invalid_source: str):
    with pytest.raises(ValueError, match="Invalid artifact source"):
        resolve_questions_dir(invalid_source)

    with pytest.raises(ValueError, match="Invalid artifact source"):
        resolve_episodes_file(invalid_source)


def test_stats_collectors_preserve_source_scoped_layout_semantics(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    template_q = tmp_path / "data/questions/template/demo/questions.json"
    procedural_q = tmp_path / "data/questions/procedural/demo/questions.json"
    llm_q = tmp_path / "data/questions/llm_gen/demo/questions.json"
    template_q.parent.mkdir(parents=True, exist_ok=True)
    procedural_q.parent.mkdir(parents=True, exist_ok=True)
    llm_q.parent.mkdir(parents=True, exist_ok=True)

    template_q.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "source": "template",
                        "difficulty": "EASY",
                        "template_name": "mean",
                    }
                ]
            }
        )
    )
    procedural_q.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "source": "procedural",
                        "difficulty": "MEDIUM",
                        "template_name": "group_by",
                    }
                ]
            }
        )
    )
    llm_q.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "source": "llm_gen",
                        "difficulty": "HARD",
                    }
                ]
            }
        )
    )

    template_episodes = tmp_path / "data/episodes/template.jsonl"
    llm_episodes = tmp_path / "data/episodes/llm_gen.jsonl"
    template_episodes.parent.mkdir(parents=True, exist_ok=True)
    template_episodes.write_text(
        json.dumps(
            {
                "verified": True,
                "csv_source": "data/kaggle/demo/data.csv",
                "question": {"difficulty": "EASY"},
            }
        )
        + "\n"
    )
    llm_episodes.write_text(
        json.dumps(
            {
                "verified": False,
                "csv_source": "data/kaggle/demo/data.csv",
                "question": {"difficulty": "HARD"},
            }
        )
        + "\n"
    )

    q_stats = collect_questions_stats()
    e_stats = collect_episodes_stats()

    assert q_stats["synthetic"]["total"] == 2
    assert q_stats["llm"]["total"] == 1
    assert q_stats["synthetic"]["by_dataset"]["demo"] == 1
    assert q_stats["llm"]["by_dataset"]["demo"] == 1

    assert e_stats["synthetic"]["total"] == 1
    assert e_stats["synthetic"]["verified"] == 1
    assert e_stats["llm"]["total"] == 1
    assert e_stats["llm"]["verified"] == 0


def test_inspect_episodes_default_candidate_order_is_preserved(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)

    template_file, _, llm_file = [
        tmp_path / p
        for p in [
            str(resolve_all_episode_files()[0]),
            str(resolve_all_episode_files()[1]),
            str(resolve_all_episode_files()[2]),
        ]
    ]
    template_file.parent.mkdir(parents=True, exist_ok=True)
    template_file.write_text(
        json.dumps(
            {
                "episode_id": "template-1",
                "verified": True,
                "question": {"question_text": "q"},
                "gold_trace": {"turns": [], "final_answer": 1},
            }
        )
        + "\n"
    )
    llm_file.write_text(
        json.dumps(
            {
                "episode_id": "llm-1",
                "verified": True,
                "question": {"question_text": "q"},
                "gold_trace": {"turns": [], "final_answer": 1},
            }
        )
        + "\n"
    )

    inspect_episodes(output=None, count=1, verified=False, show_hooks=False)
    out = capsys.readouterr().out

    assert "template.jsonl" in out


def test_pipeline_uses_resolver_paths_for_stage_arguments(monkeypatch):
    stage_calls: list[tuple[str, list[str]]] = []
    synth_calls: list[tuple[str, str, str, int | None, str]] = []

    def _fake_run_stage(name: str, cmd: list[str]) -> bool:
        stage_calls.append((name, cmd))
        return True

    def _fake_run_synthetic_stage(
        name: str,
        questions_dir: str,
        output_path: str,
        max_questions: int | None,
        source: str,
    ) -> bool:
        synth_calls.append((name, questions_dir, output_path, max_questions, source))
        return True

    monkeypatch.setattr(pipeline_module, "run_stage", _fake_run_stage)
    monkeypatch.setattr(
        pipeline_module, "run_synthetic_stage", _fake_run_synthetic_stage
    )

    rc = pipeline_module.main(mode="all", test=False, max_questions=5)

    assert rc == 0
    assert (
        "Stage 2a: Generate Template Episodes",
        "data/questions/template",
        "data/episodes/template.jsonl",
        5,
        "template",
    ) in synth_calls
    assert (
        "Stage 2b: Generate Procedural Episodes",
        "data/questions/procedural",
        "data/episodes/procedural.jsonl",
        5,
        "procedural",
    ) in synth_calls

    llm_stage = dict(stage_calls)["Stage 2c: Generate LLM Episodes"]
    assert "--questions-dir" in llm_stage
    assert "data/questions/llm_gen" in llm_stage
    assert "--output" in llm_stage
    assert "data/episodes/llm_gen.jsonl" in llm_stage
