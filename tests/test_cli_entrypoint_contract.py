import pytest
from pathlib import Path

from src.cli import (
    build_parser,
    cmd_generate_questions,
    cmd_generate_episodes,
    cmd_run,
    _show_episode_preflight,
    _fail_fast_on_existing_outputs,
    _fail_fast_on_unwriteable_targets,
    _run_fail_fast_preflight,
)


def _parse(argv: list[str]):
    parser = build_parser()
    return parser.parse_args(argv)


def test_legacy_synth_flag_hard_fails():
    with pytest.raises(SystemExit):
        _parse(["generate", "questions", "--synth"])


def test_conflicting_mode_flags_hard_fail():
    with pytest.raises(SystemExit):
        _parse(["run", "--template", "--llm-gen"])


def test_inspect_questions_requires_explicit_source():
    with pytest.raises(SystemExit):
        _parse(["inspect", "questions"])


def test_run_test_without_mode_hard_fails():
    with pytest.raises(SystemExit):
        _parse(["run", "--test"])


@pytest.mark.parametrize(
    "argv, expected_mode",
    [
        (["run", "--template"], "template"),
        (["run", "--procedural"], "procedural"),
        (["run", "--llm-gen"], "llm_gen"),
        (["run", "--all"], "all"),
        (["generate", "questions", "--all"], "all"),
        (["generate", "episodes", "--all"], "all"),
    ],
)
def test_canonical_modes_parse(argv, expected_mode):
    args = _parse(argv)
    assert args.mode == expected_mode


def test_generation_target_flags_parse():
    q_args = _parse(
        [
            "generate",
            "questions",
            "--llm-gen",
            "--num-questions",
            "40",
            "--even-difficulty",
        ]
    )
    assert q_args.num_questions == 40
    assert q_args.even_difficulty is True

    e_args = _parse(
        ["generate", "episodes", "--llm-gen", "--n-consistency", "2"]
    )
    assert e_args.n_consistency == 2


def test_generate_questions_fail_fast_on_existing_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = Path("data/questions/template/sample/questions.json")
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("[]")

    rc = cmd_generate_questions(
        mode="template",
        max_datasets=1,
        dry_run=False,
        regenerate=False,
    )
    assert rc == 2


def test_generate_episodes_all_preflights_all_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Conflict exists only for llm output; --all should still fail before any run starts.
    existing = Path("data/episodes/llm_gen.jsonl")
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text('{"existing": true}\n')

    rc = cmd_generate_episodes(
        mode="all",
        max_questions=1,
        dry_run=False,
        fresh=False,
    )
    assert rc == 2


def test_fail_fast_helper_respects_explicit_overwrite(tmp_path):
    existing = tmp_path / "out.jsonl"
    existing.write_text("{}\n")

    assert _fail_fast_on_existing_outputs(
        [existing], explicit_overwrite=False, command_name="x"
    )
    assert not _fail_fast_on_existing_outputs(
        [existing], explicit_overwrite=True, command_name="x"
    )


def test_source_scoped_question_preflight_does_not_cross_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    procedural_existing = Path("data/questions/procedural/sample/questions.json")
    procedural_existing.parent.mkdir(parents=True, exist_ok=True)
    procedural_existing.write_text("[]")

    should_abort = _run_fail_fast_preflight(
        mode="template",
        dry_run=False,
        explicit_overwrite=False,
        is_episode_generation=False,
    )
    assert not should_abort


def test_episode_preflight_counts_flat_list_question_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    questions = Path("data/questions/template/sample/questions.json")
    questions.parent.mkdir(parents=True, exist_ok=True)
    questions.write_text(
        '[{"id": "q1", "source": "template"}, {"id": "q2", "source": "template"}]'
    )
    episodes = Path("data/episodes/template.jsonl")
    episodes.parent.mkdir(parents=True, exist_ok=True)
    episodes.write_text(
        '{"source": "template", "question": {"id": "q1", "source": "template"}}\n'
    )

    total, existing = _show_episode_preflight(
        "template", Path("data/questions/template"), episodes
    )

    assert total == 2
    assert existing == 1


def test_fail_fast_on_legacy_layout_presence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data/questions_synthetic").mkdir(parents=True, exist_ok=True)

    should_abort = _run_fail_fast_preflight(
        mode="template",
        dry_run=False,
        explicit_overwrite=False,
        is_episode_generation=False,
    )
    assert should_abort


def test_run_all_preflight_aborts_before_pipeline_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = Path("data/episodes/llm_gen.jsonl")
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text('{"existing": true}\n')

    import src.datagen.pipeline as pipeline

    invoked = {"called": False}

    def _unexpected_pipeline_main(*, mode, test):
        invoked["called"] = True
        raise AssertionError("pipeline should not run when preflight fails")

    monkeypatch.setattr(pipeline, "main", _unexpected_pipeline_main)

    rc = cmd_run(mode="all", test=False, dry_run=False)

    assert rc == 2
    assert not invoked["called"]


def test_run_preflight_writeability_failure_aborts_before_pipeline_main(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    import src.cli as cli
    import src.datagen.pipeline as pipeline

    invoked = {"called": False}

    def _unexpected_pipeline_main(*, mode, test):
        invoked["called"] = True
        raise AssertionError("pipeline should not run when preflight fails")

    def _fail_llm_target(target: Path):
        if target == Path("data/episodes/llm_gen.jsonl"):
            return False, "permission denied"
        return True, None

    monkeypatch.setattr(pipeline, "main", _unexpected_pipeline_main)
    monkeypatch.setattr(cli, "_probe_target_writeability", _fail_llm_target)

    rc = cmd_run(mode="all", test=False, dry_run=False)

    assert rc == 2
    assert not invoked["called"]


def test_unwriteable_target_failure_message_includes_target_path(monkeypatch):
    import src.cli as cli

    emitted: list[str] = []

    def _capture(*args, **kwargs):
        emitted.append(" ".join(str(arg) for arg in args))

    def _always_fail(target: Path):
        return False, "permission denied"

    monkeypatch.setattr(cli.console, "print", _capture)
    monkeypatch.setattr(cli, "_probe_target_writeability", _always_fail)

    target = Path("data/episodes/llm_gen.jsonl")
    should_abort = _fail_fast_on_unwriteable_targets(
        [target], command_name="csvagent run --all"
    )

    assert should_abort
    assert any("data/episodes/llm_gen.jsonl" in line for line in emitted)
