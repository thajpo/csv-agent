import pytest
from pathlib import Path

from src.cli import (
    build_parser,
    cmd_generate_questions,
    cmd_generate_episodes,
    cmd_run,
    _fail_fast_on_existing_outputs,
    _run_fail_fast_preflight,
    _source_specs_for_mode,
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


@pytest.mark.parametrize(
    "mode, expected_sources",
    [
        ("template", ["template"]),
        ("procedural", ["procedural"]),
        ("llm_gen", ["llm_gen"]),
        ("all", ["template", "procedural", "llm_gen"]),
    ],
)
def test_source_table_mode_selection(mode, expected_sources):
    selected = [spec["mode"] for spec in _source_specs_for_mode(mode)]
    assert selected == expected_sources


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


def test_run_all_preflight_blocks_pipeline_entrypoint_on_existing_output(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    existing = Path("data/episodes/llm_gen.jsonl")
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text('{"existing": true}\n')

    called = {"pipeline": False}

    def _fake_pipeline_main(**kwargs):
        called["pipeline"] = True
        return 0

    monkeypatch.setattr("src.datagen.pipeline.main", _fake_pipeline_main)

    rc = cmd_run(mode="all", test=False, dry_run=False)
    assert rc == 2
    assert not called["pipeline"]


def test_run_all_preflight_blocks_pipeline_entrypoint_on_unwritable_target(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    called = {"pipeline": False}

    def _fake_pipeline_main(**kwargs):
        called["pipeline"] = True
        return 0

    def _deny_write_probe(_path):
        return False, "permission denied"

    monkeypatch.setattr("src.datagen.pipeline.main", _fake_pipeline_main)
    monkeypatch.setattr("src.cli._probe_parent_write_access", _deny_write_probe)

    rc = cmd_run(mode="all", test=False, dry_run=False)
    assert rc == 2
    assert not called["pipeline"]
