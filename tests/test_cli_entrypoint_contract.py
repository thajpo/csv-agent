import pytest
from pathlib import Path

from src.cli import (
    build_parser,
    cmd_generate_questions,
    cmd_generate_episodes,
    _fail_fast_on_existing_outputs,
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


def test_generate_questions_fail_fast_on_existing_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = Path("data/questions_synthetic/sample/questions.json")
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
    existing = Path("data/episodes/episodes_llm.jsonl")
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
