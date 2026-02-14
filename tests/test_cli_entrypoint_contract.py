import pytest

from src.cli import build_parser


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
