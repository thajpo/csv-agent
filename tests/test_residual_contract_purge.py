from pathlib import Path

from src.datagen.question_gen import try_parse_questions


ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text()


def test_no_skip_existing_left_in_runtime_paths():
    targets = [
        "src/cli.py",
        "src/datagen/episode_gen.py",
        "src/datagen/validate_synthetic.py",
        "src/datagen/pipeline.py",
    ]
    for rel in targets:
        assert "skip_existing" not in _read(rel), f"Found skip_existing in {rel}"


def test_no_legacy_question_fallback_in_contract_surfaces():
    strict_targets = [
        "src/datagen/validate_question.py",
        "src/datagen/episode_gen.py",
        "src/datagen/shared/episode_factory.py",
    ]
    for rel in strict_targets:
        text = _read(rel)
        assert 'get("question",' not in text, f"Legacy question fallback left in {rel}"

    explorer_text = _read("src/gui/panels/explorer.py")
    trace_text = _read("src/gui/panels/trace.py")
    assert 'q.get("question")' not in explorer_text
    assert 'question.get("question")' not in trace_text


def test_llm_source_filters_use_llm_gen_contract():
    inspect_text = _read("src/utils/inspect.py")
    cli_text = _read("src/cli.py")

    assert '"llm_gen": lambda q: q.get("source") == "llm_gen"' in inspect_text
    assert (
        'return question_source == "llm_gen" or episode_source == "llm_gen"' in cli_text
    )
    assert 'q.get("source") == "llm"' not in inspect_text


def test_question_record_has_no_parallel_procedural_flag():
    text = _read("src/datagen/shared/questions_io.py")
    assert "is_procedural" not in text


def test_question_parser_normalizes_question_key_to_question_text():
    response = (
        '{"questions": ['
        '{"question": "What is the mean of col_a?", '
        '"hint": "Use mean", "n_steps": 1, "difficulty": "easy"}'
        "]}"
    )

    parsed = try_parse_questions(response)
    assert parsed is not None
    assert len(parsed) == 1
    assert parsed[0]["question_text"] == "What is the mean of col_a?"
    assert "question" not in parsed[0]
