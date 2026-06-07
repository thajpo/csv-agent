import json
from pathlib import Path

from src.utils.stats import collect_questions_stats


def test_collect_questions_stats_accepts_flat_list_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("data/questions/template/sample/questions.json")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "source": "template",
                    "dataset": "sample",
                    "question_text": "What is the mean?",
                    "difficulty": "EASY",
                    "template_name": "mean_template",
                }
            ]
        )
    )

    stats = collect_questions_stats()

    assert stats["synthetic"]["total"] == 1
    assert stats["synthetic"]["by_dataset"] == {"sample": 1}
    assert stats["synthetic"]["by_difficulty"]["EASY"] == 1
    assert stats["synthetic"]["by_template"]["mean_template"] == 1
