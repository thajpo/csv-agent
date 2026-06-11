from collections import Counter
from datetime import datetime

from csv_spec import EpisodeJSONL
from src.training.split_episodes import split_episodes


def _episode(episode_id: str, source: str) -> EpisodeJSONL:
    return EpisodeJSONL(
        episode_id=episode_id,
        timestamp=datetime.now(),
        csv_source="data/kaggle/sample/data.csv",
        question={
            "id": episode_id,
            "source": source,
            "dataset": "sample",
            "question_text": "Return 1",
            "difficulty": "EASY",
        },
        gold_trace={
            "turns": [],
            "final_answer": 1,
            "final_answer_hash": "answer-hash",
            "success": True,
        },
        consistency_traces=[],
        verified=True,
        triangulation={
            "n_consistency_runs": 0,
            "n_consistency_succeeded": 0,
            "majority_answer_hash": None,
            "majority_count": 0,
            "gold_matches_majority": True,
        },
        timing={
            "gold_elapsed": 0.0,
            "consistency_elapsed": [],
            "total_elapsed": 0.0,
            "avg_elapsed": 0.0,
        },
        source=source,
    )


def _source_counts(episodes: list[EpisodeJSONL]) -> Counter:
    return Counter(ep.source for ep in episodes)


def test_split_episodes_respects_source_stratification():
    episodes = [
        _episode(f"template-{i}", "template") for i in range(4)
    ] + [
        _episode(f"llm-{i}", "llm_gen") for i in range(4)
    ]

    train, val, test = split_episodes(
        episodes,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        stratify_by="source",
        seed=1,
    )

    assert _source_counts(train) == {"template": 2, "llm_gen": 2}
    assert _source_counts(val) == {"template": 1, "llm_gen": 1}
    assert _source_counts(test) == {"template": 1, "llm_gen": 1}
