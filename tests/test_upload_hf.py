import json

import pytest

from scripts import upload_hf


def _episode(episode_id: str, csv_source: str = "data/kaggle/sample/data.csv"):
    return {
        "episode_id": episode_id,
        "source": "template",
        "csv_source": csv_source,
        "verified": True,
        "question": {"difficulty": "EASY", "dataset": "sample"},
        "gold_trace": {"turns": [], "final_answer": 1, "success": True},
        "consistency_traces": [],
        "triangulation": {},
        "timing": {},
    }


def test_load_splits_and_stats(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    episode = _episode("ep1")
    (splits / "train.jsonl").write_text(json.dumps(episode) + "\n")
    (splits / "val.jsonl").write_text(json.dumps(_episode("ep2")) + "\n")
    (splits / "test.jsonl").write_text(json.dumps(_episode("ep3")) + "\n")

    dataset = upload_hf.load_splits(splits)
    stats = upload_hf.dataset_stats(dataset)

    assert set(dataset.keys()) == {"train", "val", "test"}
    assert stats["splits"] == {"train": 1, "val": 1, "test": 1}
    assert stats["total"] == 3
    assert stats["difficulties"] == {"EASY": 3}
    assert stats["sources"] == {"template": 3}
    assert stats["question_datasets"] == {"sample": 3}
    assert stats["csv_sources"] == 1
    assert json.loads(dataset["train"][0]["episode_json"])["episode_id"] == "ep1"


def test_load_splits_requires_all_default_splits(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(_episode("ep1")) + "\n")

    with pytest.raises(ValueError, match="Missing required split"):
        upload_hf.load_splits(splits)


def test_validate_csv_references(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, episode_id in [("train", "ep1"), ("val", "ep2"), ("test", "ep3")]:
        (splits / f"{split}.jsonl").write_text(
            json.dumps(_episode(episode_id)) + "\n"
        )
    csv_path = tmp_path / "data/kaggle/sample/data.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("a\n1\n")

    dataset = upload_hf.load_splits(splits)
    upload_hf.validate_csv_references(dataset, tmp_path / "data/kaggle")


def test_validate_csv_references_fails_missing_csv(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, episode_id in [("train", "ep1"), ("val", "ep2"), ("test", "ep3")]:
        (splits / f"{split}.jsonl").write_text(
            json.dumps(_episode(episode_id)) + "\n"
        )

    dataset = upload_hf.load_splits(splits)

    with pytest.raises(FileNotFoundError, match="episode CSV reference"):
        upload_hf.validate_csv_references(dataset, tmp_path / "data/kaggle")


def test_data_card_documents_prime_loader():
    card = upload_hf.data_card(
        "owner/csv-agent-episodes",
        {
            "splits": {"train": 1},
            "total": 1,
            "difficulties": {"EASY": 1},
            "sources": {"template": 1},
            "csv_sources": 1,
        },
        include_csvs=True,
    )

    assert 'dataset_name="owner/csv-agent-episodes"' in card
    assert "Raw CSV files included in this repo: `True`" in card
