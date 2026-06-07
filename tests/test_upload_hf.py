import json

from scripts import upload_hf


def test_load_splits_and_stats(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    episode = {
        "episode_id": "ep1",
        "csv_source": "data/kaggle/sample/data.csv",
        "verified": True,
        "question": {"difficulty": "EASY"},
        "gold_trace": {"turns": [], "final_answer": 1, "success": True},
        "consistency_traces": [],
        "triangulation": {},
        "timing": {},
    }
    (splits / "train.jsonl").write_text(json.dumps(episode) + "\n")
    (splits / "val.jsonl").write_text(json.dumps({**episode, "episode_id": "ep2"}) + "\n")

    dataset = upload_hf.load_splits(splits)
    stats = upload_hf.dataset_stats(dataset)

    assert set(dataset.keys()) == {"train", "val"}
    assert stats["splits"] == {"train": 1, "val": 1}
    assert stats["total"] == 2
    assert stats["difficulties"] == {"EASY": 2}
    assert stats["csv_sources"] == 1
    assert json.loads(dataset["train"][0]["episode_json"])["episode_id"] == "ep1"


def test_data_card_documents_prime_loader():
    card = upload_hf.data_card(
        "owner/csv-agent-episodes",
        {
            "splits": {"train": 1},
            "total": 1,
            "difficulties": {"EASY": 1},
            "csv_sources": 1,
        },
        include_csvs=True,
    )

    assert 'dataset_name="owner/csv-agent-episodes"' in card
    assert "Raw CSV files included in this repo: `True`" in card
