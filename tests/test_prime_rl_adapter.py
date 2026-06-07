import json
from pathlib import Path

import pytest
import verifiers as vf
from datasets import Dataset

from csv_spec import hash_artifact
import src.training.rl_env as rl_env
from src.training.rl_env import CSVAgentRLEnv, episodes_to_dataset
from src.training.rl_rubric import CSVAgentRubric


def _episode(csv_path: Path, answer=3) -> dict:
    answer_hash = hash_artifact(answer)
    return {
        "episode_id": "ep_test",
        "csv_source": str(csv_path),
        "verified": True,
        "question": {
            "id": "q_test",
            "question_text": "What is the sum of value?",
            "difficulty": "EASY",
            "n_steps": 1,
            "ground_truth": answer,
            "ground_truth_hash": answer_hash,
            "ground_truth_hashes": [answer_hash],
        },
        "gold_trace": {
            "turns": [
                {
                    "turn_index": 0,
                    "code": "result = df['value'].sum()\nsubmit(result)",
                    "execution": {
                        "success": True,
                        "stdout": "submitted",
                        "stderr": "",
                        "hooks": [
                            {
                                "value_hash": "hook123",
                                "code_line": "result = df['value'].sum()",
                                "value": answer,
                            }
                        ],
                        "submitted_answer": answer,
                    },
                }
            ],
            "final_answer": answer,
            "final_answer_hash": answer_hash,
            "success": True,
        },
    }


def test_episodes_to_dataset_uses_canonical_episode_fields(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("value\n1\n2\n")

    dataset = episodes_to_dataset([_episode(csv_path)], include_data_overview=True)

    assert len(dataset) == 1
    row = dataset[0]
    assert row["answer"] == hash_artifact(3)
    assert row["task"] == "EASY"
    assert row["prompt"][0]["role"] == "system"
    assert "DATA OVERVIEW" in row["prompt"][0]["content"]
    assert row["prompt"][1] == {
        "role": "user",
        "content": "What is the sum of value?",
    }

    info = json.loads(row["info"])
    assert info["episode_id"] == "ep_test"
    assert info["csv_source"] == str(csv_path)
    assert info["expected_answer"] == 3
    assert info["expected_answer_hash"] == hash_artifact(3)
    assert info["expected_answer_hashes"] == [hash_artifact(3)]
    assert info["expected_hooks"][0]["value_hash"] == "hook123"


def test_verifiers_can_load_csv_agent_environment_by_id(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("value\n1\n2\n")
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps(_episode(csv_path)) + "\n")

    env = vf.load_environment("csv-agent", episodes_path=str(episodes_path))

    assert isinstance(env, CSVAgentRLEnv)
    assert len(env.get_dataset()) == 1
    assert env.get_dataset()[0]["answer"] == hash_artifact(3)


def test_environment_can_load_hf_dataset_split(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("value\n1\n2\n")
    episode = _episode(csv_path)

    def fake_load_dataset(path, **kwargs):
        assert path == "owner/csv-agent-episodes"
        assert kwargs["split"] == "train"
        return Dataset.from_list(
            [
                {
                    "episode_id": episode["episode_id"],
                    "csv_source": episode["csv_source"],
                    "difficulty": "EASY",
                    "episode_json": json.dumps(episode),
                }
            ]
        )

    monkeypatch.setattr(rl_env, "load_dataset", fake_load_dataset)
    env = CSVAgentRLEnv(
        dataset_name="owner/csv-agent-episodes",
        dataset_split="train",
        download_csvs=False,
    )

    assert len(env.get_dataset()) == 1
    assert env.get_dataset()[0]["answer"] == hash_artifact(3)


def test_eval_dataset_defaults_to_primary_dataset(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("value\n1\n2\n")
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps(_episode(csv_path)) + "\n")

    env = CSVAgentRLEnv(episodes_path=str(episodes_path), include_data_overview=False)

    assert len(env.get_dataset()) == 1
    assert len(env.get_eval_dataset()) == 1
    assert env.get_eval_dataset()[0]["info"] == env.get_dataset()[0]["info"]


def test_environment_requires_local_or_hf_episode_source():
    with pytest.raises(ValueError, match="episodes_path or dataset_name"):
        CSVAgentRLEnv()


def test_environment_rejects_unknown_sandbox_backend(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("value\n1\n2\n")
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps(_episode(csv_path)) + "\n")

    with pytest.raises(ValueError, match="sandbox_backend"):
        CSVAgentRLEnv(
            episodes_path=str(episodes_path),
            sandbox_backend="unknown",
        )


def test_adapter_can_rebase_absolute_csv_sources(tmp_path):
    csv_root = tmp_path / "data" / "kaggle"
    rebased_csv = csv_root / "sample-dataset" / "data.csv"
    rebased_csv.parent.mkdir(parents=True)
    rebased_csv.write_text("value\n1\n2\n")
    episode = _episode(Path("/old/workspace/data/kaggle/sample-dataset/data.csv"))

    dataset = episodes_to_dataset(
        [episode],
        csv_root=str(csv_root),
        include_data_overview=False,
    )

    info = json.loads(dataset[0]["info"])
    assert info["csv_source"] == str(rebased_csv)


def test_rubric_scores_submitted_answer_and_hook_state():
    rubric = CSVAgentRubric(hook_reward=0.1, final_reward=1.0)
    answer_hash = hash_artifact(3)
    state = {
        "info": {
            "expected_answer": 3,
            "expected_answer_hash": answer_hash,
            "expected_answer_hashes": [answer_hash],
            "expected_hooks": [{"value_hash": "hook123"}],
        },
        "submitted_answer": 3,
        "captured_hooks": [{"value_hash": "hook123"}],
        "completion": "",
    }

    assert rubric.compute_reward(state) == pytest.approx(1.1)


@pytest.mark.asyncio
async def test_environment_stops_after_submission_flag(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("value\n1\n2\n")
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps(_episode(csv_path)) + "\n")
    env = CSVAgentRLEnv(episodes_path=str(episodes_path))

    assert await env.submitted_answer({"terminal": True})
    assert not await env.submitted_answer({})


@pytest.mark.asyncio
async def test_setup_state_accepts_json_encoded_info(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("value\n1\n2\n")
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps(_episode(csv_path)) + "\n")

    class DummyCSVAnalysisEnv:
        def __init__(self, csv_path, **kwargs):
            self.csv_path = csv_path

        async def setup_state(self, state):
            return {"sandbox_id": "dummy", "python_state": {}}

    monkeypatch.setattr(rl_env, "LocalCSVAnalysisEnv", DummyCSVAnalysisEnv)

    env = CSVAgentRLEnv(episodes_path=str(episodes_path), include_data_overview=False)
    state = {"info": env.get_dataset()[0]["info"]}
    await env.setup_state(state)

    assert state["info"]["csv_source"] == str(csv_path)
    assert state["csv_env"].csv_path == str(csv_path)
    assert state["sandbox_state"]["sandbox_id"] == "dummy"


@pytest.mark.asyncio
async def test_setup_state_can_select_prime_sandbox_backend(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("value\n1\n2\n")
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps(_episode(csv_path)) + "\n")

    class DummyPrimeSandboxEnv:
        def __init__(self, csv_path, pip_install_packages):
            self.csv_path = csv_path
            self.pip_install_packages = pip_install_packages

        async def setup_state(self, state):
            return {"sandbox_id": "prime-dummy", "python_state": {}}

    monkeypatch.setattr(rl_env, "PrimeSandboxCSVAnalysisEnv", DummyPrimeSandboxEnv)

    env = CSVAgentRLEnv(
        episodes_path=str(episodes_path),
        include_data_overview=False,
        sandbox_backend="prime",
        sandbox_pip_install_packages="pandas",
    )
    state = {"info": env.get_dataset()[0]["info"]}
    await env.setup_state(state)

    assert state["csv_env"].csv_path == str(csv_path)
    assert state["csv_env"].pip_install_packages == "pandas"
    assert state["sandbox_state"]["sandbox_id"] == "prime-dummy"
