"""Contract tests for Git- versus Hugging-Face-owned data."""

from pathlib import Path
import subprocess
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_git_tracks_only_deterministic_data_fixtures():
    result = _git("ls-files", "data")

    assert result.returncode == 0, result.stderr
    tracked = [line for line in result.stdout.splitlines() if line]
    assert tracked
    assert all(path.startswith("data/fixtures/") for path in tracked), tracked


def test_generated_data_and_secret_env_files_are_ignored():
    ignored_paths = [
        "data/kaggle/example/data.csv",
        "data/episodes/template.jsonl",
        "data/hf/template/train.parquet",
        "configs/prime_rl/secrets.env",
    ]
    for path in ignored_paths:
        result = _git("check-ignore", "--no-index", "-q", path)
        assert result.returncode == 0, f"Expected Git to ignore {path}"

    fixture = _git("check-ignore", "--no-index", "-q", "data/fixtures/example/data.csv")
    assert fixture.returncode == 1, "data/fixtures/** must remain trackable"


def test_template_dataset_snapshot_is_pinned():
    config_path = REPO_ROOT / "configs/datasets/template.toml"
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config == {
        "repo": "ThaJpo/csv-agent-template-episodes",
        "revision": "e19fadf8d713c5afb7fe1476e2160b9bece1233a",
        "train_split": "train",
        "validation_split": "val",
        "test_split": "test",
    }
