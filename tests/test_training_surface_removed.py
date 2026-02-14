from importlib.util import find_spec
from pathlib import Path


def test_training_entrypoint_removed():
    assert Path("entrypoints/sft_train.py").exists() is False
    assert Path("scripts/train_rl.sh").exists() is False


def test_training_modules_removed():
    assert find_spec("src.training") is None
