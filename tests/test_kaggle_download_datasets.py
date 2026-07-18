"""Tests for Kaggle source-manifest parsing without Kaggle credentials."""

import json

import pytest

from scripts.kaggle.download_datasets import load_curated_list


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (["owner/one", "owner/two"], ["owner/one", "owner/two"]),
        ({"datasets": ["owner/one"]}, ["owner/one"]),
        (
            [
                {"ref": "owner/one", "slug": "owner_one"},
                {"ref": "owner/two", "slug": "owner_two"},
            ],
            ["owner/one", "owner/two"],
        ),
    ],
)
def test_load_curated_list_accepts_supported_manifests(tmp_path, payload, expected):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))

    assert load_curated_list(manifest) == expected


@pytest.mark.parametrize("payload", [{}, [{"slug": "missing_ref"}], [123]])
def test_load_curated_list_rejects_invalid_entries(tmp_path, payload):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Invalid format"):
        load_curated_list(manifest)
