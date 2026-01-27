"""Dataset metadata loading utilities.

Centralizes dataset name and description loading from CSV paths.
Provides consistent behavior across LLM and synthetic generation pipelines.
"""

from pathlib import Path
from typing import Optional
import json


def load_dataset_meta(csv_path: str | Path) -> tuple[str, str]:
    """Load dataset name and description from a CSV path.

    Dataset name:
    - If filename is `data.csv`, use parent folder name.
    - Otherwise, use the file stem.

    Dataset description:
    - Read from `meta.json` in same folder, else `{dataset}.meta.json`
    - Fields: `description`, `subtitle`, `title` (first non-empty)
    - Empty string if unavailable

    If description is empty, LLM question generation should synthesize a
    short description from `data_overview` instead of skipping the dataset.

    Args:
        csv_path: Absolute path to the CSV file.

    Returns:
        (dataset_name, dataset_description)
    """
    csv_path = Path(csv_path)

    # Dataset name
    if csv_path.name == "data.csv":
        dataset_name = csv_path.parent.name
    else:
        dataset_name = csv_path.stem

    # Dataset description
    dataset_description = ""

    # Try meta.json in same folder
    meta_path = csv_path.parent / "meta.json"
    if not meta_path.exists():
        meta_path = csv_path.with_suffix(".meta.json")

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            dataset_description = (
                meta.get("description")
                or meta.get("subtitle")
                or meta.get("title")
                or ""
            )
        except (json.JSONDecodeError, IOError):
            dataset_description = ""

    return dataset_name, dataset_description


def generate_description_from_overview(data_overview: str) -> str:
    """Generate a short dataset description from data_overview string.

    This is a placeholder implementation. In practice, use the LLM
    to generate a concise 1-2 sentence description.

    Args:
        data_overview: Programmatic data overview string.

    Returns:
        A short description string.
    """
    # Simple heuristic: take first 3 sentences
    sentences = data_overview.split(".")
    return ". ".join(sentences[:3]).strip()
