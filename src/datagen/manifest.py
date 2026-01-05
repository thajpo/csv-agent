"""
Datagen manifest for tracking processed questions.

The manifest enables incremental data generation by fingerprinting questions
and recording their processing status. This avoids re-running expensive
triangulation on questions that have already been processed.

Design principles:
- Manifest is for datagen, episodes are for training (separate concerns)
- JSONL append format for crash safety and concurrent writes
- Later entries override earlier ones (last-write-wins for duplicates)
- No backward compatibility - delete manifest to start fresh

Fingerprinting strategy:
- Synthetic: Hash template code + params + dataset content (deterministic input)
- LLM: Hash normalized question text + dataset content (non-deterministic generation)
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field, asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from csv_spec import hash_artifact

# Default manifest location
DEFAULT_MANIFEST_PATH = Path("data/datagen_manifest.jsonl")


@dataclass
class ManifestEntry:
    """A single entry in the manifest."""

    type: Literal["synthetic", "llm"]
    fingerprint: str
    status: Literal["success", "failure"]
    dataset: str
    timestamp: str
    episode_id: str | None = None
    # Synthetic-specific fields
    template_name: str | None = None
    template_params: dict | None = None
    # LLM-specific fields
    question_text: str | None = None
    n_consistency: int | None = None  # Number of consistency traces (LLM only)
    # Metadata (informational, not part of fingerprint)
    model: str | None = None  # Model used for triangulation/validation
    elapsed_seconds: float | None = None  # Time taken for processing

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> ManifestEntry:
        """Create from dict."""
        return cls(
            type=data["type"],
            fingerprint=data["fingerprint"],
            status=data["status"],
            dataset=data["dataset"],
            timestamp=data["timestamp"],
            episode_id=data.get("episode_id"),
            template_name=data.get("template_name"),
            template_params=data.get("template_params"),
            question_text=data.get("question_text"),
            n_consistency=data.get("n_consistency"),
            model=data.get("model"),
            elapsed_seconds=data.get("elapsed_seconds"),
        )


class DatagenManifest:
    """
    Tracks processed questions to enable incremental data generation.

    Uses JSONL append format for:
    - Crash safety (partial writes don't corrupt existing data)
    - Concurrent writes (multiple jobs can append safely)
    - Incremental progress (save after each question)

    On load, later entries override earlier ones for the same fingerprint.
    """

    def __init__(self, path: Path | str = DEFAULT_MANIFEST_PATH):
        self.path = Path(path)
        self.synthetic: dict[str, ManifestEntry] = {}
        self.llm: dict[str, ManifestEntry] = {}
        self._loaded = False

    def load(self) -> None:
        """
        Load manifest from JSONL file.

        Deduplicates by fingerprint (last entry wins).
        Creates empty manifest if file doesn't exist.
        """
        self.synthetic = {}
        self.llm = {}

        if not self.path.exists():
            self._loaded = True
            return

        with open(self.path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = ManifestEntry.from_dict(data)
                    if entry.type == "synthetic":
                        self.synthetic[entry.fingerprint] = entry
                    else:
                        self.llm[entry.fingerprint] = entry
                except (json.JSONDecodeError, KeyError) as e:
                    # Skip malformed entries but warn
                    print(
                        f"Warning: Skipping malformed manifest entry at line {line_num}: {e}"
                    )

        self._loaded = True

    def append(self, entry: ManifestEntry) -> None:
        """
        Append a single entry to the manifest file.

        Also updates in-memory state.
        """
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        with open(self.path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        # Update in-memory state
        if entry.type == "synthetic":
            self.synthetic[entry.fingerprint] = entry
        else:
            self.llm[entry.fingerprint] = entry

    def get_synthetic(self, fingerprint: str) -> ManifestEntry | None:
        """Get synthetic entry by fingerprint."""
        return self.synthetic.get(fingerprint)

    def get_llm(self, fingerprint: str) -> ManifestEntry | None:
        """Get LLM entry by fingerprint."""
        return self.llm.get(fingerprint)

    def has_synthetic(self, fingerprint: str, include_failures: bool = False) -> bool:
        """
        Check if synthetic fingerprint exists in manifest.

        Args:
            fingerprint: The fingerprint to check
            include_failures: If False, only return True for successful entries
        """
        entry = self.synthetic.get(fingerprint)
        if entry is None:
            return False
        if include_failures:
            return True
        return entry.status == "success"

    def has_llm(self, fingerprint: str, include_failures: bool = False) -> bool:
        """
        Check if LLM fingerprint exists in manifest.

        Args:
            fingerprint: The fingerprint to check
            include_failures: If False, only return True for successful entries
        """
        entry = self.llm.get(fingerprint)
        if entry is None:
            return False
        if include_failures:
            return True
        return entry.status == "success"

    def record_synthetic(
        self,
        fingerprint: str,
        status: Literal["success", "failure"],
        dataset: str,
        template_name: str,
        template_params: dict | None = None,
        episode_id: str | None = None,
        model: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Record a synthetic question result."""
        entry = ManifestEntry(
            type="synthetic",
            fingerprint=fingerprint,
            status=status,
            dataset=dataset,
            timestamp=datetime.now().isoformat(),
            episode_id=episode_id,
            template_name=template_name,
            template_params=template_params,
            model=model,
            elapsed_seconds=elapsed_seconds,
        )
        self.append(entry)

    def record_llm(
        self,
        fingerprint: str,
        status: Literal["success", "failure"],
        dataset: str,
        question_text: str,
        episode_id: str | None = None,
        model: str | None = None,
        n_consistency: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Record an LLM question result."""
        entry = ManifestEntry(
            type="llm",
            fingerprint=fingerprint,
            status=status,
            dataset=dataset,
            timestamp=datetime.now().isoformat(),
            episode_id=episode_id,
            question_text=question_text[:200],  # Truncate for readability
            n_consistency=n_consistency,
            model=model,
            elapsed_seconds=elapsed_seconds,
        )
        self.append(entry)

    def compact(self) -> None:
        """
        Rewrite manifest file removing duplicates.

        Call this periodically if the file grows too large from repeated retries.
        """
        if not self._loaded:
            self.load()

        # Write all unique entries
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            for entry in self.synthetic.values():
                f.write(json.dumps(entry.to_dict()) + "\n")
            for entry in self.llm.values():
                f.write(json.dumps(entry.to_dict()) + "\n")

    def stats(self) -> dict:
        """Return summary statistics."""
        synthetic_success = sum(
            1 for e in self.synthetic.values() if e.status == "success"
        )
        synthetic_failure = sum(
            1 for e in self.synthetic.values() if e.status == "failure"
        )
        llm_success = sum(1 for e in self.llm.values() if e.status == "success")
        llm_failure = sum(1 for e in self.llm.values() if e.status == "failure")

        return {
            "synthetic_total": len(self.synthetic),
            "synthetic_success": synthetic_success,
            "synthetic_failure": synthetic_failure,
            "llm_total": len(self.llm),
            "llm_success": llm_success,
            "llm_failure": llm_failure,
        }

    def dataset_summary(self) -> dict[str, dict]:
        """
        Aggregate statistics by dataset.

        Returns:
            {dataset: {"synthetic_success": N, "synthetic_failure": N,
                       "llm_success": N, "llm_failure": N, "models": [...]}}
        """
        summary: dict[str, dict] = {}

        for entry in self.synthetic.values():
            ds = entry.dataset
            if ds not in summary:
                summary[ds] = {
                    "synthetic_success": 0,
                    "synthetic_failure": 0,
                    "llm_success": 0,
                    "llm_failure": 0,
                    "models": set(),
                }
            if entry.status == "success":
                summary[ds]["synthetic_success"] += 1
            else:
                summary[ds]["synthetic_failure"] += 1
            if entry.model:
                summary[ds]["models"].add(entry.model)

        for entry in self.llm.values():
            ds = entry.dataset
            if ds not in summary:
                summary[ds] = {
                    "synthetic_success": 0,
                    "synthetic_failure": 0,
                    "llm_success": 0,
                    "llm_failure": 0,
                    "models": set(),
                }
            if entry.status == "success":
                summary[ds]["llm_success"] += 1
            else:
                summary[ds]["llm_failure"] += 1
            if entry.model:
                summary[ds]["models"].add(entry.model)

        # Convert sets to sorted lists for JSON serialization
        for ds in summary:
            summary[ds]["models"] = sorted(summary[ds]["models"])

        return summary

    def template_summary(self) -> dict[str, dict]:
        """
        Aggregate synthetic statistics by template.

        Returns:
            {template_name: {"success": N, "failure": N}}
        """
        summary: dict[str, dict] = {}

        for entry in self.synthetic.values():
            template = entry.template_name or "unknown"
            if template not in summary:
                summary[template] = {"success": 0, "failure": 0}
            if entry.status == "success":
                summary[template]["success"] += 1
            else:
                summary[template]["failure"] += 1

        return summary

    def model_summary(self) -> dict[str, dict]:
        """
        Aggregate statistics by model.

        Returns:
            {model: {"synthetic": N, "llm": N, "total": N}}
        """
        summary: dict[str, dict] = {}

        for entry in self.synthetic.values():
            model = entry.model or "unknown"
            if model not in summary:
                summary[model] = {"synthetic": 0, "llm": 0, "total": 0}
            summary[model]["synthetic"] += 1
            summary[model]["total"] += 1

        for entry in self.llm.values():
            model = entry.model or "unknown"
            if model not in summary:
                summary[model] = {"synthetic": 0, "llm": 0, "total": 0}
            summary[model]["llm"] += 1
            summary[model]["total"] += 1

        return summary

    def detect_template_changes(
        self,
        dataset_hash: str,
    ) -> dict[str, str]:
        """
        Detect which templates have changed since entries were recorded.

        Compares current template code fingerprints against stored fingerprints.

        Args:
            dataset_hash: Hash of the dataset to check against

        Returns:
            {template_name: "new" | "changed"} for templates that need re-running
        """
        from src.datagen.synthetic.templates import ALL_TEMPLATES

        changes: dict[str, str] = {}

        # Build map of template_name -> set of fingerprints we have
        existing_fingerprints: dict[str, set[str]] = {}
        for entry in self.synthetic.values():
            if entry.template_name:
                if entry.template_name not in existing_fingerprints:
                    existing_fingerprints[entry.template_name] = set()
                existing_fingerprints[entry.template_name].add(entry.fingerprint)

        # Check each template
        for template in ALL_TEMPLATES:
            # Compute current fingerprint (with empty params as baseline)
            current_fp = compute_synthetic_fingerprint(
                template_code=template.code_template,
                alternative_codes=template.alternative_code_templates,
                params={},
                dataset_hash=dataset_hash,
            )

            if template.name not in existing_fingerprints:
                changes[template.name] = "new"
            elif current_fp not in existing_fingerprints[template.name]:
                # Fingerprint changed - template code must have changed
                changes[template.name] = "changed"

        return changes


# =============================================================================
# Fingerprint computation functions
# =============================================================================


def compute_dataset_hash(csv_path: str | Path) -> str:
    """
    Compute hash of entire CSV file content.

    This ensures any change to the dataset invalidates cached results.
    """
    path = Path(csv_path)
    with open(path, "rb") as f:
        content = f.read()
    return sha256(content).hexdigest()[:16]


def normalize_question_text(text: str) -> str:
    """
    Normalize question text for consistent fingerprinting.

    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_synthetic_fingerprint(
    template_code: str,
    alternative_codes: list[str] | None,
    params: dict | None,
    dataset_hash: str,
) -> str:
    """
    Compute fingerprint for a synthetic question.

    Includes template code, alternatives, params, and dataset hash.
    Any change to the template or dataset invalidates the fingerprint.
    """
    return hash_artifact(
        {
            "template_code": template_code,
            "alternative_codes": alternative_codes or [],
            "params": params or {},
            "dataset_hash": dataset_hash,
        }
    )


def compute_llm_fingerprint(
    question_text: str,
    dataset_hash: str,
) -> str:
    """
    Compute fingerprint for an LLM-generated question.

    Uses normalized question text to handle minor phrasing variations.
    """
    return hash_artifact(
        {
            "question_text": normalize_question_text(question_text),
            "dataset_hash": dataset_hash,
        }
    )


def get_template_by_name(template_name: str):
    """
    Look up a template by name from ALL_TEMPLATES.

    Returns None if not found.
    """
    from src.datagen.synthetic.templates import ALL_TEMPLATES

    for template in ALL_TEMPLATES:
        if template.name == template_name:
            return template
    return None


def compute_synthetic_fingerprint_from_question(
    question: dict,
    dataset_hash: str,
) -> str | None:
    """
    Compute fingerprint for a synthetic question dict.

    Looks up the template by name to get the actual code for hashing.
    Returns None if template not found.
    """
    template_name = question.get("template_name")
    if not template_name:
        return None

    template = get_template_by_name(template_name)
    if not template:
        return None

    return compute_synthetic_fingerprint(
        template_code=template.code_template,
        alternative_codes=template.alternative_code_templates,
        params=question.get("template_params"),
        dataset_hash=dataset_hash,
    )
