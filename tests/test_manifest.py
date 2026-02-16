"""Tests for datagen manifest module."""

import tempfile
from pathlib import Path


from src.datagen.manifest import (
    DatagenManifest,
    ManifestEntry,
    compute_dataset_hash,
    compute_llm_fingerprint,
    compute_synthetic_fingerprint,
    normalize_question_text,
)


class TestNormalizeQuestionText:
    """Tests for question text normalization."""

    def test_lowercase(self):
        assert normalize_question_text("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert normalize_question_text("What's the mean?") == "whats the mean"

    def test_collapses_whitespace(self):
        assert normalize_question_text("hello   world") == "hello world"
        assert normalize_question_text("  hello  world  ") == "hello world"

    def test_combined_normalization(self):
        text = "What is the AVERAGE salary?!  "
        assert normalize_question_text(text) == "what is the average salary"

    def test_empty_string(self):
        assert normalize_question_text("") == ""

    def test_only_punctuation(self):
        assert normalize_question_text("???!!!") == ""


class TestComputeDatasetHash:
    """Tests for dataset hashing."""

    def test_hashes_file_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,3\n")
            path = f.name

        hash1 = compute_dataset_hash(path)
        assert len(hash1) == 16  # Truncated SHA256
        assert hash1.isalnum()

        Path(path).unlink()

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,3\n")
            path1 = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n4,5,6\n")
            path2 = f.name

        hash1 = compute_dataset_hash(path1)
        hash2 = compute_dataset_hash(path2)
        assert hash1 != hash2

        Path(path1).unlink()
        Path(path2).unlink()

    def test_same_content_same_hash(self):
        content = "a,b,c\n1,2,3\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            path1 = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            path2 = f.name

        hash1 = compute_dataset_hash(path1)
        hash2 = compute_dataset_hash(path2)
        assert hash1 == hash2

        Path(path1).unlink()
        Path(path2).unlink()


class TestComputeSyntheticFingerprint:
    """Tests for synthetic question fingerprinting."""

    def test_basic_fingerprint(self):
        fp = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=None,
            params=None,
            dataset_hash="abc123",
        )
        assert len(fp) == 16
        assert fp.isalnum()

    def test_different_code_different_fingerprint(self):
        fp1 = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=None,
            params=None,
            dataset_hash="abc123",
        )
        fp2 = compute_synthetic_fingerprint(
            template_code="df.std()",
            alternative_codes=None,
            params=None,
            dataset_hash="abc123",
        )
        assert fp1 != fp2

    def test_different_params_different_fingerprint(self):
        fp1 = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=None,
            params={"threshold": 0.5},
            dataset_hash="abc123",
        )
        fp2 = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=None,
            params={"threshold": 0.9},
            dataset_hash="abc123",
        )
        assert fp1 != fp2

    def test_different_dataset_different_fingerprint(self):
        fp1 = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=None,
            params=None,
            dataset_hash="abc123",
        )
        fp2 = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=None,
            params=None,
            dataset_hash="def456",
        )
        assert fp1 != fp2

    def test_alternatives_affect_fingerprint(self):
        fp1 = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=None,
            params=None,
            dataset_hash="abc123",
        )
        fp2 = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=["df.average()"],
            params=None,
            dataset_hash="abc123",
        )
        assert fp1 != fp2

    def test_deterministic(self):
        """Same inputs should produce same fingerprint."""
        fp1 = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=["df.average()"],
            params={"x": 1},
            dataset_hash="abc123",
        )
        fp2 = compute_synthetic_fingerprint(
            template_code="df.mean()",
            alternative_codes=["df.average()"],
            params={"x": 1},
            dataset_hash="abc123",
        )
        assert fp1 == fp2


class TestComputeLLMFingerprint:
    """Tests for LLM question fingerprinting."""

    def test_basic_fingerprint(self):
        fp = compute_llm_fingerprint(
            question_text="What is the average salary?",
            dataset_hash="abc123",
        )
        assert len(fp) == 16
        assert fp.isalnum()

    def test_normalization_applied(self):
        """Different surface forms but same normalized text should match."""
        fp1 = compute_llm_fingerprint(
            question_text="What is the average salary?",
            dataset_hash="abc123",
        )
        fp2 = compute_llm_fingerprint(
            question_text="what is the average salary",
            dataset_hash="abc123",
        )
        assert fp1 == fp2

    def test_different_questions_different_fingerprint(self):
        fp1 = compute_llm_fingerprint(
            question_text="What is the average salary?",
            dataset_hash="abc123",
        )
        fp2 = compute_llm_fingerprint(
            question_text="What is the median salary?",
            dataset_hash="abc123",
        )
        assert fp1 != fp2

    def test_different_dataset_different_fingerprint(self):
        fp1 = compute_llm_fingerprint(
            question_text="What is the average salary?",
            dataset_hash="abc123",
        )
        fp2 = compute_llm_fingerprint(
            question_text="What is the average salary?",
            dataset_hash="def456",
        )
        assert fp1 != fp2


class TestManifestEntry:
    """Tests for ManifestEntry dataclass."""

    def test_to_dict_excludes_none(self):
        entry = ManifestEntry(
            type="synthetic",
            fingerprint="abc123",
            status="success",
            dataset="sales.csv",
            timestamp="2026-01-04T10:00:00",
            template_name="bootstrap_ci",
            # episode_id and others are None
        )
        d = entry.to_dict()
        assert "episode_id" not in d
        assert "question_text" not in d
        assert d["template_name"] == "bootstrap_ci"

    def test_from_dict_roundtrip(self):
        entry = ManifestEntry(
            type="llm",
            fingerprint="def456",
            status="failure",
            dataset="hr.csv",
            timestamp="2026-01-04T11:00:00",
            question_text="What is the average?",
        )
        d = entry.to_dict()
        restored = ManifestEntry.from_dict(d)
        assert restored.type == entry.type
        assert restored.fingerprint == entry.fingerprint
        assert restored.status == entry.status
        assert restored.question_text == entry.question_text


class TestDatagenManifest:
    """Tests for DatagenManifest class."""

    def test_load_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()
            assert manifest.synthetic == {}
            assert manifest.llm == {}

    def test_append_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"

            # Append entries
            manifest = DatagenManifest(path)
            manifest.load()
            manifest.record_synthetic(
                fingerprint="fp1",
                status="success",
                dataset="sales.csv",
                template_name="mean_test",
                episode_id="ep_001",
            )
            manifest.record_llm(
                fingerprint="fp2",
                status="failure",
                dataset="hr.csv",
                question_text="What is the average?",
            )

            # Reload and verify
            manifest2 = DatagenManifest(path)
            manifest2.load()
            assert "fp1" in manifest2.synthetic
            assert manifest2.synthetic["fp1"].status == "success"
            assert manifest2.synthetic["fp1"].episode_id == "ep_001"
            assert "fp2" in manifest2.llm
            assert manifest2.llm["fp2"].status == "failure"

    def test_last_write_wins(self):
        """Later entries for same fingerprint should override earlier ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"

            manifest = DatagenManifest(path)
            manifest.load()

            # Record failure
            manifest.record_synthetic(
                fingerprint="fp1",
                status="failure",
                dataset="sales.csv",
                template_name="mean_test",
            )

            # Record success (retry)
            manifest.record_synthetic(
                fingerprint="fp1",
                status="success",
                dataset="sales.csv",
                template_name="mean_test",
                episode_id="ep_001",
            )

            # Reload and verify last write wins
            manifest2 = DatagenManifest(path)
            manifest2.load()
            assert manifest2.synthetic["fp1"].status == "success"
            assert manifest2.synthetic["fp1"].episode_id == "ep_001"

    def test_has_synthetic_excludes_failures_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()

            manifest.record_synthetic(
                fingerprint="fp_success",
                status="success",
                dataset="sales.csv",
                template_name="test1",
            )
            manifest.record_synthetic(
                fingerprint="fp_failure",
                status="failure",
                dataset="sales.csv",
                template_name="test2",
            )

            # Default: only success counts
            assert manifest.has_synthetic("fp_success") is True
            assert manifest.has_synthetic("fp_failure") is False

            # With include_failures: both count
            assert manifest.has_synthetic("fp_success", include_failures=True) is True
            assert manifest.has_synthetic("fp_failure", include_failures=True) is True

            # Non-existent
            assert manifest.has_synthetic("fp_missing") is False
            assert manifest.has_synthetic("fp_missing", include_failures=True) is False

    def test_has_llm_excludes_failures_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()

            manifest.record_llm(
                fingerprint="fp_success",
                status="success",
                dataset="hr.csv",
                question_text="Question 1",
                episode_id="ep_001",
            )
            manifest.record_llm(
                fingerprint="fp_failure",
                status="failure",
                dataset="hr.csv",
                question_text="Question 2",
            )

            assert manifest.has_llm("fp_success") is True
            assert manifest.has_llm("fp_failure") is False
            assert manifest.has_llm("fp_failure", include_failures=True) is True

    def test_compact(self):
        """Compact should remove duplicates from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()

            # Record same fingerprint twice
            manifest.record_synthetic(
                fingerprint="fp1",
                status="failure",
                dataset="sales.csv",
                template_name="test",
            )
            manifest.record_synthetic(
                fingerprint="fp1",
                status="success",
                dataset="sales.csv",
                template_name="test",
                episode_id="ep_001",
            )

            # File should have 2 lines
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            # Compact
            manifest.compact()

            # File should have 1 line
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1

            # Verify correct entry kept
            manifest2 = DatagenManifest(path)
            manifest2.load()
            assert manifest2.synthetic["fp1"].status == "success"

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()

            manifest.record_synthetic("fp1", "success", "s.csv", "t1", episode_id="e1")
            manifest.record_synthetic("fp2", "success", "s.csv", "t2", episode_id="e2")
            manifest.record_synthetic("fp3", "failure", "s.csv", "t3")
            manifest.record_llm("fp4", "success", "h.csv", "Q1", episode_id="e3")
            manifest.record_llm("fp5", "failure", "h.csv", "Q2")

            stats = manifest.stats()
            assert stats["synthetic_total"] == 3
            assert stats["synthetic_success"] == 2
            assert stats["synthetic_failure"] == 1
            assert stats["llm_total"] == 2
            assert stats["llm_success"] == 1
            assert stats["llm_failure"] == 1

    def test_handles_malformed_lines(self):
        """Malformed JSON lines should be skipped with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"

            # Write mix of valid and invalid lines
            with open(path, "w") as f:
                f.write(
                    '{"type": "synthetic", "fingerprint": "fp1", "status": "success", "dataset": "s.csv", "timestamp": "2026-01-04"}\n'
                )
                f.write("not valid json\n")
                f.write(
                    '{"type": "llm", "fingerprint": "fp2", "status": "success", "dataset": "h.csv", "timestamp": "2026-01-04"}\n'
                )

            manifest = DatagenManifest(path)
            manifest.load()  # Should not raise

            # Valid entries should be loaded
            assert "fp1" in manifest.synthetic
            assert "fp2" in manifest.llm

    def test_creates_parent_directory(self):
        """Append should create parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()
            manifest.record_synthetic(
                fingerprint="fp1",
                status="success",
                dataset="s.csv",
                template_name="test",
            )
            assert path.exists()

    def test_new_fields_model_and_elapsed(self):
        """Test that model and elapsed_seconds are stored and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()

            manifest.record_synthetic(
                fingerprint="fp1",
                status="success",
                dataset="sales.csv",
                template_name="test_template",
                model="gpt-4o",
                elapsed_seconds=12.5,
            )
            manifest.record_llm(
                fingerprint="fp2",
                status="success",
                dataset="hr.csv",
                question_text="What is the average?",
                model="claude-sonnet-4-20250514",
                n_consistency=5,
                elapsed_seconds=45.2,
            )

            # Reload and verify
            manifest2 = DatagenManifest(path)
            manifest2.load()

            assert manifest2.synthetic["fp1"].model == "gpt-4o"
            assert manifest2.synthetic["fp1"].elapsed_seconds == 12.5

            assert manifest2.llm["fp2"].model == "claude-sonnet-4-20250514"
            assert manifest2.llm["fp2"].n_consistency == 5
            assert manifest2.llm["fp2"].elapsed_seconds == 45.2

    def test_dataset_summary(self):
        """Test dataset_summary aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()

            manifest.record_synthetic(
                "fp1", "success", "sales.csv", "t1", model="gpt-4o"
            )
            manifest.record_synthetic(
                "fp2", "failure", "sales.csv", "t2", model="gpt-4o"
            )
            manifest.record_llm("fp3", "success", "sales.csv", "Q1", model="gpt-4o")
            manifest.record_llm(
                "fp4", "success", "hr.csv", "Q2", model="claude-sonnet-4-20250514"
            )

            summary = manifest.dataset_summary()

            assert "sales.csv" in summary
            assert summary["sales.csv"]["synthetic_success"] == 1
            assert summary["sales.csv"]["synthetic_failure"] == 1
            assert summary["sales.csv"]["llm_success"] == 1
            assert "gpt-4o" in summary["sales.csv"]["models"]

            assert "hr.csv" in summary
            assert summary["hr.csv"]["llm_success"] == 1
            assert "claude-sonnet-4-20250514" in summary["hr.csv"]["models"]

    def test_template_summary(self):
        """Test template_summary aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()

            manifest.record_synthetic("fp1", "success", "s.csv", "bootstrap_ci")
            manifest.record_synthetic("fp2", "success", "s.csv", "bootstrap_ci")
            manifest.record_synthetic("fp3", "failure", "s.csv", "bootstrap_ci")
            manifest.record_synthetic("fp4", "success", "s.csv", "anova_test")

            summary = manifest.template_summary()

            assert summary["bootstrap_ci"]["success"] == 2
            assert summary["bootstrap_ci"]["failure"] == 1
            assert summary["anova_test"]["success"] == 1
            assert summary["anova_test"]["failure"] == 0

    def test_model_summary(self):
        """Test model_summary aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            manifest = DatagenManifest(path)
            manifest.load()

            manifest.record_synthetic("fp1", "success", "s.csv", "t1", model="gpt-4o")
            manifest.record_synthetic("fp2", "success", "s.csv", "t2", model="gpt-4o")
            manifest.record_llm("fp3", "success", "h.csv", "Q1", model="gpt-4o")
            manifest.record_llm(
                "fp4", "success", "h.csv", "Q2", model="claude-sonnet-4-20250514"
            )

            summary = manifest.model_summary()

            assert summary["gpt-4o"]["synthetic"] == 2
            assert summary["gpt-4o"]["llm"] == 1
            assert summary["gpt-4o"]["total"] == 3

            assert summary["claude-sonnet-4-20250514"]["synthetic"] == 0
            assert summary["claude-sonnet-4-20250514"]["llm"] == 1
            assert summary["claude-sonnet-4-20250514"]["total"] == 1
