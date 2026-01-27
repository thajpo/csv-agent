"""
Episode contract tests - validate that episodes meet downstream requirements
regardless of how they were generated (template, program, or LLM).

These tests define the "API boundary" between data generation and training.
If an episode passes these tests, it can be used for SFT, RL, and PRM training.

Run with: uv run pytest tests/test_episode_contract.py -v
"""

import json
from datetime import datetime
from pathlib import Path

import pytest


FIXTURE_PATH = Path("data/fixtures/sample_episode.json")


@pytest.fixture
def episode():
    """Load sample episode fixture."""
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture
def verified_episode(episode):
    """Episode that passed triangulation."""
    if not episode.get("verified"):
        pytest.skip("Fixture episode is not verified")
    return episode


# ============= Schema Validation =============


class TestEpisodeSchema:
    """Validate episode structure matches documented schema (docs/episode_schema.md)."""

    def test_required_top_level_fields(self, episode):
        """Episode must have all required top-level fields."""
        required = {
            "episode_id",
            "timestamp",
            "csv_source",
            "question",
            "gold_trace",
            "consistency_traces",
            "verified",
            "triangulation",
            "timing",
        }
        missing = required - set(episode.keys())
        assert not missing, f"Missing required fields: {missing}"

    def test_question_structure(self, episode):
        """Question must have required fields."""
        q = episode["question"]
        required = {"id", "question_text", "difficulty", "n_steps"}
        missing = required - set(q.keys())
        assert not missing, f"Question missing fields: {missing}"

    def test_question_has_ground_truth_hash(self, episode):
        """Question must have at least one ground truth hash for validation."""
        q = episode["question"]
        has_single = "ground_truth_hash" in q and q["ground_truth_hash"]
        has_multi = "ground_truth_hashes" in q and q["ground_truth_hashes"]
        assert has_single or has_multi, "Question needs ground_truth_hash or ground_truth_hashes"

    def test_gold_trace_structure(self, episode):
        """Gold trace must have turns, final_answer, and success flag."""
        trace = episode["gold_trace"]
        assert "turns" in trace, "gold_trace missing 'turns'"
        assert "final_answer" in trace, "gold_trace missing 'final_answer'"
        assert "final_answer_hash" in trace, "gold_trace missing 'final_answer_hash'"
        assert "success" in trace, "gold_trace missing 'success'"
        assert len(trace["turns"]) >= 1, "gold_trace must have at least one turn"

    def test_turn_structure(self, episode):
        """Each turn must have code and execution result."""
        for i, turn in enumerate(episode["gold_trace"]["turns"]):
            assert "turn_index" in turn, f"Turn {i} missing 'turn_index'"
            assert "code" in turn, f"Turn {i} missing 'code'"
            assert "execution" in turn, f"Turn {i} missing 'execution'"

    def test_execution_structure(self, episode):
        """Execution result must have success, stdout, stderr."""
        for i, turn in enumerate(episode["gold_trace"]["turns"]):
            exec_result = turn["execution"]
            assert "success" in exec_result, f"Turn {i} execution missing 'success'"
            assert "stdout" in exec_result, f"Turn {i} execution missing 'stdout'"
            assert "stderr" in exec_result, f"Turn {i} execution missing 'stderr'"

    def test_timestamp_is_valid_iso(self, episode):
        """Timestamp must be valid ISO format."""
        try:
            datetime.fromisoformat(episode["timestamp"])
        except ValueError as e:
            pytest.fail(f"Invalid timestamp format: {e}")


# ============= SFT Derivability =============


class TestSFTDerivability:
    """Ensure episodes can derive SFT training data (see docs/episode_schema.md)."""

    def test_can_construct_conversation(self, episode):
        """Must be able to build user/assistant message sequence."""
        messages = []

        # User asks question
        messages.append({
            "role": "user",
            "content": episode["question"]["question_text"],
        })

        # For each turn: assistant provides code, user provides execution result
        for turn in episode["gold_trace"]["turns"]:
            messages.append({
                "role": "assistant",
                "content": turn["code"],
            })
            messages.append({
                "role": "user",
                "content": turn["execution"]["stdout"],
            })

        # Verify alternating roles
        roles = [m["role"] for m in messages]
        for i in range(len(roles) - 1):
            assert roles[i] != roles[i + 1], f"Roles must alternate, got {roles}"

    def test_final_answer_extractable(self, episode):
        """Must be able to extract final answer for reward signal."""
        trace = episode["gold_trace"]
        # If trace succeeded, must have final_answer
        if trace["success"]:
            assert trace["final_answer"] is not None, "Successful trace must have final_answer"
            assert trace["final_answer_hash"], "Successful trace must have final_answer_hash"


# ============= RL Derivability =============


class TestRLDerivability:
    """Ensure episodes can derive RL training signals."""

    def test_verified_episode_has_matching_hashes(self, verified_episode):
        """Verified=True means gold answer matches a ground truth hash."""
        q = verified_episode["question"]
        trace = verified_episode["gold_trace"]

        # Get all valid hashes
        if "ground_truth_hashes" in q and q["ground_truth_hashes"]:
            valid_hashes = set(q["ground_truth_hashes"])
        else:
            valid_hashes = {q.get("ground_truth_hash")}

        assert trace["final_answer_hash"] in valid_hashes, (
            f"Gold hash {trace['final_answer_hash']} not in valid hashes {valid_hashes}"
        )

    def test_triangulation_metadata_complete(self, verified_episode):
        """Triangulation metadata must support confidence scoring."""
        tri = verified_episode["triangulation"]
        required = {
            "n_consistency_runs",
            "n_consistency_succeeded",
            "majority_answer_hash",
            "majority_count",
            "gold_matches_majority",
        }
        missing = required - set(tri.keys())
        assert not missing, f"Triangulation missing fields: {missing}"


# ============= PRM Derivability =============


class TestPRMDerivability:
    """Ensure episodes can derive PRM (Process Reward Model) training data."""

    def test_hooks_structure_when_present(self, episode):
        """If hooks are present, they must have required fields."""
        for turn in episode["gold_trace"]["turns"]:
            for hook in turn["execution"].get("hooks", []):
                # Hooks should have at minimum a value_hash for verification
                assert "value_hash" in hook, "Hook missing 'value_hash'"


# ============= Consistency Traces =============


class TestConsistencyTraces:
    """Validate consistency traces structure."""

    def test_consistency_traces_same_structure_as_gold(self, episode):
        """Consistency traces should have same structure as gold trace."""
        gold_keys = set(episode["gold_trace"].keys())

        for i, trace in enumerate(episode["consistency_traces"]):
            trace_keys = set(trace.keys())
            missing = gold_keys - trace_keys
            assert not missing, f"Consistency trace {i} missing keys: {missing}"


# ============= Multi-Outcome Support =============


class TestMultiOutcomeEpisodes:
    """Test episodes that accept multiple valid answers (controlled ambiguity)."""

    @pytest.fixture
    def multi_outcome_episode(self):
        """Load multi-outcome fixture if available."""
        path = Path("data/fixtures/multi_outcome_episode.json")
        if not path.exists():
            pytest.skip("No multi-outcome fixture available")
        return json.loads(path.read_text())

    def test_multi_outcome_has_hash_list(self, multi_outcome_episode):
        """Multi-outcome questions must list all valid hashes."""
        q = multi_outcome_episode["question"]
        hashes = q.get("ground_truth_hashes", [])
        assert len(hashes) > 1, "Multi-outcome should have >1 valid hash"

    def test_gold_matches_any_valid_hash(self, multi_outcome_episode):
        """Gold trace should match at least one valid hash."""
        q = multi_outcome_episode["question"]
        trace = multi_outcome_episode["gold_trace"]

        valid_hashes = set(q["ground_truth_hashes"])
        assert trace["final_answer_hash"] in valid_hashes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
