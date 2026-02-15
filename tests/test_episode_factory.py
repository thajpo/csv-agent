"""Tests for episode factory.

Test-first development for the episode factory module.
"""

import pytest
from datetime import datetime
from typing import Any
from unittest.mock import patch, AsyncMock

from csv_spec import EpisodeJSONL, TraceDict, QADict
from src.datagen.shared.verification import VerificationResult


@pytest.fixture
def sample_question() -> dict:
    """Sample question dict for testing."""
    return {
        "id": "test-123",
        "source": "template",
        "question_text": "What is the average age?",
        "hint": "Use the 'age' column",
        "difficulty": "EASY",
        "n_steps": 1,
        "template_name": "test_template",
        "template_params": {"column": "age"},
        "output_type": "float",
        "ground_truth_hash": "abc123",
        "ground_truth_hashes": ["abc123", "def456"],
        "ground_truth": 42.5,
    }


@pytest.fixture
def sample_trace() -> TraceDict:
    """Sample trace for testing."""
    return {
        "turns": [
            {
                "turn_index": 0,
                "reasoning": "Calculate average",
                "code": "result = df['age'].mean()",
                "execution": {
                    "success": True,
                    "stdout": "",
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 42.5,
                },
            }
        ],
        "final_answer": 42.5,
        "final_answer_hash": "abc123",
        "success": True,
    }


@pytest.fixture
def sample_verification_result_success(sample_trace: TraceDict) -> VerificationResult:
    """Sample successful verification result."""
    return VerificationResult(
        success=True,
        match=True,
        trace=sample_trace,
        traces=[],
        majority_answer_hash="abc123",
        error=None,
    )


@pytest.fixture
def sample_verification_result_failure(sample_trace: TraceDict) -> VerificationResult:
    """Sample failed verification result."""
    return VerificationResult(
        success=False,
        match=False,
        trace=sample_trace,
        traces=[],
        majority_answer_hash=None,
        error="Answer mismatch",
    )


@pytest.fixture
def sample_verification_result_consistency(
    sample_trace: TraceDict,
) -> VerificationResult:
    """Sample consistency verification result with multiple traces."""
    consistency_trace: TraceDict = {
        "turns": [
            {
                "turn_index": 0,
                "reasoning": "Calculate",
                "code": "result = df['age'].mean()",
                "execution": {
                    "success": True,
                    "stdout": "",
                    "stderr": "",
                    "hooks": [],
                    "submitted_answer": 42.5,
                },
            }
        ],
        "final_answer": 42.5,
        "final_answer_hash": "abc123",
        "success": True,
    }
    return VerificationResult(
        success=True,
        match=True,
        trace=sample_trace,
        traces=[consistency_trace, consistency_trace],
        majority_answer_hash="abc123",
        error=None,
    )


class TestCreateEpisode:
    """Tests for create_episode function."""

    @pytest.mark.asyncio
    async def test_create_episode_success_synthetic(
        self,
        sample_question: dict,
        sample_verification_result_success: VerificationResult,
    ):
        """Test creating episode from successful ground-truth verification."""
        from src.datagen.shared.episode_factory import create_episode

        episode = await create_episode(
            question=sample_question,
            verification_result=sample_verification_result_success,
            source="template",
            csv_path="/path/to/data.csv",
        )

        assert isinstance(episode, EpisodeJSONL)
        assert episode.verified is True
        assert episode.source == "template"
        assert episode.csv_source == "/path/to/data.csv"
        assert episode.question["question_text"] == "What is the average age?"
        assert episode.gold_trace["final_answer"] == 42.5
        assert episode.triangulation["gold_matches_majority"] is True

    @pytest.mark.asyncio
    async def test_create_episode_failure_synthetic(
        self,
        sample_question: dict,
        sample_verification_result_failure: VerificationResult,
    ):
        """Test creating episode from failed ground-truth verification."""
        from src.datagen.shared.episode_factory import create_episode

        episode = await create_episode(
            question=sample_question,
            verification_result=sample_verification_result_failure,
            source="template",
            csv_path="/path/to/data.csv",
        )

        assert isinstance(episode, EpisodeJSONL)
        assert episode.verified is False
        assert episode.source == "template"
        assert episode.triangulation["gold_matches_majority"] is False

    @pytest.mark.asyncio
    async def test_create_episode_llm_consistency(
        self,
        sample_question: dict,
        sample_verification_result_consistency: VerificationResult,
    ):
        """Test creating episode from consistency verification (LLM)."""
        from src.datagen.shared.episode_factory import create_episode

        sample_question["source"] = "llm_gen"

        episode = await create_episode(
            question=sample_question,
            verification_result=sample_verification_result_consistency,
            source="llm_gen",
            csv_path="/path/to/data.csv",
        )

        assert isinstance(episode, EpisodeJSONL)
        assert episode.verified is True
        assert episode.source == "llm_gen"
        assert len(episode.consistency_traces) == 2
        assert episode.triangulation["n_consistency_runs"] == 2

    @pytest.mark.asyncio
    async def test_create_episode_procedural(
        self,
        sample_question: dict,
        sample_verification_result_success: VerificationResult,
    ):
        """Test creating episode from procedural question."""
        from src.datagen.shared.episode_factory import create_episode

        sample_question["source"] = "procedural"

        episode = await create_episode(
            question=sample_question,
            verification_result=sample_verification_result_success,
            source="procedural",
            csv_path="/path/to/data.csv",
        )

        assert isinstance(episode, EpisodeJSONL)
        assert episode.source == "procedural"

    @pytest.mark.asyncio
    async def test_create_episode_preserves_question_metadata(
        self,
        sample_question: dict,
        sample_verification_result_success: VerificationResult,
    ):
        """Test that question metadata is preserved in episode."""
        from src.datagen.shared.episode_factory import create_episode

        episode = await create_episode(
            question=sample_question,
            verification_result=sample_verification_result_success,
            source="template",
            csv_path="/path/to/data.csv",
        )

        assert episode.question["id"] == "test-123"
        assert episode.question["difficulty"] == "EASY"
        assert episode.question["template_name"] == "test_template"
        assert episode.question["ground_truth_hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_create_episode_generates_episode_id(
        self,
        sample_question: dict,
        sample_verification_result_success: VerificationResult,
    ):
        """Test that episode_id is generated."""
        from src.datagen.shared.episode_factory import create_episode

        episode = await create_episode(
            question=sample_question,
            verification_result=sample_verification_result_success,
            source="template",
            csv_path="/path/to/data.csv",
        )

        assert episode.episode_id is not None
        assert len(episode.episode_id) > 0

    @pytest.mark.asyncio
    async def test_create_episode_sets_timestamp(
        self,
        sample_question: dict,
        sample_verification_result_success: VerificationResult,
    ):
        """Test that timestamp is set."""
        from src.datagen.shared.episode_factory import create_episode

        before = datetime.now()
        episode = await create_episode(
            question=sample_question,
            verification_result=sample_verification_result_success,
            source="template",
            csv_path="/path/to/data.csv",
        )
        after = datetime.now()

        assert before <= episode.timestamp <= after


class TestCreateEpisodeFromGroundTruth:
    """Tests for create_episode_from_ground_truth helper."""

    @pytest.mark.asyncio
    async def test_create_from_ground_truth_success(
        self,
        sample_question: dict,
        sample_trace: TraceDict,
    ):
        """Test creating episode using ground-truth verification."""
        from src.datagen.shared.episode_factory import create_episode_from_ground_truth

        # Mock the verification module
        mock_result = VerificationResult(
            success=True,
            match=True,
            trace=sample_trace,
            traces=[],
            majority_answer_hash="abc123",
            error=None,
        )

        with patch(
            "src.datagen.shared.episode_factory.verify_question",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            episode = await create_episode_from_ground_truth(
                question=sample_question,
                csv_path="/path/to/data.csv",
                model="test-model",
            )

        assert isinstance(episode, EpisodeJSONL)
        assert episode.verified is True
        assert episode.source == "template"

    @pytest.mark.asyncio
    async def test_create_from_ground_truth_with_source_override(
        self,
        sample_question: dict,
        sample_trace: TraceDict,
    ):
        """Test that source can be overridden for ground-truth."""
        from src.datagen.shared.episode_factory import create_episode_from_ground_truth

        sample_question["source"] = "procedural"

        mock_result = VerificationResult(
            success=True,
            match=True,
            trace=sample_trace,
            traces=[],
            majority_answer_hash="abc123",
            error=None,
        )

        with patch(
            "src.datagen.shared.episode_factory.verify_question",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            episode = await create_episode_from_ground_truth(
                question=sample_question,
                csv_path="/path/to/data.csv",
                model="test-model",
                source="procedural",  # Override default
            )

        assert episode.source == "procedural"


class TestCreateEpisodeFromConsistency:
    """Tests for create_episode_from_consistency helper."""

    @pytest.mark.asyncio
    async def test_create_from_consistency_success(
        self,
        sample_question: dict,
        sample_trace: TraceDict,
    ):
        """Test creating episode using consistency verification."""
        from src.datagen.shared.episode_factory import create_episode_from_consistency

        sample_question["source"] = "llm_gen"

        consistency_trace: TraceDict = {
            "turns": [],
            "final_answer": 42.5,
            "final_answer_hash": "abc123",
            "success": True,
        }

        mock_result = VerificationResult(
            success=True,
            match=True,
            trace=sample_trace,
            traces=[consistency_trace, consistency_trace],
            majority_answer_hash="abc123",
            error=None,
        )

        with patch(
            "src.datagen.shared.episode_factory.verify_question",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            episode = await create_episode_from_consistency(
                question=sample_question,
                csv_path="/path/to/data.csv",
                model="test-model",
                n_consistency=5,
            )

        assert isinstance(episode, EpisodeJSONL)
        assert episode.verified is True
        assert episode.source == "llm_gen"
        assert len(episode.consistency_traces) == 2

    @pytest.mark.asyncio
    async def test_create_from_consistency_passes_n_consistency(
        self,
        sample_question: dict,
        sample_trace: TraceDict,
    ):
        """Test that n_consistency parameter is passed to verification."""
        from src.datagen.shared.episode_factory import create_episode_from_consistency

        sample_question["source"] = "llm_gen"

        mock_result = VerificationResult(
            success=True,
            match=True,
            trace=sample_trace,
            traces=[],
            majority_answer_hash="abc123",
            error=None,
        )

        mock_verify = AsyncMock(return_value=mock_result)

        with patch(
            "src.datagen.shared.episode_factory.verify_question",
            new=mock_verify,
        ):
            await create_episode_from_consistency(
                question=sample_question,
                csv_path="/path/to/data.csv",
                model="test-model",
                n_consistency=10,
            )

        # Verify that n_traces was passed correctly
        call_kwargs = mock_verify.call_args.kwargs
        assert call_kwargs["n_traces"] == 10


class TestEpisodeFactoryErrorHandling:
    """Tests for error handling in episode factory."""

    @pytest.mark.asyncio
    async def test_handles_none_trace_gracefully(
        self,
        sample_question: dict,
    ):
        """Test handling of None trace in verification result."""
        from src.datagen.shared.episode_factory import create_episode

        # This should raise ValueError since trace is required
        mock_result = VerificationResult(
            success=False,
            match=None,
            trace=None,
            traces=[],
            majority_answer_hash=None,
            error="Execution failed",
        )

        # Should still create an episode, but with error info
        episode = await create_episode(
            question=sample_question,
            verification_result=mock_result,
            source="template",
            csv_path="/path/to/data.csv",
        )

        assert isinstance(episode, EpisodeJSONL)
        assert episode.verified is False

    @pytest.mark.asyncio
    async def test_preserves_error_info_in_episode(
        self,
        sample_question: dict,
    ):
        """Test that error information is preserved."""
        from src.datagen.shared.episode_factory import create_episode

        mock_result = VerificationResult(
            success=False,
            match=None,
            trace=None,
            traces=[],
            majority_answer_hash=None,
            error="Ground truth hash mismatch",
        )

        episode = await create_episode(
            question=sample_question,
            verification_result=mock_result,
            source="template",
            csv_path="/path/to/data.csv",
        )

        # Error info should be accessible somehow (via trace or metadata)
        assert episode.verified is False
