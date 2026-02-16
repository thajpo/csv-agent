"""
Main evaluation logic for CSV agent models.

This module implements the Evaluator class which:
1. Loads episodes from JSONL files
2. Runs model inference using Environment.rollout()
3. Compares answers using answers_match() from teacher.py
4. Computes aggregate metrics

Key design decisions:
- REUSE Environment.rollout() for execution (same path as training)
- REUSE answers_match() for comparison (existing float tolerance logic)
- Async throughout for parallel evaluation
"""

import asyncio
import json
import time

from src.core.environment import Environment
from csv_spec import EpisodeJSONL
from src.datagen.teacher import answers_match
from src.eval.metrics import EvalResult, EvalMetrics


class Evaluator:
    """
    Evaluates model performance on test episodes.

    Uses the same execution path as training data generation to ensure
    consistency between training and evaluation.
    """

    def __init__(
        self,
        model: str,
        csv_path: str | None = None,
        max_turns: int = 10,
        sampling_args: dict | None = None,
        float_tol: float = 0.1,
        p_value_tol: float = 0.002,
    ):
        """
        Initialize Evaluator.

        Args:
            model: Model identifier (see config.teacher_model)
            csv_path: Optional CSV path override (if None, uses episode's csv_source)
            max_turns: Maximum conversation turns per episode
            sampling_args: Model sampling parameters (temperature, max_tokens, top_p)
            float_tol: Tolerance for float comparison (default ±0.1)
            p_value_tol: Tolerance for p-value comparison (default ±0.002)
        """
        self.model = model
        self.csv_path_override = csv_path
        self.max_turns = max_turns
        self.sampling_args = sampling_args or {}
        self.float_tol = float_tol
        self.p_value_tol = p_value_tol

    def load_episodes(self, episodes_path: str) -> list[EpisodeJSONL]:
        """
        Load episodes from JSONL file.

        Args:
            episodes_path: Path to episodes JSONL file

        Returns:
            List of EpisodeJSONL objects
        """
        episodes = []
        with open(episodes_path, "r") as f:
            for line in f:
                if line.strip():
                    episode_data = json.loads(line)
                    # Parse as EpisodeJSONL (Pydantic will validate)
                    episode = EpisodeJSONL(**episode_data)
                    episodes.append(episode)
        return episodes

    async def evaluate_episode(self, episode: EpisodeJSONL) -> EvalResult:
        """
        Evaluate a single episode by running model inference and comparing answer.

        Protocol:
        1. Extract question from episode
        2. Run Environment.rollout() (no hint - student mode)
        3. Compare final answer using answers_match()

        Args:
            episode: Episode to evaluate

        Returns:
            EvalResult with correctness, execution success, and metadata
        """
        start_time = time.time()

        # Determine CSV path
        csv_path = self.csv_path_override or episode.csv_source

        # Extract question from episode
        question_text = episode.question["question_text"]
        difficulty = episode.question.get("difficulty")

        # Expected answer from teacher gold trace
        expected_answer = episode.rl_verification_data["expected_final_answer"]
        expected_hash = episode.rl_verification_data.get("expected_final_answer_hash")

        # Run model inference (student mode - no hint)
        try:
            env_instance = await Environment.from_params(
                csv_path=csv_path,
                model=self.model,
                question=question_text,
                hint=None,  # Student doesn't get hint
                mode="student",
                max_turns=self.max_turns,
                sampling_args=self.sampling_args,
            )
            final_state = await env_instance.rollout()

            # Extract answer
            actual_answer = final_state.submitted_answer
            execution_success = actual_answer is not None

            # Compare answers
            if execution_success:
                final_answer_correct = answers_match(
                    hash1=expected_hash,
                    hash2=None,  # Don't have hash for actual answer
                    val1=expected_answer,
                    val2=actual_answer,
                    float_tol=self.float_tol,
                    p_value_tol=self.p_value_tol,
                )
            else:
                final_answer_correct = False

            # Extract metadata
            num_turns = final_state.current_turn
            elapsed_seconds = time.time() - start_time

            return EvalResult(
                episode_id=episode.episode_id,
                question_text=question_text,
                difficulty=difficulty,
                final_answer_correct=final_answer_correct,
                execution_success=execution_success,
                num_turns=num_turns,
                elapsed_seconds=elapsed_seconds,
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                error_message=None,
            )

        except Exception as e:
            # Handle execution errors
            elapsed_seconds = time.time() - start_time
            return EvalResult(
                episode_id=episode.episode_id,
                question_text=question_text,
                difficulty=difficulty,
                final_answer_correct=False,
                execution_success=False,
                num_turns=0,
                elapsed_seconds=elapsed_seconds,
                expected_answer=expected_answer,
                actual_answer=None,
                error_message=str(e),
            )

    async def evaluate_batch(
        self, episodes: list[EpisodeJSONL], concurrency: int = 5
    ) -> list[EvalResult]:
        """
        Evaluate multiple episodes in parallel.

        Args:
            episodes: List of episodes to evaluate
            concurrency: Maximum number of concurrent evaluations

        Returns:
            List of EvalResult objects (same order as input)
        """
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_with_semaphore(episode: EpisodeJSONL) -> EvalResult:
            async with semaphore:
                return await self.evaluate_episode(episode)

        # Run evaluations in parallel
        results = await asyncio.gather(
            *[evaluate_with_semaphore(ep) for ep in episodes]
        )

        return results

    def compute_metrics(self, results: list[EvalResult]) -> EvalMetrics:
        """
        Compute aggregate metrics from evaluation results.

        Args:
            results: List of EvalResult objects

        Returns:
            EvalMetrics with accuracy, execution success rate, and breakdowns
        """
        if not results:
            return EvalMetrics(
                accuracy=0.0,
                execution_success_rate=0.0,
                avg_turns=0.0,
                avg_elapsed_seconds=0.0,
                total_episodes=0,
            )

        # Overall counts
        total_episodes = len(results)
        total_correct = sum(1 for r in results if r.final_answer_correct)
        total_executed = sum(1 for r in results if r.execution_success)

        # Overall metrics
        accuracy = total_correct / total_episodes if total_episodes > 0 else 0.0
        execution_success_rate = (
            total_executed / total_episodes if total_episodes > 0 else 0.0
        )

        # Efficiency metrics
        avg_turns = sum(r.num_turns for r in results) / total_episodes
        avg_elapsed_seconds = sum(r.elapsed_seconds for r in results) / total_episodes

        # Breakdown by difficulty
        episodes_by_difficulty = {}
        correct_by_difficulty = {}

        for result in results:
            difficulty = result.difficulty or "UNKNOWN"

            episodes_by_difficulty[difficulty] = (
                episodes_by_difficulty.get(difficulty, 0) + 1
            )

            if result.final_answer_correct:
                correct_by_difficulty[difficulty] = (
                    correct_by_difficulty.get(difficulty, 0) + 1
                )

        # Compute accuracy by difficulty
        accuracy_by_difficulty = {}
        for difficulty, count in episodes_by_difficulty.items():
            correct_count = correct_by_difficulty.get(difficulty, 0)
            accuracy_by_difficulty[difficulty] = (
                correct_count / count if count > 0 else 0.0
            )

        return EvalMetrics(
            accuracy=accuracy,
            execution_success_rate=execution_success_rate,
            avg_turns=avg_turns,
            avg_elapsed_seconds=avg_elapsed_seconds,
            accuracy_by_difficulty=accuracy_by_difficulty,
            total_episodes=total_episodes,
            total_correct=total_correct,
            total_executed=total_executed,
            episodes_by_difficulty=episodes_by_difficulty,
            correct_by_difficulty=correct_by_difficulty,
        )
