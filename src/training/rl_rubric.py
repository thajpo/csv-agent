"""
Dense reward rubric for CSV agent using execution hooks.

Provides partial credit for:
1. Matching intermediate computation hooks (dense signal)
2. Correct final answer (sparse signal)
"""

from typing import Any
import re
import json

from verifiers import Rubric
from verifiers.types import State

from src.datagen.teacher import answers_match
from src.utils.hashing import hash_artifact


class CSVAgentRubric(Rubric):
    """
    Rubric that computes dense rewards from hook matching.

    Reward structure:
    - Final answer correct: +1.0
    - Each matching hook: +0.1 (up to num_hooks * 0.1)

    Expected episode info fields (from RolloutInput.info):
    - expected_hooks: list[dict] with 'value_hash' and 'code_line'
    - expected_answer_hash: str (16-char hex)
    - expected_answer: Any (for fallback comparison)
    """

    def __init__(
        self,
        hook_reward: float = 0.1,
        final_reward: float = 1.0,
        float_tolerance: float = 0.1,
    ):
        """
        Initialize the rubric.

        Args:
            hook_reward: Reward per matching hook (default: 0.1)
            final_reward: Reward for correct final answer (default: 1.0)
            float_tolerance: Tolerance for float comparison
        """
        self.hook_reward = hook_reward
        self.final_reward = final_reward
        self.float_tolerance = float_tolerance

        super().__init__(funcs=[self.compute_reward])

    def extract_submission(self, completion: str) -> tuple[Any, list[dict]]:
        """
        Extract submitted answer and hooks from completion text.

        Looks for the submit() output format:
        ✓ Submitted: {"__csv_agent_answer__": value, "hooks": [...]}

        Returns:
            (final_answer, hooks_list) or (None, []) if not found
        """
        # Look for submission line
        pattern = r'✓ Submitted: ({.*})'
        match = re.search(pattern, completion)

        if not match:
            return None, []

        try:
            data = json.loads(match.group(1))
            answer = data.get("__csv_agent_answer__")
            hooks = data.get("hooks", [])
            return answer, hooks
        except json.JSONDecodeError:
            return None, []

    def count_matching_hooks(
        self,
        actual_hooks: list[dict],
        expected_hooks: list[dict],
    ) -> int:
        """
        Count how many hooks match between actual and expected.

        Matches by value_hash only (code_line is for human inspection).
        """
        expected_hashes = {h.get("value_hash") for h in expected_hooks if h.get("value_hash")}

        matched = 0
        for hook in actual_hooks:
            if hook.get("value_hash") in expected_hashes:
                matched += 1

        return matched

    def compute_reward(
        self,
        state: State,
        **kwargs,
    ) -> float:
        """
        Compute dense reward from execution hooks and final answer.

        Args:
            state: Verifiers State containing:
                - input.info: Expected hooks/answer from episode
                - completion: Model's final output
                - trajectory: Execution history

        Returns:
            Total reward (hook_matches * hook_reward + final_match * final_reward)
        """
        # Extract expected values from input info
        info = state.get("input", {}).get("info", {})
        expected_hooks = info.get("expected_hooks", [])
        expected_answer_hash = info.get("expected_answer_hash")
        expected_answer = info.get("expected_answer")

        # Extract actual submission from completion
        completion = state.get("completion", "")
        if isinstance(completion, list):
            # If completion is a list of messages, concatenate content
            completion = "\n".join(
                m.get("content", "") for m in completion
                if isinstance(m, dict) and m.get("content")
            )

        actual_answer, actual_hooks = self.extract_submission(completion)

        # Compute hook reward (dense)
        hook_matches = self.count_matching_hooks(actual_hooks, expected_hooks)
        hook_reward_total = hook_matches * self.hook_reward

        # Compute final answer reward (sparse)
        actual_hash = hash_artifact(actual_answer) if actual_answer is not None else None

        final_correct = answers_match(
            actual_hash, expected_answer_hash,
            actual_answer, expected_answer,
            float_tol=self.float_tolerance,
        )
        final_reward_total = self.final_reward if final_correct else 0.0

        return hook_reward_total + final_reward_total


def create_rubric(
    hook_reward: float = 0.1,
    final_reward: float = 1.0,
    float_tolerance: float = 0.1,
) -> CSVAgentRubric:
    """Factory function for creating the CSV agent rubric."""
    return CSVAgentRubric(
        hook_reward=hook_reward,
        final_reward=final_reward,
        float_tolerance=float_tolerance,
    )
