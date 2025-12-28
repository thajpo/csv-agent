"""
Verifiers-compatible RL environment for CSV agent training.

Loads episodes from JSONL and creates a training dataset with:
- Questions as prompts
- Expected answer hashes for verification
- Expected hooks for dense rewards

Uses csv-spec for contract types between trainer and environment.
"""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
import verifiers as vf
from verifiers import MultiTurnEnv
from verifiers.types import State, Messages

from csv_spec import parse_action, parse_step_result, CodeAction

from src.training.rl_rubric import CSVAgentRubric
from src.core.prompts import build_system_prompt, generate_data_overview
from src.envs.csv_env import LocalCSVAnalysisEnv


def load_episodes(episodes_path: str) -> list[dict]:
    """Load episodes from JSONL file."""
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def episodes_to_dataset(
    episodes: list[dict],
    include_unverified: bool = False,
) -> Dataset:
    """
    Convert episodes to HuggingFace Dataset for verifiers.

    Creates columns:
    - prompt: The question text
    - answer: Expected answer hash (for basic verification)
    - task: Question difficulty level
    - info: Dict with expected_hooks, expected_answer, etc.
    """
    prompts = []
    answers = []
    tasks = []
    infos = []

    for ep in episodes:
        # Skip unverified unless requested
        if not include_unverified and not ep.get("verified", False):
            continue

        question = ep.get("question", {})
        rl_data = ep.get("rl_verification_data", {})
        gold_trace = ep.get("teacher_gold_trace", {})

        prompts.append(question.get("question_text", ""))
        answers.append(rl_data.get("expected_final_answer_hash", ""))
        tasks.append(question.get("difficulty", "UNKNOWN"))

        # Store all verification data for rubric
        infos.append(
            {
                "expected_hooks": gold_trace.get("hooks", []),
                "expected_answer_hash": rl_data.get("expected_final_answer_hash"),
                "expected_answer": rl_data.get("expected_final_answer"),
                "hint": question.get("hint", ""),
                "n_steps": question.get("n_steps", 1),
                "episode_id": ep.get("episode_id", ""),
            }
        )

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "answer": answers,
            "task": tasks,
            "info": infos,
        }
    )


class CSVAgentRLEnv(MultiTurnEnv):
    """
    RL Environment for training CSV analysis agents.

    Wraps the csv-agent execution environment with:
    - Episode-based dataset loading
    - Dense reward rubric from hooks
    - System prompt generation

    Compatible with verifiers RL trainer.
    """

    def __init__(
        self,
        episodes_path: str,
        csv_path: str = "data/csv/data.csv",
        dataset_description: str = "",
        max_turns: int = 10,
        include_unverified: bool = False,
        hook_reward: float = 0.1,
        final_reward: float = 1.0,
        float_tolerance: float = 0.1,
        **kwargs,
    ):
        """
        Initialize the RL environment.

        Args:
            episodes_path: Path to episodes JSONL file
            csv_path: Path to CSV data file
            dataset_description: Description of the dataset
            max_turns: Maximum turns per episode
            include_unverified: Include unverified episodes in training
            hook_reward: Reward per matching hook
            final_reward: Reward for correct final answer
            float_tolerance: Tolerance for float comparison
            **kwargs: Passed to MultiTurnEnv
        """
        self.csv_path = csv_path
        self.dataset_description = dataset_description
        self.episodes_path = episodes_path

        # Load episodes and create dataset
        episodes = load_episodes(episodes_path)
        dataset = episodes_to_dataset(episodes, include_unverified)

        if len(dataset) == 0:
            raise ValueError(f"No valid episodes found in {episodes_path}")

        # Create rubric for dense rewards
        rubric = CSVAgentRubric(
            hook_reward=hook_reward,
            final_reward=final_reward,
            float_tolerance=float_tolerance,
        )

        # Generate system prompt
        data_overview = generate_data_overview(csv_path)
        system_prompt = self._build_system_prompt(dataset_description, data_overview)

        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )

    def _build_system_prompt(
        self,
        dataset_description: str,
        data_overview: str,
    ) -> str:
        """Build the system prompt for the student agent."""
        # Use the student prompt template
        return build_system_prompt(
            mode="student",
            dataset_description=dataset_description,
            data_overview=data_overview,
            question=None,  # Question comes from dataset
        )

    async def setup_state(self, state: State) -> State:
        """
        Initialize sandbox environment for code execution.

        Creates a LocalCSVAnalysisEnv container that persists for the episode.
        """
        # Create sandbox for this episode
        env = LocalCSVAnalysisEnv(csv_path=self.csv_path)
        sandbox_state: dict = {}
        sandbox_state = await env.setup_state(sandbox_state)

        # Store in state for use during episode
        state["csv_env"] = env
        state["sandbox_state"] = sandbox_state

        return state

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """
        Process model completion and provide environment feedback.

        Extracts Python code from the last assistant message, executes it
        in the sandbox, and returns the execution result as a user message.

        Uses csv_spec contract types for parsing.
        """
        # Get the last assistant message
        assert isinstance(messages, list) and len(messages) > 0
        last_message = messages[-1]

        # Handle tool calls (for compatibility with tool-using agents)
        if "tool_calls" in last_message:
            # Not implemented - our agent uses markdown code blocks
            return [
                {
                    "role": "user",
                    "content": "Tool calls not supported. Use ```python code blocks.",
                }
            ]

        # Extract content from assistant message
        content = last_message.get("content", "")
        if not content:
            return [
                {
                    "role": "user",
                    "content": "No content in response. Please write code.",
                }
            ]

        # Parse action using csv_spec contract
        action = parse_action(content)

        if action is None:
            # No code block found
            return [
                {
                    "role": "user",
                    "content": "No Python code block found. Please write code in ```python blocks.",
                }
            ]

        if not isinstance(action, CodeAction):
            # Unexpected action type
            return [
                {
                    "role": "user",
                    "content": f"Unexpected action type: {type(action).__name__}",
                }
            ]

        # Execute code in sandbox
        env: LocalCSVAnalysisEnv = state["csv_env"]
        sandbox_state = state["sandbox_state"]

        output = await env.python(
            code=action.code,
            sandbox_id=sandbox_state["sandbox_id"],
            python_state=sandbox_state.get("python_state"),
        )

        # Parse result using csv_spec contract
        step_result = parse_step_result(output)

        # Store hooks and submission in state for rubric scoring
        if step_result.hooks:
            state.setdefault("captured_hooks", []).extend(step_result.hooks)
        if step_result.submitted_answer is not None:
            state["submitted_answer"] = step_result.submitted_answer
            state["terminal"] = True

        # Format observation for model
        if step_result.success:
            observation = (
                f"✓ Code executed successfully\n\nOutput:\n{step_result.stdout}"
            )
        else:
            observation = f"✗ Code execution failed\n\nError:\n{step_result.stderr or step_result.stdout}"

        # Add continuation prompt if not terminal
        if not step_result.terminal:
            observation += (
                "\n\nContinue your analysis or call submit(answer) when ready."
            )

        return [{"role": "user", "content": observation}]

    async def cleanup_state(self, state: State) -> None:
        """Clean up sandbox container after episode."""
        if "csv_env" in state and "sandbox_state" in state:
            env: LocalCSVAnalysisEnv = state["csv_env"]
            sandbox_id = state["sandbox_state"].get("sandbox_id")
            if sandbox_id:
                await env.destroy_sandbox(sandbox_id)


def load_environment(
    episodes_path: str,
    csv_path: str = "data/csv/data.csv",
    **kwargs,
) -> CSVAgentRLEnv:
    """
    Factory function for loading the CSV Agent RL environment.

    This is the standard entry point for verifiers.

    Args:
        episodes_path: Path to episodes JSONL file
        csv_path: Path to CSV data file
        **kwargs: Additional environment arguments

    Returns:
        Configured CSVAgentRLEnv instance
    """
    return CSVAgentRLEnv(
        episodes_path=episodes_path,
        csv_path=csv_path,
        **kwargs,
    )


# Entry point for verifiers config
__all__ = ["CSVAgentRLEnv", "load_environment", "load_episodes", "episodes_to_dataset"]
