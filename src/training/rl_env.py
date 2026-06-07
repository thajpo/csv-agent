import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
import verifiers as vf
from verifiers import MultiTurnEnv
from verifiers.types import State, Messages
from verifiers.utils.message_utils import concat_messages

from csv_spec import parse_action, parse_step_result, CodeAction

from src.training.rl_rubric import CSVAgentRubric
from src.core.prompts import generate_data_overview
from src.envs.csv_env import LocalCSVAnalysisEnv


def load_episodes(episodes_path: str) -> list[dict]:
    """Load episodes from a local JSONL file."""
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def load_hf_episodes(
    dataset_name: str,
    *,
    dataset_split: str = "train",
    dataset_config: str | None = None,
    hf_revision: str | None = None,
    hf_token: bool | str | None = None,
) -> list[dict]:
    """Load canonical episodes from a Hugging Face dataset split."""
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=dataset_split,
        revision=hf_revision,
        token=hf_token,
    )
    episodes = []
    for row in dataset:
        row_dict = dict(row)
        if "episode_json" in row_dict:
            episodes.append(json.loads(row_dict["episode_json"]))
        else:
            episodes.append(row_dict)
    return episodes


def _json_default(value: Any) -> str:
    return str(value)


def _info_dict(info: Any) -> dict:
    if isinstance(info, dict):
        return info
    if isinstance(info, str):
        try:
            parsed = json.loads(info)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


@lru_cache(maxsize=128)
def _cached_data_overview(csv_path: str) -> str:
    return generate_data_overview(csv_path)


def _flatten_hooks(gold_trace: dict) -> list[dict]:
    hooks: list[dict] = []
    for turn in gold_trace.get("turns", []):
        execution = turn.get("execution", {})
        hooks.extend(execution.get("hooks", []) or [])
    return hooks


def _answer_hashes(question: dict, gold_trace: dict) -> list[str]:
    hashes = question.get("ground_truth_hashes") or []
    if question.get("ground_truth_hash"):
        hashes = [*hashes, question["ground_truth_hash"]]
    if gold_trace.get("final_answer_hash"):
        hashes = [*hashes, gold_trace["final_answer_hash"]]
    return list(dict.fromkeys(h for h in hashes if h))


def _resolve_csv_source(
    raw_csv_source: str | None,
    *,
    csv_path: str | None,
    csv_root: str | None,
) -> str:
    csv_source = raw_csv_source or csv_path
    if not csv_source:
        raise ValueError("Episode has no csv_source and no csv_path fallback was provided")

    raw_path = Path(csv_source)
    if csv_root:
        root = Path(csv_root)
        parts = raw_path.parts
        if "data" in parts:
            data_index = parts.index("data")
            if len(parts) > data_index + 1 and parts[data_index + 1] == "kaggle":
                candidate = root.joinpath(*parts[data_index + 2 :])
                if candidate.exists() or not raw_path.exists():
                    return str(candidate)
        if not raw_path.is_absolute():
            candidate = root / raw_path
            if candidate.exists() or not raw_path.exists():
                return str(candidate)

    return str(raw_path)


def _hf_csv_root(
    dataset_name: str | None,
    *,
    csv_root: str | None,
    download_csvs: bool,
    hf_revision: str | None,
    hf_token: bool | str | None,
) -> str | None:
    if csv_root or not dataset_name or not download_csvs:
        return csv_root

    snapshot = Path(
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            revision=hf_revision,
            token=hf_token,
            allow_patterns=["data/kaggle/**"],
        )
    )
    candidate = snapshot / "data" / "kaggle"
    return str(candidate) if candidate.exists() else csv_root


def _system_prompt(
    *,
    csv_source: str,
    dataset_description: str,
    data_overview: str,
) -> str:
    description = dataset_description or f"CSV file: {csv_source}"
    return f"""Solve CSV data analysis tasks using Python and pandas.

DATASET:
{description}

DATA OVERVIEW:
```
{data_overview}
```

The dataframe `df` is already loaded.

=== APPROACH ===
1. Check available columns when useful: `print(df.columns.tolist())`
2. Explore relevant data before computing
3. Use clear intermediate variables
4. Submit your final answer with `submit(result)`

=== OUTPUT FORMAT ===
- Numbers: `submit(10.5)` or `submit(round(result, 2))`
- JSON-like answers: `submit({{"key": "value"}})`
- Statistical tests: `submit({{"p_value": 0.01, "answer": "Yes"}})`

=== TURN STRUCTURE ===
1. Brief reasoning
2. ONE ```python code block
3. Stop and wait for execution output

If you encounter an error, read it carefully and fix the issue in your next turn.
""".strip()


def _episode_to_row(
    ep: dict,
    *,
    csv_path: str | None,
    csv_root: str | None,
    dataset_description: str,
    include_data_overview: bool,
) -> dict:
    question = ep.get("question", {})
    gold_trace = ep.get("gold_trace", {})
    csv_source = _resolve_csv_source(
        ep.get("csv_source"),
        csv_path=csv_path,
        csv_root=csv_root,
    )

    data_overview = (
        _cached_data_overview(str(csv_source))
        if include_data_overview
        else "Data overview omitted by environment configuration."
    )
    answer_hashes = _answer_hashes(question, gold_trace)
    expected_answer_hash = answer_hashes[0] if answer_hashes else None
    expected_answer = question.get("ground_truth", gold_trace.get("final_answer"))
    info = {
        "episode_id": ep.get("episode_id"),
        "question_id": question.get("id"),
        "csv_source": str(csv_source),
        "difficulty": question.get("difficulty", "UNKNOWN"),
        "expected_hooks": _flatten_hooks(gold_trace),
        "expected_answer": expected_answer,
        "expected_answer_hash": expected_answer_hash,
        "expected_answer_hashes": answer_hashes,
        "gold_final_answer": gold_trace.get("final_answer"),
        "gold_final_answer_hash": gold_trace.get("final_answer_hash"),
    }

    return {
        "prompt": [
            {
                "role": "system",
                "content": _system_prompt(
                    csv_source=str(csv_source),
                    dataset_description=dataset_description,
                    data_overview=data_overview,
                ),
            },
            {"role": "user", "content": question.get("question_text", "")},
        ],
        "answer": expected_answer_hash or "",
        "task": question.get("difficulty", "UNKNOWN"),
        "info": json.dumps(info, default=_json_default),
    }


def episodes_to_dataset(
    episodes: list[dict],
    include_unverified: bool = False,
    csv_path: str | None = None,
    csv_root: str | None = None,
    dataset_description: str = "",
    include_data_overview: bool = True,
) -> Dataset:
    """
    Convert episodes to HuggingFace Dataset for verifiers.

    Creates columns:
    - prompt: Pre-formatted system+user messages
    - answer: Primary expected answer hash
    - task: Question difficulty
    - info: JSON-encoded reward metadata
    """
    rows = []
    for ep in episodes:
        if not include_unverified and not ep.get("verified", False):
            continue
        rows.append(
            _episode_to_row(
                ep,
                csv_path=csv_path,
                csv_root=csv_root,
                dataset_description=dataset_description,
                include_data_overview=include_data_overview,
            )
        )

    return Dataset.from_list(rows)


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
        episodes_path: str | None = None,
        dataset_name: str | None = None,
        dataset_split: str = "train",
        dataset_config: str | None = None,
        hf_revision: str | None = None,
        hf_token: bool | str | None = None,
        download_csvs: bool = True,
        csv_path: str | None = None,
        csv_root: str | None = None,
        eval_episodes_path: str | None = None,
        eval_dataset_split: str | None = None,
        dataset_description: str = "",
        max_turns: int = 10,
        include_unverified: bool = False,
        include_data_overview: bool = True,
        hook_reward: float = 0.1,
        final_reward: float = 1.0,
        float_tolerance: float = 0.1,
        **kwargs,
    ):
        """
        Initialize the RL environment.

        Args:
            episodes_path: Path to local episodes JSONL file
            dataset_name: Optional Hugging Face dataset repo ID with episode splits
            dataset_split: HF split to use when dataset_name is provided
            dataset_config: Optional HF dataset config name
            hf_revision: Optional HF revision/branch/tag
            hf_token: HF token, True for cached login, or None for default behavior
            download_csvs: If True, snapshot data/kaggle from the HF dataset repo
            csv_path: Fallback CSV path when episodes do not carry csv_source
            csv_root: Optional root used to rebase episode csv_source paths
            eval_episodes_path: Optional local eval episodes JSONL file
            eval_dataset_split: Optional HF eval split; defaults to no eval dataset
            dataset_description: Description of the dataset
            max_turns: Maximum turns per episode
            include_unverified: Include unverified episodes in training
            hook_reward: Reward per matching hook
            final_reward: Reward for correct final answer
            float_tolerance: Tolerance for float comparison
            **kwargs: Passed to MultiTurnEnv
        """
        if not episodes_path and not dataset_name:
            raise ValueError("CSVAgentRLEnv requires episodes_path or dataset_name")

        csv_root = _hf_csv_root(
            dataset_name,
            csv_root=csv_root,
            download_csvs=download_csvs,
            hf_revision=hf_revision,
            hf_token=hf_token,
        )

        self.csv_path = csv_path
        self.csv_root = csv_root
        self.dataset_description = dataset_description
        self.episodes_path = episodes_path
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.dataset_config = dataset_config
        self.hf_revision = hf_revision
        self.eval_episodes_path = eval_episodes_path
        self.eval_dataset_split = eval_dataset_split

        episodes = (
            load_episodes(episodes_path)
            if episodes_path
            else load_hf_episodes(
                dataset_name or "",
                dataset_split=dataset_split,
                dataset_config=dataset_config,
                hf_revision=hf_revision,
                hf_token=hf_token,
            )
        )
        dataset = episodes_to_dataset(
            episodes,
            include_unverified=include_unverified,
            csv_path=csv_path,
            csv_root=csv_root,
            dataset_description=dataset_description,
            include_data_overview=include_data_overview,
        )
        eval_dataset = None
        if eval_episodes_path:
            eval_dataset = episodes_to_dataset(
                load_episodes(eval_episodes_path),
                include_unverified=include_unverified,
                csv_path=csv_path,
                csv_root=csv_root,
                dataset_description=dataset_description,
                include_data_overview=include_data_overview,
            )
        elif dataset_name and eval_dataset_split:
            eval_dataset = episodes_to_dataset(
                load_hf_episodes(
                    dataset_name,
                    dataset_split=eval_dataset_split,
                    dataset_config=dataset_config,
                    hf_revision=hf_revision,
                    hf_token=hf_token,
                ),
                include_unverified=include_unverified,
                csv_path=csv_path,
                csv_root=csv_root,
                dataset_description=dataset_description,
                include_data_overview=include_data_overview,
            )
        else:
            eval_dataset = dataset

        if len(dataset) == 0:
            source = episodes_path or f"{dataset_name}:{dataset_split}"
            raise ValueError(f"No valid episodes found in {source}")

        rubric = CSVAgentRubric(
            hook_reward=hook_reward,
            final_reward=final_reward,
            float_tolerance=float_tolerance,
        )

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            max_turns=max_turns,
            env_id="csv-agent",
            **kwargs,
        )

    async def setup_state(self, state: State) -> State:
        """
        Initialize sandbox environment for code execution.

        Creates a LocalCSVAnalysisEnv container that persists for the episode.
        """
        info = _info_dict(state.get("info", {}))
        if info:
            state["info"] = info
        csv_source = info.get("csv_source") or self.csv_path
        if not csv_source:
            raise ValueError("CSVAgentRLEnv requires csv_source in episode info or csv_path")

        env = LocalCSVAnalysisEnv(csv_path=str(csv_source))
        sandbox_state: dict = {}
        sandbox_state = await env.setup_state(sandbox_state)

        state["csv_env"] = env
        state["sandbox_state"] = sandbox_state
        state["captured_hooks"] = []

        return state

    async def add_model_response(self, state, prompt_messages, response):
        """
        Record the model response, then eagerly execute its code.

        Verifiers 0.1.8 only calls env_response while preparing the next model
        prompt. Eager execution lets submit() stop the rollout before an
        unnecessary extra model turn while preserving the normal prompt shape
        for non-terminal turns.
        """
        await super().add_model_response(state, prompt_messages, response)
        step = state["trajectory"][-1]
        messages = concat_messages([step["prompt"], step["completion"]])
        state["pending_env_response"] = await self.env_response(messages, state)

    async def get_prompt_messages(self, state: State) -> Messages:
        if len(state["trajectory"]) == 0:
            return state["prompt"]

        prev_turn_prompt = state["trajectory"][-1]["prompt"]
        prev_turn_completion = state["trajectory"][-1]["completion"]
        messages = concat_messages([prev_turn_prompt, prev_turn_completion])
        env_response = state.pop("pending_env_response", None)
        if env_response is None:
            env_response = await self.env_response(messages, state)
        return concat_messages([messages, env_response])

    @vf.stop(priority=50)
    async def submitted_answer(self, state: State) -> bool:
        return bool(state.get("terminal"))

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
        if isinstance(last_message, dict) and "tool_calls" in last_message:
            # Not implemented - our agent uses markdown code blocks
            return [
                {
                    "role": "user",
                    "content": "Tool calls not supported. Use ```python code blocks.",
                }
            ]

        # Extract content from assistant message
        if isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = getattr(last_message, "content", "")
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
            state.setdefault("captured_hooks", []).extend(
                [dict(hook) for hook in step_result.hooks]
            )
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

    @vf.cleanup
    async def cleanup_state(self, state: State) -> None:
        """Clean up sandbox container after episode."""
        if "csv_env" in state and "sandbox_state" in state:
            env: LocalCSVAnalysisEnv = state["csv_env"]
            sandbox_id = state["sandbox_state"].get("sandbox_id")
            if sandbox_id:
                await env.destroy_sandbox(sandbox_id)


def load_environment(
    episodes_path: str | None = None,
    csv_path: str | None = None,
    **kwargs,
) -> CSVAgentRLEnv:
    """
    Factory function for loading the CSV Agent RL environment.

    This is the standard entry point for verifiers.

    Args:
        episodes_path: Path to local episodes JSONL file
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
