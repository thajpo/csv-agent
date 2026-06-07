import json
import asyncio
import base64
from functools import lru_cache
from pathlib import Path
import shlex
import textwrap
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
from src.envs.csv_env import LocalCSVAnalysisEnv, SETUP_CODE


DEFAULT_SANDBOX_PIP_PACKAGES = "pandas numpy scipy scikit-learn statsmodels"


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


class PrimeSandboxCSVAnalysisEnv:
    """
    Prime-hosted Python sandbox backend for CSV RL rollouts.

    This avoids requiring Docker on the training host. It preserves the small
    interface used by CSVAgentRLEnv: setup_state(), python(), destroy_sandbox().
    """

    _START_COMMAND_TEMPLATE = textwrap.dedent(
        """
        bash -lc '
        set -euo pipefail

        command_fifo="{command_fifo}"
        response_fifo="{response_fifo}"
        ready_flag="{ready_flag}"
        worker_path="{worker_path}"

        rm -f "$command_fifo" "$response_fifo" "$ready_flag"
        pip install -q {pip_install_packages}

        python - <<'PY'
import base64
from pathlib import Path

Path("{worker_path}").write_bytes(base64.b64decode("{worker_b64}"))
PY

        python -u "$worker_path" &
        tail -f /dev/null
        '
        """
    )

    def __init__(
        self,
        csv_path: str,
        pip_install_packages: str = DEFAULT_SANDBOX_PIP_PACKAGES,
        cpu_cores: int = 2,
        memory_gb: int = 8,
        disk_size_gb: int = 8,
        timeout_minutes: int = 60,
        timeout_per_command_seconds: int = 60,
    ) -> None:
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
        except ImportError as exc:  # pragma: no cover - depends on Prime-RL venv
            raise ImportError(
                "prime-sandboxes is required for sandbox_backend='prime'. "
                "Install Prime-RL/Verifiers dependencies first."
            ) from exc

        worker_code = LocalCSVAnalysisEnv._WORKER_SCRIPT.format(
            command_fifo=LocalCSVAnalysisEnv._COMMAND_FIFO,
            response_fifo=LocalCSVAnalysisEnv._RESPONSE_FIFO,
            ready_flag=LocalCSVAnalysisEnv._READY_FLAG,
        )
        start_command = self._START_COMMAND_TEMPLATE.format(
            command_fifo=LocalCSVAnalysisEnv._COMMAND_FIFO,
            response_fifo=LocalCSVAnalysisEnv._RESPONSE_FIFO,
            ready_flag=LocalCSVAnalysisEnv._READY_FLAG,
            worker_path=LocalCSVAnalysisEnv._WORKER_PATH,
            worker_b64=base64.b64encode(worker_code.encode("utf-8")).decode("utf-8"),
            pip_install_packages=shlex.quote(pip_install_packages),
        )

        self.timeout_per_command_seconds = timeout_per_command_seconds
        self.sandbox_client = AsyncSandboxClient()
        self.sandbox_request = CreateSandboxRequest(
            name="csv-agent-python",
            docker_image="python:3.11-slim",
            start_command=start_command,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=0,
            timeout_minutes=timeout_minutes,
        )

    async def _bash(self, sandbox_id: str, command: str) -> str:
        await self.sandbox_client.wait_for_creation(sandbox_id)
        try:
            result = await asyncio.wait_for(
                self.sandbox_client.execute_command(sandbox_id, command),
                timeout=self.timeout_per_command_seconds,
            )
        except asyncio.TimeoutError:
            return f"Error: Command timed out after {self.timeout_per_command_seconds}s"

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stderr:
            return f"{stdout}\nstderr:\n{stderr}".strip()
        return stdout or "(no output)"

    async def _wait_for_worker_ready(self, sandbox_id: str) -> None:
        ready_flag = LocalCSVAnalysisEnv._READY_FLAG
        command = textwrap.dedent(
            f"""
            bash -lc '
            for i in $(seq 1 400); do
              if [ -f "{ready_flag}" ]; then
                exit 0
              fi
              sleep 0.1
            done
            echo "python worker failed to start" >&2
            exit 1
            '
            """
        )
        output = await self._bash(sandbox_id, command)
        if "python worker failed" in output:
            raise TimeoutError(output)

    async def setup_state(self, state: dict, **kwargs) -> dict:
        sandbox = await self.sandbox_client.create(self.sandbox_request)
        sandbox_id = sandbox.id
        try:
            await self.sandbox_client.wait_for_creation(sandbox_id)
            await self._wait_for_worker_ready(sandbox_id)
            await self.sandbox_client.upload_file(
                sandbox_id,
                "/data.csv",
                str(self.csv_path),
                timeout=self.timeout_per_command_seconds,
            )

            state["sandbox_id"] = sandbox_id
            state["python_state"] = {"ready": True, "execution_count": 0}

            csv_setup = (
                SETUP_CODE
                + """
try:
    df = pd.read_csv("/data.csv", na_values=['?', 'NA', 'N/A', 'na', 'n/a'], keep_default_na=True)
except UnicodeDecodeError:
    df = pd.read_csv("/data.csv", encoding='latin-1', na_values=['?', 'NA', 'N/A', 'na', 'n/a'], keep_default_na=True)
print(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
"""
            )
            await self.python(
                code=csv_setup,
                sandbox_id=sandbox_id,
                python_state=state["python_state"],
            )
            return state
        except Exception:
            await self.destroy_sandbox(sandbox_id)
            raise

    async def python(
        self, code: str, sandbox_id: str, python_state: dict | None = None, **kwargs
    ) -> str:
        payload = json.dumps({"code": code})
        payload_b64 = base64.b64encode(payload.encode()).decode()
        command = textwrap.dedent(
            f"""
            python - <<'PY'
import base64
import sys

data = base64.b64decode('{payload_b64}').decode()
with open('{LocalCSVAnalysisEnv._COMMAND_FIFO}', 'w', encoding='utf-8') as command_file:
    command_file.write(data)
with open('{LocalCSVAnalysisEnv._RESPONSE_FIFO}', 'r', encoding='utf-8') as response_file:
    sys.stdout.write(response_file.read())
PY
            """
        )
        raw_response = await self._bash(sandbox_id, command)
        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError:
            return f"Error: Failed to parse worker response.\n{raw_response[:500]}"

        if python_state is not None:
            python_state["execution_count"] = response.get("execution_count", 0)
        return LocalCSVAnalysisEnv._format_response(self, response)

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        try:
            await self.sandbox_client.delete(sandbox_id)
        except Exception:
            pass


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
        sandbox_backend: str = "docker",
        sandbox_pip_install_packages: str = DEFAULT_SANDBOX_PIP_PACKAGES,
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
            sandbox_backend: Code execution backend: "docker" or "prime"
            sandbox_pip_install_packages: Packages installed in Prime sandboxes
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
        if sandbox_backend not in {"docker", "prime"}:
            raise ValueError('sandbox_backend must be "docker" or "prime"')
        self.sandbox_backend = sandbox_backend
        self.sandbox_pip_install_packages = sandbox_pip_install_packages

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

        if self.sandbox_backend == "prime":
            env = PrimeSandboxCSVAnalysisEnv(
                csv_path=str(csv_source),
                pip_install_packages=self.sandbox_pip_install_packages,
            )
        else:
            env = LocalCSVAnalysisEnv(
                csv_path=str(csv_source),
                pip_install_packages=self.sandbox_pip_install_packages,
            )
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
