"""Tests for replaying a recorded turn boundary before continuation."""

import json

import pytest
from csv_spec import Question, TrajectoryPrefix

from src.core.config import DataConfig, ExecutionConfig, ModelConfig, TaskConfig
from src.core.environment import Environment, PrefixReplayError


class FakeSandbox:
    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.destroyed = False

    async def python(self, code: str, **_kwargs) -> str:
        return self.outputs[code]

    async def destroy_sandbox(self, _sandbox_id: str) -> None:
        self.destroyed = True


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    async def __call__(self, _messages: list[dict]) -> str:
        return next(self.responses)


def _turn(stdout: str = "3") -> dict:
    return {
        "turn_index": 0,
        "reasoning": "Count the available rows first.",
        "code": "print(len(df))",
        "execution": {
            "success": True,
            "stdout": stdout,
            "stderr": "",
            "hooks": [],
            "submitted_answer": None,
        },
    }


def _prefix(stdout: str = "3") -> TrajectoryPrefix:
    return TrajectoryPrefix(
        prefix_id="episode-1:1",
        episode_id="episode-1",
        csv_source="data/fixtures/smoke/student_performance/data.csv",
        system_prompt="Solve the CSV task using Python.",
        question_text="How many rows are present?",
        turns=[_turn(stdout)],
        max_turns=3,
    )


def _environment(sandbox: FakeSandbox, llm=None) -> Environment:
    return Environment(
        data=DataConfig(csv_path="data/fixtures/smoke/student_performance/data.csv"),
        model=ModelConfig(model_name="fake-model"),
        execution=ExecutionConfig(max_turns=3),
        task=TaskConfig(
            mode="student",
            question=Question(question_text="How many rows are present?"),
        ),
        env=sandbox,
        state={"sandbox_id": "fake", "python_state": {}},
        llm=llm or FakeLLM([]),
    )


@pytest.mark.asyncio
async def test_replay_restores_conversation_and_turn_count() -> None:
    sandbox = FakeSandbox({"print(len(df))": "3"})
    environment = _environment(sandbox)
    environment.init_state()

    await environment.replay_turns(_prefix().turns)

    assert environment.current_turn == 1
    assert environment.code_cells == ["print(len(df))"]
    assert [message["role"] for message in environment.conversation.messages] == [
        "assistant",
        "user",
    ]
    assert environment.conversation.system_prompt != _prefix().system_prompt


@pytest.mark.asyncio
async def test_replay_rejects_divergent_execution() -> None:
    environment = _environment(FakeSandbox({"print(len(df))": "3"}))
    environment.init_state()

    with pytest.raises(PrefixReplayError, match="stdout"):
        await environment.replay_turns(_prefix(stdout="4").turns)


@pytest.mark.asyncio
async def test_rollout_can_continue_after_replay_and_cleans_up() -> None:
    continuation_code = "answer = len(df)\nsubmit(answer)"
    submission = {"__csv_agent_answer__": 3}
    sandbox = FakeSandbox(
        {
            "print(len(df))": "3",
            continuation_code: f"✓ Submitted: {json.dumps(submission)}",
        }
    )
    llm = FakeLLM(
        [f"The row count is now established.\n```python\n{continuation_code}\n```"]
    )
    environment = _environment(sandbox, llm=llm)

    result = await environment.rollout_from_prefix(_prefix())

    assert result.submitted_answer == 3
    assert result.current_turn == 2
    assert result.conversation.system_prompt == _prefix().system_prompt
    assert sandbox.destroyed is True
