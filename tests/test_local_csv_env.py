"""Focused tests for bounded local Docker command execution."""

import asyncio
import inspect

import pytest

from src.envs.csv_env import (
    DOCKER_COMMAND_TIMEOUT_SECONDS,
    LocalCSVAnalysisEnv,
)


def test_docker_commands_have_a_finite_default_timeout() -> None:
    default = (
        inspect.signature(LocalCSVAnalysisEnv._run_docker).parameters["timeout"].default
    )

    assert default == DOCKER_COMMAND_TIMEOUT_SECONDS
    assert default > 0


@pytest.mark.asyncio
async def test_timed_out_docker_command_is_killed(monkeypatch) -> None:
    class HangingProcess:
        returncode = None
        killed = False

        async def communicate(self):
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            return self.returncode

    process = HangingProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    environment = LocalCSVAnalysisEnv.__new__(LocalCSVAnalysisEnv)

    with pytest.raises(TimeoutError, match="Docker command timed out"):
        await environment._run_docker("stop", "sandbox", timeout=0.001)

    assert process.killed
