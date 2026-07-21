"""Focused tests for bounded local Docker command execution."""

import asyncio
import inspect

import pytest

import src.envs.csv_env as csv_env_module
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
    class HangingTransport:
        closed = False

        def close(self):
            self.closed = True

    class HangingProcess:
        returncode = None
        killed = False
        _transport = HangingTransport()

        async def communicate(self):
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            await asyncio.Event().wait()

    process = HangingProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(csv_env_module, "DOCKER_KILL_TIMEOUT_SECONDS", 0.001)
    environment = LocalCSVAnalysisEnv.__new__(LocalCSVAnalysisEnv)

    with pytest.raises(TimeoutError, match="Docker command timed out"):
        await environment._run_docker("stop", "sandbox", timeout=0.001)

    assert process.killed
    assert process._transport.closed
