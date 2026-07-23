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


@pytest.mark.asyncio
async def test_setup_failure_destroys_started_sandbox(monkeypatch) -> None:
    environment = LocalCSVAnalysisEnv(
        "data/fixtures/smoke/student_performance/data.csv",
        session_id="setup-failure",
    )
    destroyed: list[str] = []

    async def ensure_image() -> None:
        return None

    async def run_docker(*args, **_kwargs):
        if args[0] == "cp":
            raise RuntimeError("copy failed")
        return "", ""

    async def destroy_sandbox(sandbox_id: str) -> None:
        destroyed.append(sandbox_id)

    monkeypatch.setattr(environment, "_ensure_image", ensure_image)
    monkeypatch.setattr(environment, "_run_docker", run_docker)
    monkeypatch.setattr(environment, "destroy_sandbox", destroy_sandbox)
    state: dict = {}

    with pytest.raises(RuntimeError, match="copy failed"):
        await environment.setup_state(state)

    assert len(destroyed) == 1
    assert destroyed[0].startswith("csv-sandbox-setup-failure-")
    assert state == {}
