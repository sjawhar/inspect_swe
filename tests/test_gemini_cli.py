"""Regression tests for Gemini CLI invocation behavior."""

import asyncio
import importlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import BRIDGE_REQUEST_HEADERS, GenerateConfig, ModelOutput
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions
from inspect_swe._gemini_cli._events.consumer import GeminiConsumer
from inspect_swe._gemini_cli.gemini_cli import _unattended_gemini_command
from inspect_swe._util.centaur import (
    CentaurOptions,
    CentaurSession,
    reset_recorder_preserving_session_exception,
)


def test_unattended_command_constructs_headless_argv_after_resume() -> None:
    """Prevent a resumed Gemini turn from entering interactive mode on closed stdin."""
    assert _unattended_gemini_command(
        [
            "gemini",
            "--model",
            "gemini-2.5-pro",
            "--output-format",
            "text",
            "--yolo",
            "--resume",
            "latest",
        ],
        "create the required files",
    ) == [
        "gemini",
        "--model",
        "gemini-2.5-pro",
        "--output-format",
        "text",
        "--yolo",
        "--resume",
        "latest",
        "--prompt",
        "create the required files",
    ]


@dataclass(frozen=True)
class _ExecResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    success: bool = True


_TELEMETRY_PATH = "/home/cc/.gemini/inspect-swe.otel.json"
_TELEMETRY_CREATE = ["bash", "-c", f"umask 077; : > {_TELEMETRY_PATH}"]


class _Sandbox:
    def __init__(self) -> None:
        self.remote_calls: list[tuple[list[str], ExecRemoteAwaitableOptions, bool]] = []
        self.exec_calls: list[tuple[list[str], str | None]] = []
        self.files: dict[str, str] = {}
        self.settings: str | None = None

    async def exec(
        self, cmd: list[str], *, user: str | None, cwd: str | None = None
    ) -> _ExecResult:
        # sandbox_exec forwards cwd; none of these provisioning calls set one
        assert cwd is None
        self.exec_calls.append((cmd, user))
        if cmd == ["sh", "-c", "echo $HOME"]:
            return _ExecResult(stdout="/home/cc\n")
        if cmd == ["mkdir", "-p", "/home/cc/.gemini"]:
            return _ExecResult()
        assert cmd == _TELEMETRY_CREATE
        self.files[_TELEMETRY_PATH] = ""
        return _ExecResult()

    async def write_file(self, path: str, contents: str) -> None:
        # only settings.json may be written without a user: the CLI appends to
        # the telemetry stream itself, so a userless (root-owned) otel file
        # makes it exit 1
        assert path == "/home/cc/.gemini/settings.json"
        self.files[path] = contents
        self.settings = contents

    async def read_file(self, path: str) -> str:
        return self.files[path]

    async def exec_remote(
        self,
        cmd: list[str],
        *,
        options: ExecRemoteAwaitableOptions,
        stream: bool,
    ) -> _ExecResult:
        self.remote_calls.append((cmd, options, stream))
        return _ExecResult()


class _Store:
    def get(self, key: str, default: int) -> int:
        assert key == "gemini_cli_model_port"
        return default

    def set(self, key: str, value: int) -> None:
        assert key == "gemini_cli_model_port"
        assert value == 3001


class _Consumer:
    def __init__(self, **_kwargs: object) -> None:
        pass

    async def refresh(self, _command: str) -> None:
        pass

    async def finalize(self) -> None:
        pass

    def reset(self) -> None:
        pass


def test_unattended_factory_executes_headless_cli() -> None:
    """Prevent false-centaur calls from returning before the Gemini subprocess."""
    module = importlib.import_module("inspect_swe._gemini_cli.gemini_cli")
    state = AgentState(messages=[])
    sbox = _Sandbox()
    bridge = SimpleNamespace(port=8901, mcp_server_configs=[], state=state)

    @asynccontextmanager
    async def bridge_context(
        *_args: object, **_kwargs: object
    ) -> AsyncIterator[SimpleNamespace]:
        yield bridge

    with (
        patch.object(module, "sandbox_env", return_value=sbox),
        patch.object(module, "store", return_value=_Store()),
        patch.object(module, "resolve_agent_cwd", AsyncMock(return_value="/workspace")),
        patch.object(
            module,
            "ensure_gemini_cli_setup",
            AsyncMock(return_value=("/opt/gemini", "/opt/node/bin/node")),
        ),
        patch.object(module, "sandbox_agent_bridge", bridge_context),
        patch.object(module, "build_user_prompt", return_value=("write files", False)),
        patch.object(module, "GeminiConsumer", _Consumer),
    ):
        asyncio.run(module.gemini_cli(version="0.58.0")(state))
    assert sbox.settings is not None
    assert '"outfile": "/home/cc/.gemini/inspect-swe.otel.json"' in sbox.settings
    assert (_TELEMETRY_CREATE, None) in sbox.exec_calls
    assert sbox.files[_TELEMETRY_PATH] == ""

    assert len(sbox.remote_calls) == 1
    command, options, stream = sbox.remote_calls[0]
    assert command == [
        "bash",
        "-c",
        'exec 0</dev/null; "$@"',
        "bash",
        "/opt/gemini",
        "--model",
        "gemini-2.5-pro",
        "--output-format",
        "text",
        "--yolo",
        "--prompt",
        "write files",
    ]
    assert options.cwd == "/workspace"
    assert options.user is None
    assert options.concurrency is False
    assert stream is False


def test_sandbox_factory_leaves_native_event_recording_unowned() -> None:
    """An attached Gemini binary never receives the native event sink."""
    module = importlib.import_module("inspect_swe._gemini_cli.gemini_cli")
    state = AgentState(messages=[])
    sbox = _Sandbox()
    bridge = SimpleNamespace(port=8901, mcp_server_configs=[], state=state)
    bridge_options: dict[str, object] = {}

    @asynccontextmanager
    async def bridge_context(
        *_args: object, **kwargs: object
    ) -> AsyncIterator[SimpleNamespace]:
        bridge_options.update(kwargs)
        yield bridge

    consumer_factory = Mock()
    with (
        patch.object(module, "sandbox_env", return_value=sbox),
        patch.object(module, "store", return_value=_Store()),
        patch.object(module, "resolve_agent_cwd", AsyncMock(return_value="/workspace")),
        patch.object(
            module,
            "ensure_gemini_cli_setup",
            AsyncMock(return_value=("/opt/gemini", "/opt/node/bin/node")),
        ),
        patch.object(module, "sandbox_agent_bridge", bridge_context),
        patch.object(module, "build_user_prompt", return_value=("write files", False)),
        patch.object(module, "GeminiConsumer", consumer_factory),
    ):
        asyncio.run(module.gemini_cli(version="sandbox")(state))

    consumer_factory.assert_not_called()
    assert bridge_options["model_event_sink"] is None
    assert bridge_options["model_event_metadata_headers"] is None
    assert sbox.settings is not None
    assert '"telemetry"' not in sbox.settings
    assert [cmd for cmd, _user in sbox.exec_calls if cmd == _TELEMETRY_CREATE] == []
    assert _TELEMETRY_PATH not in sbox.files


def test_centaur_quit_before_first_gemini_call_drains_initialized_telemetry() -> None:
    """An immediate quit reads the owned empty telemetry stream, not a missing file."""
    module = importlib.import_module("inspect_swe._gemini_cli.gemini_cli")
    state = AgentState(messages=[])
    sbox = _Sandbox()
    bridge = SimpleNamespace(port=8901, mcp_server_configs=[], state=state)

    @asynccontextmanager
    async def bridge_context(
        *_args: object, **_kwargs: object
    ) -> AsyncIterator[SimpleNamespace]:
        yield bridge

    async def quit_before_first_gemini_call(
        options: CentaurOptions,
        instructions: str,
        bashrc: str,
        session: CentaurSession,
        commands_filter: object | None = None,
    ) -> AgentState:
        del options, instructions, bashrc, commands_filter
        assert session.refresh is not None
        assert session.finalize is not None
        await session.refresh("quit")
        await session.finalize()
        return session.state

    with (
        patch.object(module, "sandbox_env", return_value=sbox),
        patch.object(module, "store", return_value=_Store()),
        patch.object(module, "resolve_agent_cwd", AsyncMock(return_value="/workspace")),
        patch.object(
            module,
            "ensure_gemini_cli_setup",
            AsyncMock(return_value=("/opt/gemini", "/opt/node/bin/node")),
        ),
        patch.object(module, "sandbox_agent_bridge", bridge_context),
        patch.object(module, "build_user_prompt", return_value=("write files", False)),
        patch.object(module, "run_centaur", quit_before_first_gemini_call),
    ):
        asyncio.run(
            module.gemini_cli(centaur=CentaurOptions(), version="0.58.0", user="cc")(
                state
            )
        )

    # the stream the immediate quit drains is created by the agent's own user
    assert (_TELEMETRY_CREATE, "cc") in sbox.exec_calls
    assert sbox.files[_TELEMETRY_PATH] == ""
    assert sbox.remote_calls == []


def test_gemini_centaur_reset_preserves_cancellation_with_pending_identity() -> None:
    """Strict Gemini identity cleanup cannot replace an active cancellation."""
    consumer = GeminiConsumer()
    pending = ModelEvent(
        model="mock/model",
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_content("mock/model", "done"),
        metadata={
            BRIDGE_REQUEST_HEADERS: {
                "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
            }
        },
    )
    consumer.on_pending(pending)

    with pytest.raises(asyncio.CancelledError):
        try:
            raise asyncio.CancelledError()
        finally:
            reset_recorder_preserving_session_exception(consumer.reset)

    with pytest.raises(RuntimeError, match="unresolved bridge ModelEvent identities"):
        reset_recorder_preserving_session_exception(consumer.reset)
