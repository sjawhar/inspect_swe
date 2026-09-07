"""Fast regression coverage for OpenCode's native bridge configuration."""

import asyncio
import importlib
import inspect
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.agent import AgentState, HumanAgentCommand
from inspect_swe._util.centaur import (
    CentaurOptions,
    CentaurSession,
    CommandsFilter,
)


class _ExecResult:
    def __init__(self, stdout: str = "") -> None:
        self.success = True
        self.stderr = ""
        self.stdout = stdout


class _Sandbox:
    def __init__(self) -> None:
        self.exec_calls: list[list[str]] = []
        self.files: dict[str, str] = {}

    async def exec(self, cmd: list[str], *, user: str | None) -> _ExecResult:
        assert user is None
        self.exec_calls.append(cmd)
        if cmd == ["sh", "-c", "echo $HOME"]:
            return _ExecResult("/home/agent\n")
        if cmd == ["git", "-C", "/workspace", "rev-parse", "--show-toplevel"]:
            return _ExecResult("/workspace\n")
        if (
            len(cmd) >= 4
            and cmd[:2] == ["bash", "-c"]
            and cmd[3] == "opencode-config-paths"
        ):
            return _ExecResult()
        if (
            len(cmd) >= 4
            and cmd[:2] == ["bash", "-c"]
            and cmd[3] == "opencode-config-home"
        ):
            return _ExecResult()
        assert cmd == ["mkdir", "-p", "/home/agent/.inspect_swe/opencode"]
        return _ExecResult()

    async def write_file(self, path: str, contents: str) -> None:
        self.files[path] = contents


class _Store:
    def get(self, key: str, default: int) -> int:
        assert key == "opencode_model_port"
        return default

    def set(self, key: str, value: int) -> None:
        assert key == "opencode_model_port"
        assert value == 3001


@pytest.mark.parametrize(
    ("opencode_model", "expected_provider", "expected_title_agent"),
    [
        (
            "google/gdm-fsm-plum",
            {
                "anthropic": {"options": {"baseURL": "http://localhost:8901/v1"}},
                "google": {
                    "npm": "@ai-sdk/google",
                    "models": {"gdm-fsm-plum": {"name": "gdm-fsm-plum"}},
                    "options": {
                        "apiKey": "sk-none",
                        "baseURL": "http://localhost:8901/v1beta",
                    },
                },
            },
            {"title": {"model": "google/gdm-fsm-plum"}},
        ),
        (
            "openai/gpt-5",
            {
                "anthropic": {"options": {"baseURL": "http://localhost:8901/v1"}},
                "openai": {"options": {"baseURL": "http://localhost:8901/v1"}},
                "google": {
                    "npm": "@ai-sdk/google",
                    "options": {
                        "apiKey": "sk-none",
                        "baseURL": "http://localhost:8901/v1beta",
                    },
                },
            },
            None,
        ),
    ],
)
def test_native_factory_defaults_bare_operator_to_selected_model(
    opencode_model: str,
    expected_provider: dict[str, object],
    expected_title_agent: dict[str, dict[str, str]] | None,
) -> None:
    module = importlib.import_module("inspect_swe._opencode.opencode")
    state = AgentState(messages=[])
    sbox = _Sandbox()
    bridge = SimpleNamespace(port=8901, mcp_server_configs=[], state=state)
    bridge_options: dict[str, object] = {}
    centaur_call: dict[str, object] = {}
    config_dependency_seed = AsyncMock()

    def resolver(_requested: str) -> None:
        return None

    @asynccontextmanager
    async def bridge_context(
        *_args: object, **kwargs: object
    ) -> AsyncIterator[SimpleNamespace]:
        bridge_options.update(kwargs)
        yield bridge

    async def capture_centaur(
        options: CentaurOptions,
        instructions: str,
        bashrc: str,
        session: CentaurSession,
        *,
        commands_filter: CommandsFilter | None = None,
    ) -> AgentState:
        centaur_call.update(
            options=options,
            instructions=instructions,
            bashrc=bashrc,
            session=session,
            commands_filter=commands_filter,
        )
        return session.state

    def identity_commands_filter(
        commands: list[HumanAgentCommand],
    ) -> list[HumanAgentCommand]:
        return commands

    commands_filter: CommandsFilter = identity_commands_filter
    with (
        patch.object(module, "sandbox_env", return_value=sbox),
        patch.object(module, "store", return_value=_Store()),
        patch.object(module, "resolve_agent_cwd", AsyncMock(return_value="/workspace")),
        patch.object(
            module,
            "ensure_opencode_setup",
            AsyncMock(return_value=("/opt/opencode", ["/opt/node/bin", "/opt/rg"])),
        ),
        patch.object(module, "sandbox_agent_bridge", bridge_context),
        patch.object(module, "build_user_prompt", return_value=("write files", False)),
        patch.object(module, "run_centaur", capture_centaur),
        patch.object(
            module, "seed_opencode_config_dependencies", config_dependency_seed
        ),
    ):
        asyncio.run(
            module.opencode(
                centaur=CentaurOptions(answer=False),
                commands_filter=commands_filter,
                opencode_model=opencode_model,
                model_resolver=resolver,
            )(state)
        )
    config = json.loads(sbox.files["/home/agent/.inspect_swe/opencode/opencode.json"])
    assert sbox.exec_calls[:3] == [
        ["sh", "-c", "echo $HOME"],
        ["mkdir", "-p", "/home/agent/.inspect_swe/opencode"],
        ["git", "-C", "/workspace", "rev-parse", "--show-toplevel"],
    ]
    assert len(sbox.exec_calls) == 5
    assert sbox.exec_calls[3][3:] == [
        "opencode-config-paths",
        "/workspace",
        "/workspace",
    ]
    assert sbox.exec_calls[4][3:] == [
        "opencode-config-home",
        "/home/agent/.opencode",
    ]
    config_dependency_seed.assert_awaited_once_with(
        sbox,
        "/opt/opencode",
        "/opt/node/bin/node",
        ["/opt/node/bin", "/opt/rg"],
        ["/home/agent/.config/opencode"],
        None,
        None,
    )

    assert bridge_options["model_resolver"] is resolver
    assert centaur_call["commands_filter"] is commands_filter
    assert config["provider"] == expected_provider
    assert config["model"] == opencode_model
    if expected_title_agent is None:
        assert "agent" not in config
    else:
        assert config["agent"] == expected_title_agent
    bashrc = centaur_call["bashrc"]
    assert isinstance(bashrc, str)
    assert "alias opencode=/opt/opencode" in bashrc
    assert "alias opencode='/opt/opencode run" not in bashrc
    assert "GOOGLE_GENERATIVE_AI_API_KEY" in bashrc
    session = centaur_call["session"]
    assert isinstance(session, CentaurSession)
    assert session.invocation == (
        "/opt/opencode",
        "run",
        "--model",
        opencode_model,
        "--format",
        "json",
    )


def test_opencode_exposes_resolver_without_provider_configuration() -> None:
    """Keep bridge routing public without leaking OpenCode provider internals."""
    module = importlib.import_module("inspect_swe._opencode.opencode")
    parameters = inspect.signature(module.opencode).parameters

    assert parameters["model_resolver"].default is None
    assert "provider_config" not in parameters
