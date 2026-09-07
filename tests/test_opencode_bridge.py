"""Fast regression coverage for OpenCode's native bridge configuration."""

import asyncio
import importlib
import inspect
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

from inspect_ai.agent import AgentState
from inspect_swe._util.centaur import CentaurOptions, CentaurSession, CommandsFilter


class _ExecResult:
    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout


class _Sandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def exec(self, cmd: list[str], *, user: str | None) -> _ExecResult:
        assert user is None
        if cmd == ["sh", "-c", "echo $HOME"]:
            return _ExecResult("/home/agent\n")
        assert cmd == ["mkdir", "-p", "/home/agent/.config/opencode"]
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


def test_native_google_factory_uses_google_catalog_and_bare_operator_alias() -> None:
    """Keep the user's `opencode run --model google/...` command unwrapped."""
    module = importlib.import_module("inspect_swe._opencode.opencode")
    state = AgentState(messages=[])
    sbox = _Sandbox()
    bridge = SimpleNamespace(port=8901, mcp_server_configs=[], state=state)
    bridge_options: dict[str, object] = {}
    centaur_call: dict[str, object] = {}

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

    commands_filter: CommandsFilter = lambda commands: commands
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
    ):
        asyncio.run(
            module.opencode(
                centaur=CentaurOptions(answer=False),
                commands_filter=commands_filter,
                opencode_model="google/gdm-fsm-plum",
                model_resolver=resolver,
            )(state)
        )

    assert bridge_options["model_resolver"] is resolver
    assert centaur_call["commands_filter"] is commands_filter
    config = json.loads(sbox.files["/home/agent/.config/opencode/opencode.json"])
    assert config["provider"] == {
        "anthropic": {"options": {"baseURL": "http://localhost:8901/v1"}},
        "google": {
            "npm": "@ai-sdk/google",
            "models": {"gdm-fsm-plum": {"name": "gdm-fsm-plum"}},
            "options": {
                "apiKey": "sk-none",
                "baseURL": "http://localhost:8901/v1beta",
            },
        },
    }
    assert config["agent"] == {"title": {"model": "google/gdm-fsm-plum"}}
    bashrc = centaur_call["bashrc"]
    assert isinstance(bashrc, str)
    assert "alias opencode=/opt/opencode" in bashrc
    assert "alias opencode='/opt/opencode run" not in bashrc
    session = centaur_call["session"]
    assert isinstance(session, CentaurSession)
    assert session.invocation == (
        "/opt/opencode",
        "run",
        "--model",
        "google/gdm-fsm-plum",
        "--format",
        "json",
    )

def test_opencode_exposes_resolver_without_provider_configuration() -> None:
    """Keep bridge routing public without leaking OpenCode provider internals."""
    module = importlib.import_module("inspect_swe._opencode.opencode")
    parameters = inspect.signature(module.opencode).parameters

    assert parameters["model_resolver"].default is None
    assert "provider_config" not in parameters
