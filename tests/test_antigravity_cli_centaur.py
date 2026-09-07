"""Behavioral coverage for Antigravity CLI's native Centaur boundary.

These tests drive the public factory and the native Centaur wrapper through their
real control flow. They keep sandbox execution and the human CLI outside this
unit while asserting the session that reaches that boundary.
"""

import asyncio
import importlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.agent._human.commands.command import HumanAgentCommand
from inspect_ai.model import ChatMessageUser
from inspect_ai.util import SandboxEnvironment
from inspect_swe._util.centaur import CentaurOptions, CentaurSession, CommandsFilter


class _Store:
    """In-memory per-sample port store used by the factory."""

    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def get(self, key: str, default: int) -> int:
        return self.values.get(key, default)

    def set(self, key: str, value: int) -> None:
        self.values[key] = value


class _Sandbox:
    """Structural sandbox fake for the factory's configuration writes."""

    async def exec(
        self, command: list[str], user: str | None = None
    ) -> SimpleNamespace:
        del user
        if command == ["sh", "-c", "echo $HOME"]:
            return SimpleNamespace(stdout="/home/operator\n")
        return SimpleNamespace(stdout="")

    async def write_file(self, path: str, content: str) -> None:
        del path, content


def test_antigravity_cli_factory_forwards_custom_bridge_and_centaur_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native operator run retains its requested bridge and Centaur inputs.

    This catches the public-factory regression where a custom resolver, aliases,
    or command filter is accepted but dropped before the bridge or the Centaur
    dispatcher.
    """
    module = importlib.import_module("inspect_swe._antigravity_cli.antigravity_cli")
    initial_state = AgentState(messages=[ChatMessageUser(content="Review the patch.")])
    bridge_state = AgentState(messages=[ChatMessageUser(content="Native traffic.")])
    sandbox = _Sandbox()
    expected_sandbox: object = sandbox
    bridge_options: dict[str, object] = {}
    centaur_call: dict[str, object] = {}

    model_aliases = {"operator-google": "google/custom-target"}

    def defer_to_configured_google_model(_requested: str) -> None:
        return None

    def operator_commands_filter(
        commands: list[HumanAgentCommand],
    ) -> list[HumanAgentCommand]:
        return list(reversed(commands))

    @asynccontextmanager
    async def bridge_context(
        state: AgentState, **kwargs: object
    ) -> AsyncIterator[SimpleNamespace]:
        assert state is initial_state
        bridge_options.update(kwargs)
        yield SimpleNamespace(
            port=3001,
            mcp_server_configs=[],
            bridged_tools={},
            state=bridge_state,
        )

    async def capture_centaur_dispatch(
        *,
        options: CentaurOptions,
        agy_cmd: list[str],
        agent_env: dict[str, str],
        session: CentaurSession,
        commands_filter: CommandsFilter | None = None,
    ) -> AgentState:
        centaur_call.update(
            options=options,
            agy_cmd=agy_cmd,
            agent_env=agent_env,
            session=session,
            commands_filter=commands_filter,
        )
        return session.state

    async def resolve_cwd(_sandbox: object, _user: str | None, _cwd: str | None) -> str:
        return "/worktree/repository"

    async def ensure_binary(*_args: object) -> str:
        return "/opt/agent-cli/agy"

    sample_store = _Store()

    def current_store() -> _Store:
        return sample_store

    monkeypatch.setattr(module, "sandbox_env", lambda _sandbox: sandbox)
    monkeypatch.setattr(module, "store", current_store)
    monkeypatch.setattr(module, "sandbox_agent_bridge", bridge_context)
    monkeypatch.setattr(module, "resolve_agent_cwd", resolve_cwd)
    monkeypatch.setattr(module, "ensure_agent_binary_installed", ensure_binary)
    monkeypatch.setattr(
        module, "_run_antigravity_cli_centaur", capture_centaur_dispatch
    )

    centaur_options = CentaurOptions(answer=False)
    agent = module.antigravity_cli(
        centaur=centaur_options,
        model="google/custom-target",
        model_resolver=defer_to_configured_google_model,
        model_aliases=model_aliases,
        commands_filter=operator_commands_filter,
        agy_model="gemini-3.7-flash",
        effort="high",
        cwd="/worktree/repository",
        env={"OPERATOR_MARKER": "enabled"},
        user="operator",
        accumulate_conversations=True,
        version="sandbox",
    )
    result = asyncio.run(agent(initial_state))

    assert result is bridge_state
    assert bridge_options["model"] == "inspect/google/custom-target"
    assert bridge_options["model_aliases"] is model_aliases
    assert bridge_options["model_resolver"] is defer_to_configured_google_model
    assert bridge_options["accumulate_conversations"] is True
    assert centaur_call["options"] is centaur_options
    assert centaur_call["commands_filter"] is operator_commands_filter

    session = centaur_call["session"]
    assert isinstance(session, CentaurSession)
    assert session.state is bridge_state
    assert session.invocation == (
        "/opt/agent-cli/agy",
        "--model",
        "gemini-3.7-flash",
        "--effort",
        "high",
    )
    assert session.environment == {
        "GOOGLE_GEMINI_BASE_URL": "http://localhost:3001",
        "GEMINI_API_KEY": "api-key",
        "AGY_CLI_DISABLE_AUTO_UPDATE": "1",
        "AGY_CLI_HIDE_LOGO": "1",
        "HOME": "/home/operator",
        "OPERATOR_MARKER": "enabled",
    }
    assert session.cwd == "/worktree/repository"
    assert session.user == "operator"
    assert session.sandbox is expected_sandbox
    assert session.bridge_port == 3001
    assert session.session_id is None


def test_antigravity_centaur_wrapper_preserves_ready_session_and_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `agy` alias and ready callback receive one intact native session.

    A state-style Centaur call loses the invocation, user, working directory,
    environment, and readiness lifecycle the human CLI needs to resume `agy`.
    """
    module = importlib.import_module("inspect_swe._antigravity_cli.antigravity_cli")
    sandbox = _Sandbox()
    bridge_state = AgentState(messages=[ChatMessageUser(content="Native traffic.")])
    agy_command = [
        "/opt/agent-cli/agy",
        "--model",
        "gemini-3.7-flash",
        "--effort",
        "high",
        "--output-format",
        "text",
        "--disable-slash-commands",
    ]
    environment = {
        "GOOGLE_GEMINI_BASE_URL": "http://localhost:3001",
        "GEMINI_API_KEY": "api-key",
        "HOME": "/home/operator",
        "OPERATOR_MARKER": "enabled",
    }
    session = CentaurSession(
        state=bridge_state,
        invocation=tuple(agy_command),
        environment=environment,
        cwd="/worktree/repository with spaces",
        user="operator",
        sandbox=cast(SandboxEnvironment, sandbox),
        bridge_port=3001,
        session_id=None,
    )
    lifecycle: list[str] = []

    @asynccontextmanager
    async def on_ready(ready_session: CentaurSession) -> AsyncIterator[None]:
        assert ready_session is session
        assert ready_session.state is bridge_state
        assert ready_session.invocation == tuple(agy_command)
        assert ready_session.environment is environment
        assert ready_session.cwd == "/worktree/repository with spaces"
        assert ready_session.user == "operator"
        lifecycle.append("entered")
        yield
        lifecycle.append("exited")

    def operator_commands_filter(
        commands: list[HumanAgentCommand],
    ) -> list[HumanAgentCommand]:
        return list(reversed(commands))

    async def capture_run_centaur(
        options: CentaurOptions,
        instructions: str,
        bashrc: str,
        passed_session: CentaurSession,
        commands_filter: CommandsFilter | None = None,
    ) -> AgentState:
        assert options is centaur_options
        assert "Antigravity CLI" in instructions
        assert passed_session is session
        assert commands_filter is operator_commands_filter
        assert bashrc == "\n".join(
            [
                'export GOOGLE_GEMINI_BASE_URL="http://localhost:3001"',
                'export GEMINI_API_KEY="api-key"',
                'export OPERATOR_MARKER="enabled"',
                "",
                (
                    "alias agy='/opt/agent-cli/agy --model gemini-3.7-flash --effort "
                    "high --output-format text --disable-slash-commands'"
                ),
                "cd -- '/worktree/repository with spaces'",
            ]
        )
        assert options.on_ready is not None
        async with options.on_ready(passed_session):
            assert lifecycle == ["entered"]
        return passed_session.state

    centaur_options = CentaurOptions(answer=False, on_ready=on_ready)
    monkeypatch.setattr(module, "run_centaur", capture_run_centaur)

    result = asyncio.run(
        module._run_antigravity_cli_centaur(
            options=centaur_options,
            agy_cmd=agy_command,
            agent_env=environment,
            session=session,
            commands_filter=operator_commands_filter,
        )
    )

    assert result is bridge_state
    assert lifecycle == ["entered", "exited"]
