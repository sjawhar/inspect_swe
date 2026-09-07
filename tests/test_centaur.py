"""Centaur operator lifecycle and live bridge-state regression coverage."""

import asyncio
import importlib
import inspect
import os
import shlex
import subprocess
import sys
from argparse import Namespace
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import inspect_swe._claude_code.claude_code as claude_code_mod
import pytest
from inspect_ai.agent import AgentState
from inspect_ai.agent._human.commands.command import (
    HumanAgentCommand,
    call_human_agent,
)
from inspect_ai.agent._human.install import human_agent_commands
from inspect_ai.agent._human.state import HumanAgentState
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.util import SandboxEnvironment
from inspect_swe import CentaurFinalize as PublicCentaurFinalize
from inspect_swe import CentaurOptions as PublicCentaurOptions
from inspect_swe import CentaurReady as PublicCentaurReady
from inspect_swe import CentaurRefresh as PublicCentaurRefresh
from inspect_swe import CentaurSession as PublicCentaurSession
from inspect_swe import CommandsFilter as PublicCommandsFilter
from inspect_swe._claude_code._events.live_consumer import LiveConsumer
from inspect_swe._util import centaur as centaur_mod
from inspect_swe._util.centaur import (
    CentaurOptions,
    CentaurReady,
    CentaurRefresh,
    CentaurSession,
    CommandsFilter,
    run_centaur,
)
from pydantic import JsonValue


def _session(state: AgentState) -> CentaurSession:
    return CentaurSession(
        state=state,
        invocation=("/opt/agent-cli/claude", "--model", "claude-sonnet-4-5"),
        environment={"ANTHROPIC_BASE_URL": "http://localhost:12345"},
        cwd="/workdir",
        user="agent",
        sandbox=MagicMock(spec=SandboxEnvironment),
        bridge_port=12345,
        session_id="native-session",
    )


def _commands_filter(
    commands: list[HumanAgentCommand],
) -> list[HumanAgentCommand]:
    return commands


def test_centaur_lifecycle_contract_is_public() -> None:
    assert PublicCentaurFinalize is centaur_mod.CentaurFinalize
    assert PublicCentaurOptions is CentaurOptions
    assert PublicCentaurSession is CentaurSession
    assert PublicCentaurReady is CentaurReady
    assert PublicCommandsFilter is CommandsFilter
    assert PublicCentaurRefresh is CentaurRefresh


def test_claude_native_routing_options_are_keyword_only() -> None:
    parameters = inspect.signature(claude_code_mod.claude_code).parameters

    assert parameters["model_resolver"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["accumulate_conversations"].kind is inspect.Parameter.KEYWORD_ONLY


def test_run_centaur_enters_ready_session_only_after_human_cli_service_is_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_state = AgentState(messages=[ChatMessageUser(content="Solve the task.")])
    session = _session(bridge_state)
    lifecycle: list[str] = []
    captured: dict[str, object] = {}
    human_cli_service_started = False
    on_ready_from_human_cli: Callable[[], AbstractAsyncContextManager[None]] | None = (
        None
    )

    @asynccontextmanager
    async def on_ready(ready: CentaurSession) -> AsyncIterator[None]:
        assert human_cli_service_started
        assert ready is session
        assert ready.state is bridge_state
        assert ready.invocation == (
            "/opt/agent-cli/claude",
            "--model",
            "claude-sonnet-4-5",
        )
        assert ready.environment == {"ANTHROPIC_BASE_URL": "http://localhost:12345"}
        assert ready.cwd == "/workdir"
        assert ready.user == "agent"
        assert ready.bridge_port == 12345
        assert ready.session_id == "native-session"
        lifecycle.append("entered")
        yield
        lifecycle.append("exited")
        assert ready.state is bridge_state
        assert [message.text for message in ready.state.messages] == [
            "Solve the task.",
            "Native CLI response.",
        ]

    def fake_human_cli(
        *,
        answer: bool | str,
        intermediate_scoring: bool,
        record_session: bool,
        instructions: str,
        bashrc: str,
        user: str | None,
        commands_filter: CommandsFilter,
        on_ready: Callable[[], AbstractAsyncContextManager[None]] | None,
    ) -> str:
        nonlocal on_ready_from_human_cli
        captured.update(
            answer=answer,
            intermediate_scoring=intermediate_scoring,
            record_session=record_session,
            instructions=instructions,
            bashrc=bashrc,
            user=user,
            commands_filter=commands_filter,
        )
        on_ready_from_human_cli = on_ready
        return "human-cli-agent"

    async def fake_run(agent: object, state: AgentState) -> AgentState:
        nonlocal human_cli_service_started
        assert agent == "human-cli-agent"
        assert state is bridge_state
        assert lifecycle == []
        human_cli_service_started = True
        assert on_ready_from_human_cli is not None
        async with on_ready_from_human_cli():
            assert lifecycle == ["entered"]
            bridge_state.messages.append(
                ChatMessageAssistant(content="Native CLI response.")
            )
            returned_state = AgentState(
                messages=[ChatMessageUser(content="Solve the task.")]
            )
            returned_state.output = ModelOutput.from_content("human_agent", "submitted")
            return returned_state

    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", fake_run)

    result = asyncio.run(
        run_centaur(
            CentaurOptions(on_ready=on_ready),
            instructions="Instructions",
            bashrc="alias claude='/opt/agent-cli/claude'",
            session=session,
            commands_filter=_commands_filter,
        )
    )

    assert result is bridge_state
    assert session.state is bridge_state
    assert session.state.output.message.text == "submitted"
    assert lifecycle == ["entered", "exited"]
    assert captured["user"] == "agent"
    assert captured["commands_filter"] is _commands_filter


def test_run_centaur_scopes_the_human_lifecycle_to_the_named_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(AgentState(messages=[]))
    session.sandbox_name = "target"
    lifecycle: list[str] = []

    @contextmanager
    def fake_sandbox_default(name: str) -> Iterator[None]:
        assert name == "target"
        lifecycle.append("entered")
        try:
            yield
        finally:
            lifecycle.append("exited")

    def fake_human_cli(
        *,
        answer: bool | str,
        intermediate_scoring: bool,
        record_session: bool,
        instructions: str,
        bashrc: str,
        user: str | None,
        on_ready: Callable[[], AbstractAsyncContextManager[None]] | None,
    ) -> str:
        del (
            answer,
            intermediate_scoring,
            record_session,
            instructions,
            bashrc,
            user,
            on_ready,
        )
        assert lifecycle == ["entered"]
        lifecycle.append("human-cli")
        return "human-cli-agent"

    async def completed_run(agent: object, state: AgentState) -> AgentState:
        assert agent == "human-cli-agent"
        assert state is session.state
        assert lifecycle == ["entered", "human-cli"]
        lifecycle.append("run")
        return state

    monkeypatch.setattr(centaur_mod, "sandbox_default", fake_sandbox_default)
    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", completed_run)

    assert (
        asyncio.run(
            run_centaur(
                CentaurOptions(),
                instructions="Instructions",
                bashrc="",
                session=session,
            )
        )
        is session.state
    )
    assert lifecycle == ["entered", "human-cli", "run", "exited"]


def test_run_centaur_preserves_unset_user_and_ambient_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(AgentState(messages=[]))
    session.user = None
    captured: dict[str, object] = {}

    def unexpected_sandbox_default(name: str) -> Iterator[None]:
        raise AssertionError(f"unexpected sandbox selection: {name}")
        yield

    def fake_human_cli(
        *,
        answer: bool | str,
        intermediate_scoring: bool,
        record_session: bool,
        instructions: str,
        bashrc: str,
        user: str | None,
        on_ready: Callable[[], AbstractAsyncContextManager[None]] | None,
    ) -> str:
        captured.update(
            answer=answer,
            intermediate_scoring=intermediate_scoring,
            record_session=record_session,
            instructions=instructions,
            bashrc=bashrc,
            user=user,
            on_ready=on_ready,
        )
        return "human-cli-agent"

    async def completed_run(agent: object, state: AgentState) -> AgentState:
        assert agent == "human-cli-agent"
        assert state is session.state
        return state

    monkeypatch.setattr(centaur_mod, "sandbox_default", unexpected_sandbox_default)
    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", completed_run)

    assert (
        asyncio.run(
            run_centaur(
                CentaurOptions(),
                instructions="Instructions",
                bashrc="alias agy='/opt/agent-cli/agy'",
                session=session,
            )
        )
        is session.state
    )
    assert captured == {
        "answer": True,
        "intermediate_scoring": False,
        "record_session": True,
        "instructions": "Instructions",
        "bashrc": "alias agy='/opt/agent-cli/agy'",
        "user": None,
        "on_ready": None,
    }


def test_run_centaur_tears_down_ready_session_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(AgentState(messages=[]))
    lifecycle: list[str] = []
    human_cli_service_started = False
    on_ready_from_human_cli: Callable[[], AbstractAsyncContextManager[None]] | None = (
        None
    )

    @asynccontextmanager
    async def on_ready(ready: CentaurSession) -> AsyncIterator[None]:
        assert human_cli_service_started
        assert ready is session
        lifecycle.append("entered")
        try:
            yield
        finally:
            lifecycle.append("exited")

    def fake_human_cli(
        *,
        on_ready: Callable[[], AbstractAsyncContextManager[None]] | None = None,
        **kwargs: object,
    ) -> str:
        nonlocal on_ready_from_human_cli
        del kwargs
        on_ready_from_human_cli = on_ready
        return "human-cli-agent"

    async def cancelled_run(agent: object, state: AgentState) -> AgentState:
        nonlocal human_cli_service_started
        assert agent == "human-cli-agent"
        assert state is session.state
        assert lifecycle == []
        human_cli_service_started = True
        assert on_ready_from_human_cli is not None
        async with on_ready_from_human_cli():
            assert lifecycle == ["entered"]
            raise asyncio.CancelledError()

    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", cancelled_run)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_centaur(
                CentaurOptions(on_ready=on_ready),
                instructions="Instructions",
                bashrc="",
                session=session,
            )
        )
    assert lifecycle == ["entered", "exited"]


def test_claude_centaur_shell_continues_the_wrapper_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(AgentState(messages=[]))
    captured: dict[str, str] = {}
    fake_claude = tmp_path / "fake-claude"
    fake_claude.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    fake_claude.chmod(0o755)

    async def capture_run_centaur(
        options: CentaurOptions,
        instructions: str,
        bashrc: str,
        ready_session: CentaurSession,
        commands_filter: CommandsFilter | None = None,
    ) -> AgentState:
        del options, instructions, ready_session, commands_filter
        captured["bashrc"] = bashrc
        return session.state

    monkeypatch.setattr(claude_code_mod, "run_centaur", capture_run_centaur)
    asyncio.run(
        claude_code_mod.run_claude_code_centaur(
            CentaurOptions(),
            [
                str(fake_claude),
                "--session-id",
                "native-session",
                "--permission-mode",
                "bypassPermissions",
            ],
            {},
            session,
            LiveConsumer(),
        )
    )
    bashrc_path = tmp_path / "centaur.bashrc"
    bashrc_path.write_text(captured["bashrc"], encoding="utf-8")
    environment = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}

    def run_claude(command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", f". {shlex.quote(str(bashrc_path))}; claude {command}"],
            check=False,
            capture_output=True,
            cwd=tmp_path,
            env=environment,
            text=True,
        )

    expected = "--resume\nnative-session\n--permission-mode\nbypassPermissions\n--\nnext task\n"
    resumed = run_claude("--resume native-session -- 'next task'")
    continued = run_claude("--continue -- 'next task'")
    foreign = run_claude("--resume foreign-session")

    assert resumed.returncode == 0
    assert resumed.stdout == expected
    assert continued.returncode == 0
    assert continued.stdout == expected
    assert foreign.returncode == 2
    assert "wrapper-owned session" in foreign.stderr


def test_run_centaur_preserves_cancellation_when_recorder_finalization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(AgentState(messages=[]))
    calls: list[str] = []

    async def finalize() -> None:
        calls.append("finalize")
        raise RuntimeError("unresolved native recorder spans")

    session.finalize = finalize

    def fake_human_cli(**kwargs: object) -> str:
        del kwargs
        return "human-cli-agent"

    async def cancelled_run(agent: object, state: AgentState) -> AgentState:
        assert agent == "human-cli-agent"
        assert state is session.state
        raise asyncio.CancelledError()

    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", cancelled_run)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_centaur(
                CentaurOptions(),
                instructions="Instructions",
                bashrc="",
                session=session,
            )
        )

    assert calls == ["finalize"]


def test_run_centaur_rejects_unresolved_recorder_spans_on_clean_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(AgentState(messages=[]))

    async def finalize() -> None:
        raise RuntimeError("unresolved native recorder spans")

    session.finalize = finalize

    def fake_human_cli(**kwargs: object) -> str:
        del kwargs
        return "human-cli-agent"

    async def completed_run(agent: object, state: AgentState) -> AgentState:
        assert agent == "human-cli-agent"
        return state

    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", completed_run)

    with pytest.raises(RuntimeError, match="unresolved native recorder spans"):
        asyncio.run(
            run_centaur(
                CentaurOptions(),
                instructions="Instructions",
                bashrc="",
                session=session,
            )
        )


def test_run_centaur_finalizes_recorder_instead_of_plain_teardown_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(AgentState(messages=[]))
    calls: list[str] = []

    async def refresh(command: str) -> None:
        calls.append(command)

    async def finalize() -> None:
        calls.append("finalize")

    session.refresh = refresh
    session.finalize = finalize

    def fake_human_cli(**kwargs: object) -> str:
        del kwargs
        return "human-cli-agent"

    async def cancelled_run(agent: object, state: AgentState) -> AgentState:
        assert agent == "human-cli-agent"
        assert state is session.state
        raise asyncio.CancelledError()

    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", cancelled_run)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_centaur(
                CentaurOptions(),
                instructions="Instructions",
                bashrc="",
                session=session,
            )
        )

    assert calls == ["finalize"]


class _ScoreRebuttalCommand(HumanAgentCommand):
    """Score command with a custom service shape and serializable CLI body."""

    def __init__(self, calls: list[tuple[str | None, str, float]]) -> None:
        self._calls = calls

    @property
    def name(self) -> str:
        return "score"

    @property
    def description(self) -> str:
        return "Score a rebuttal."

    @property
    def cli_args(self) -> list[HumanAgentCommand.CLIArg]:
        return [
            HumanAgentCommand.CLIArg(
                name="answer",
                description="Answer to score.",
            ),
        ]

    def cli(self, args: Namespace) -> None:
        print(call_human_agent("score", **vars(args)))

    def service(self, state: HumanAgentState) -> Callable[..., Awaitable[JsonValue]]:
        del state

        async def score_task(
            answer: str | None, rebuttal: str, *, confidence: float
        ) -> JsonValue:
            self._calls.append((answer, rebuttal, confidence))
            return "scored"

        return score_task


def test_positional_centaur_session_preserves_lifecycle_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = AgentState(messages=[])
    events: list[str] = []
    score_calls: list[tuple[str | None, str, float]] = []

    async def refresh(command: str) -> None:
        events.append(f"refresh:{command}")

    async def finalize() -> None:
        events.append("finalize")

    session = CentaurSession(
        state,
        ("/opt/agent-cli/agy",),
        {},
        "/workdir",
        "agent",
        MagicMock(spec=SandboxEnvironment),
        12345,
        "native-session",
        refresh,
        finalize,
    )
    assert session.sandbox_name is None
    assert session.refresh is refresh
    assert session.finalize is finalize

    ready_callback: Callable[[], AbstractAsyncContextManager[None]] | None = None
    command_filter: CommandsFilter | None = None

    @asynccontextmanager
    async def on_ready(ready: CentaurSession) -> AsyncIterator[None]:
        assert ready is session
        events.append("ready-entered")
        yield
        events.append("ready-exited")

    def fake_human_cli(
        *,
        answer: bool | str,
        intermediate_scoring: bool,
        record_session: bool,
        instructions: str,
        bashrc: str,
        user: str | None,
        commands_filter: CommandsFilter,
        on_ready: Callable[[], AbstractAsyncContextManager[None]] | None,
    ) -> str:
        del answer, intermediate_scoring, record_session, instructions, bashrc, user
        nonlocal command_filter, ready_callback
        command_filter = commands_filter
        ready_callback = on_ready
        return "human-cli-agent"

    async def completed_run(agent: object, received_state: AgentState) -> AgentState:
        assert agent == "human-cli-agent"
        assert received_state is state
        assert ready_callback is not None
        assert command_filter is not None
        async with ready_callback():
            command = command_filter([_ScoreRebuttalCommand(score_calls)])[0]
            handler = command.service(MagicMock(spec=HumanAgentState))
            assert await handler("answer", "rebuttal", confidence=0.8) == "scored"
        return state

    monkeypatch.setattr(centaur_mod, "human_cli", fake_human_cli)
    monkeypatch.setattr(centaur_mod, "run", completed_run)

    assert (
        asyncio.run(
            run_centaur(
                CentaurOptions(on_ready=on_ready),
                instructions="Instructions",
                bashrc="alias agy='/opt/agent-cli/agy'",
                session=session,
                commands_filter=_commands_filter,
            )
        )
        is state
    )
    assert score_calls == [("answer", "rebuttal", 0.8)]
    assert events == ["ready-entered", "refresh:score", "ready-exited", "finalize"]


def test_refreshing_command_preserves_custom_service_and_cli_serialization(
    tmp_path: Path,
) -> None:
    session = _session(AgentState(messages=[]))
    calls: list[tuple[str | None, str, float]] = []

    async def refresh(command: str) -> None:
        assert command == "score"

    session.refresh = refresh
    filtered = centaur_mod._commands_filter_with_refresh(None, session)(
        [_ScoreRebuttalCommand(calls)]
    )
    serialized_cli = human_agent_commands(filtered)
    task_py = tmp_path / "task.py"
    task_py.write_text(serialized_cli, encoding="utf-8")
    (tmp_path / "human_agent.py").write_text(
        "def call_human_agent(method, **params):\n"
        "    return f\"{method}:{params['answer']}\"\n",
        encoding="utf-8",
    )
    executed = subprocess.run(
        [sys.executable, str(task_py), "score", "answer"],
        check=False,
        capture_output=True,
        text=True,
    )
    handler = filtered[0].service(MagicMock(spec=HumanAgentState))

    async def run_handler() -> JsonValue:
        return await handler("answer", "rebuttal", confidence=0.8)

    assert executed.returncode == 0
    assert executed.stdout == "score:answer\n"
    assert executed.stderr == ""
    assert "def score(args: Namespace)" in serialized_cli
    assert 'call_human_agent("score", **vars(args))' in serialized_cli
    assert "self._command" not in serialized_cli
    assert asyncio.run(run_handler()) == "scored"
    assert calls == [("answer", "rebuttal", 0.8)]


def test_antigravity_centaur_shell_starts_in_the_session_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    antigravity_cli_mod = importlib.import_module(
        "inspect_swe._antigravity_cli.antigravity_cli"
    )
    session = _session(AgentState(messages=[]))
    session.cwd = str(tmp_path)
    captured: dict[str, str] = {}

    async def capture_run_centaur(
        options: CentaurOptions,
        instructions: str,
        bashrc: str,
        ready_session: CentaurSession,
        commands_filter: CommandsFilter | None = None,
    ) -> AgentState:
        del options, instructions, commands_filter
        assert ready_session is session
        captured["bashrc"] = bashrc
        return session.state

    monkeypatch.setattr(antigravity_cli_mod, "run_centaur", capture_run_centaur)
    assert (
        asyncio.run(
            antigravity_cli_mod._run_antigravity_cli_centaur(
                CentaurOptions(),
                ["/opt/agent-cli/agy", "--model", "gemini-3.6-flash"],
                {"HOME": "/home/agent", "GOOGLE_GEMINI_BASE_URL": "http://bridge"},
                session,
            )
        )
        is session.state
    )

    bashrc_path = tmp_path / "centaur.bashrc"
    bashrc_path.write_text(captured["bashrc"], encoding="utf-8")
    shell = subprocess.run(
        ["bash", "-c", f". {shlex.quote(str(bashrc_path))}; pwd"],
        check=False,
        capture_output=True,
        cwd="/",
        env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
        text=True,
    )

    assert shell.returncode == 0
    assert shell.stdout == f"{tmp_path}\n"
