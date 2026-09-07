import sys
from argparse import Namespace
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from logging import getLogger
from typing import Literal

from inspect_ai.agent import AgentState, human_cli, run
from inspect_ai.agent._human.commands.command import HumanAgentCommand
from inspect_ai.agent._human.state import HumanAgentState
from inspect_ai.util import SandboxEnvironment
from pydantic import BaseModel, Field, JsonValue

logger = getLogger(__name__)

CommandsFilter = Callable[[list[HumanAgentCommand]], list[HumanAgentCommand]]

CentaurRefresh = Callable[[str], Awaitable[None]]


CentaurFinalize = Callable[[], Awaitable[None]]


@dataclass
class CentaurSession:
    """Live wrapper session available while the human CLI is running.

    ``state`` is the state written by the entered wrapper bridge. It remains that
    object throughout the ready callback so task commands always read native CLI
    traffic rather than the copied state that :func:`inspect_ai.agent.run` returns.
    ``refresh`` is an optional wrapper-owned native-event drain called immediately
    before human CLI scoring or submission. ``finalize`` drains and verifies the
    recorder at teardown when the wrapper needs stronger completion semantics.
    """

    state: AgentState
    invocation: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: str
    user: str | None
    sandbox: SandboxEnvironment
    bridge_port: int
    session_id: str | None
    refresh: CentaurRefresh | None = None
    finalize: CentaurFinalize | None = None


CentaurReady = Callable[[CentaurSession], AbstractAsyncContextManager[None]]


class CentaurOptions(BaseModel):
    """Options for centaur mode."""

    answer: bool | str = Field(default=True)
    """
    Is an explicit answer required for this task or is it scored
    based on files in the container? Pass a `str` with a regex to validate
    that the answer matches the expected format.
    """

    intermediate_scoring: bool = Field(default=False)
    """Allow the human agent to check their score while working."""

    record_session: bool = Field(default=True)
    """Record all user commands and outputs in the sandbox bash session."""

    on_ready: CentaurReady | None = Field(default=None, exclude=True)
    """Optional lifecycle hook entered after the wrapper bridge is ready."""


async def run_centaur(
    options: CentaurOptions,
    instructions: str,
    bashrc: str,
    session: CentaurSession,
    commands_filter: CommandsFilter | None = None,
) -> AgentState:
    """Run one human CLI session and preserve its output on the live bridge state."""
    ready_callback = options.on_ready
    on_ready: Callable[[], AbstractAsyncContextManager[None]] | None = None
    if ready_callback is not None:

        def on_ready() -> AbstractAsyncContextManager[None]:
            return ready_callback(session)

    if session.refresh is not None:
        commands_filter = _commands_filter_with_refresh(commands_filter, session)

    if commands_filter is not None:
        agent = human_cli(
            answer=options.answer,
            intermediate_scoring=options.intermediate_scoring,
            record_session=options.record_session,
            instructions=instructions,
            bashrc=bashrc,
            user=session.user,
            commands_filter=commands_filter,
            on_ready=on_ready,
        )
    else:
        agent = human_cli(
            answer=options.answer,
            intermediate_scoring=options.intermediate_scoring,
            record_session=options.record_session,
            instructions=instructions,
            bashrc=bashrc,
            user=session.user,
            on_ready=on_ready,
        )

    try:
        completed_state = await run(agent, session.state)
        # human_cli returns a copy whose only current mutation is its answer. The
        # bridge state has the native model conversation, so retain its messages.
        session.state.output = completed_state.output
    except BaseException:
        try:
            await _finalize_centaur_session(session)
        except BaseException as error:
            logger.warning(
                "Centaur recorder finalization failed while preserving the "
                "original session exception",
                exc_info=error,
            )
        raise
    else:
        await _finalize_centaur_session(session)
        return session.state


async def _finalize_centaur_session(session: CentaurSession) -> None:
    """Drain native event recording, enforcing integrity after a clean exit."""
    if session.finalize is not None:
        await session.finalize()
    elif session.refresh is not None:
        await session.refresh("teardown")


def reset_recorder_preserving_session_exception(reset: Callable[[], None]) -> None:
    """Reset a native recorder without replacing an active session exception.

    A clean Centaur exit must reject unresolved native identities. During a
    cancellation or another active session failure, cleanup still runs but its
    strict recorder error is logged so the caller receives the original failure.
    """
    active_error = sys.exc_info()[1]
    try:
        reset()
    except BaseException as error:
        if active_error is None:
            raise
        logger.warning(
            "Centaur recorder reset failed while preserving the original session "
            "exception",
            exc_info=error,
        )


def _commands_filter_with_refresh(
    commands_filter: CommandsFilter | None, session: CentaurSession
) -> CommandsFilter:
    """Compose a recorder drain into terminal command service bodies."""

    def filtered(commands: list[HumanAgentCommand]) -> list[HumanAgentCommand]:
        selected = (
            commands_filter(commands) if commands_filter is not None else commands
        )
        return [
            _RefreshingCommand(command, session)
            if command.name in {"score", "submit", "quit"}
            else command
            for command in selected
        ]

    return filtered


class _RefreshingCommand(HumanAgentCommand):
    """Delegate one terminal command after draining its native CLI recorder."""

    def __init__(self, command: HumanAgentCommand, session: CentaurSession) -> None:
        self._command = command
        self._session = session

    @property
    def name(self) -> str:
        return self._command.name

    @property
    def description(self) -> str:
        return self._command.description

    @property
    def group(self) -> Literal[1, 2, 3]:
        return self._command.group

    @property
    def contexts(self) -> list[Literal["cli", "service"]]:
        return self._command.contexts

    @property
    def cli_args(self) -> list[HumanAgentCommand.CLIArg]:
        return self._command.cli_args

    @property
    def cli(self) -> Callable[[Namespace], None]:
        """Expose the wrapped command's source for human-agent CLI generation."""
        return self._command.cli

    def service(self, state: HumanAgentState) -> Callable[..., Awaitable[JsonValue]]:
        handler = self._command.service(state)

        async def refreshed(*args: object, **kwargs: object) -> JsonValue:
            return await self._call(handler, *args, **kwargs)

        return refreshed

    async def _call(
        self,
        handler: Callable[..., Awaitable[JsonValue]],
        *args: object,
        **kwargs: object,
    ) -> JsonValue:
        refresh = self._session.refresh
        if refresh is None:
            raise RuntimeError(
                "Centaur recorder refresh was cleared before the terminal command ran."
            )
        await refresh(self.name)
        return await handler(*args, **kwargs)
