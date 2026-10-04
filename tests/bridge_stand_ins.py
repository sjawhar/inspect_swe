"""Stand-ins for ``sandbox_agent_bridge`` in fast agent tests.

No sandbox, no Docker, no API keys: an agent is run only until it opens its
model bridge, and the installed inspect-ai's bridge signature can be swapped
for one with or without ``poll_timeout_recovery``.
"""

from contextlib import asynccontextmanager, contextmanager
from types import ModuleType
from typing import AsyncIterator, Iterator
from unittest.mock import patch

import anyio
import pytest
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessageUser
from inspect_ai.util import Store
from inspect_swe._util import poll_timeout_recovery


class BridgeEntered(Exception):
    """Raised by the stand-in bridge of `bridge_call_kwargs` on entry."""


def bridge_call_kwargs(agent_module: ModuleType, agent: Agent) -> dict[str, object]:
    """Run `agent` until it opens its model bridge; return that call's kwargs.

    `agent_module` is the module whose `sandbox_agent_bridge`, `checkpointer`
    and `store` the agent calls. The stand-in bridge records the keyword
    arguments and raises `BridgeEntered` on entry, so the agent never reaches a
    sandbox.
    """
    calls: list[dict[str, object]] = []

    @asynccontextmanager
    async def bridge(state: AgentState, **kwargs: object) -> AsyncIterator[None]:
        calls.append(kwargs)
        raise BridgeEntered()
        yield

    @asynccontextmanager
    async def no_checkpointer() -> AsyncIterator[None]:
        yield None

    with (
        patch.object(agent_module, "sandbox_agent_bridge", bridge),
        patch.object(agent_module, "checkpointer", no_checkpointer),
        patch.object(agent_module, "store", return_value=Store()),
        pytest.raises(BridgeEntered),
    ):
        anyio.run(agent, AgentState(messages=[ChatMessageUser(content="Solve it.")]))
    assert len(calls) == 1
    return calls[0]


def _bridge_with_poll_timeout_recovery(
    state: AgentState | None = None,
    *,
    port: int = 13131,
    poll_timeout_recovery: float | None = None,
) -> None:
    raise AssertionError("signature stand-in; never called")


def _bridge_without_poll_timeout_recovery(
    state: AgentState | None = None,
    *,
    port: int = 13131,
) -> None:
    raise AssertionError("signature stand-in; never called")


@contextmanager
def installed_bridge_accepts_poll_timeout_recovery(accepts: bool) -> Iterator[None]:
    """Make the installed `sandbox_agent_bridge` (not) accept the parameter.

    Agents read support for `poll_timeout_recovery` off the installed
    inspect-ai's `sandbox_agent_bridge` signature; this swaps in a signature
    with or without the parameter, independent of the installed version.
    """
    stand_in = (
        _bridge_with_poll_timeout_recovery
        if accepts
        else _bridge_without_poll_timeout_recovery
    )
    with patch.object(poll_timeout_recovery, "sandbox_agent_bridge", stand_in):
        yield
