"""Pass-through of the model bridge's ``poll_timeout_recovery`` setting.

Every agent opens ``sandbox_agent_bridge`` itself, so the agent is the only
place a caller can choose how long the bridge keeps re-polling its proxy
server after a poll of it times out. inspect-ai releases whose
``sandbox_agent_bridge`` has no ``poll_timeout_recovery`` parameter remain
supported: leaving the setting at ``None`` omits the argument, so the bridge
call is exactly what it was before the setting existed, and setting it
against such a bridge raises rather than silently running without recovery.
"""

import inspect
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from inspect_ai.agent import sandbox_agent_bridge

POLL_TIMEOUT_RECOVERY_PARAM = "poll_timeout_recovery"


def poll_timeout_recovery_bridge_args(
    poll_timeout_recovery: float | None,
) -> dict[str, Any]:
    """Keyword arguments that carry ``poll_timeout_recovery`` to the bridge.

    Call this when the agent is constructed, so an unsupported setting fails
    before any sample starts, and spread the result into the agent's
    ``sandbox_agent_bridge(...)`` call.

    Args:
        poll_timeout_recovery: The agent's ``poll_timeout_recovery`` argument.

    Returns:
        An empty mapping for ``None``; otherwise ``poll_timeout_recovery``
        alone.

    Raises:
        RuntimeError: ``poll_timeout_recovery`` is set and the installed
            inspect-ai's ``sandbox_agent_bridge`` does not accept it.
    """
    if poll_timeout_recovery is None:
        return {}
    if (
        POLL_TIMEOUT_RECOVERY_PARAM
        not in inspect.signature(sandbox_agent_bridge).parameters
    ):
        raise RuntimeError(
            f"poll_timeout_recovery={poll_timeout_recovery!r} needs an inspect-ai "
            "whose sandbox_agent_bridge() accepts poll_timeout_recovery, and the "
            f"installed inspect-ai ({_inspect_ai_version()}) does not. Install an "
            "inspect-ai with poll-timeout recovery support, or leave "
            "poll_timeout_recovery unset."
        )
    return {POLL_TIMEOUT_RECOVERY_PARAM: poll_timeout_recovery}


def _inspect_ai_version() -> str:
    try:
        return version("inspect_ai")
    except PackageNotFoundError:
        return "version unknown"
