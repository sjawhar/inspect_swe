"""OpenCode request identity carried by the bridge's allowlisted context."""

from collections.abc import Mapping
from dataclasses import dataclass


_OPENCODE_SESSION_ID_HEADER = "x-opencode-session"
_GENERIC_SESSION_ID_HEADER = "x-session-id"
_PARENT_SESSION_ID_HEADER = "x-parent-session-id"


@dataclass(frozen=True)
class OpenCodeRequestIdentity:
    """Native OpenCode session relationship for one bridge model request."""

    session_id: str
    parent_session_id: str | None


def request_identity(
    identity_headers: Mapping[str, str],
) -> OpenCodeRequestIdentity | None:
    """Read OpenCode's selected session headers from public bridge context."""
    session_headers = {
        name: identity_headers[name]
        for name in (_OPENCODE_SESSION_ID_HEADER, _GENERIC_SESSION_ID_HEADER)
        if identity_headers.get(name)
    }
    if len(session_headers) > 1:
        raise RuntimeError(
            "OpenCode bridge request supplied conflicting native session headers."
        )
    session_id = next(iter(session_headers.values()), None)
    if session_id is None:
        return None
    return OpenCodeRequestIdentity(
        session_id=session_id,
        parent_session_id=identity_headers.get(_PARENT_SESSION_ID_HEADER),
    )
