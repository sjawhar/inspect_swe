"""Native OpenCode session recorder for bridged model events.

OpenCode sends its session UUID and parent-session UUID as request headers. The
wrapper opts those two headers into the bridge's public request-context metadata,
and the caller supplies the corresponding identity resolver here. This consumer
never reads model text or model identity to assign an agent.

OpenCode's ``task`` tool reports its child UUID in a structured result envelope.
Those results bind a real parent tool call to an already-open child session span
and close that span on terminal task states. The background completion envelope
is a native synthetic user message rather than a tool result; it is accepted
only when its UUID already names an active child of the request's real session.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._model import ModelEventSink
from inspect_ai.util._span import current_span_id

from .identity import OpenCodeRequestIdentity

IdentityResolver = Callable[[ModelEvent], OpenCodeRequestIdentity | None]


@dataclass
class _OpenCodeSession:
    """Known native session and its associated Inspect span, if it is a child."""

    session_id: str
    parent_session_id: str | None
    span_id: str | None
    last_span_id: str | None
    begin_event: SpanBeginEvent | None
    closed: bool = False


@dataclass(frozen=True)
class _TaskResult:
    """Native OpenCode task completion result received by its parent session."""

    session_id: str
    state: Literal["running", "completed", "error"]


_TASK_RESULT_START = re.compile(
    r'\A<task id="(?P<session_id>[^"]+)" state="(?P<state>running|completed|error)">'
)


class OpenCodeConsumer(ModelEventSink):
    """Route bridge events to their native OpenCode session spans."""

    def __init__(self, identity_resolver: IdentityResolver) -> None:
        """Create a recorder using the bridge's selected request identity context."""
        self._identity_resolver = identity_resolver
        self._sessions: dict[str, _OpenCodeSession] = {}
        self._emitted_events: set[int] = set()

    def on_pending(self, event: ModelEvent) -> None:
        """Open or locate the request's native session span before generation."""
        identity = self._identity_resolver(event)
        if identity is None:
            raise RuntimeError(
                "OpenCode bridge request is missing its allowlisted native session ID."
            )

        self._handle_task_results(identity, event.input)
        event.span_id = self._span_for(identity)
        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        """Publish the completed model event after bridge generation finishes."""
        if id(event) not in self._emitted_events:
            return
        self._emitted_events.discard(id(event))
        transcript()._event_updated(event)

    def on_native_event(self, event: Mapping[str, object]) -> None:
        """Record a native ``session.compacted`` event from the OpenCode plugin."""
        if event.get("type") != "session.compacted":
            raise RuntimeError("Unexpected OpenCode native event.")
        properties = event.get("properties")
        if not isinstance(properties, Mapping):
            raise RuntimeError("OpenCode compaction event is missing its properties.")
        session_id = properties.get("sessionID")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("OpenCode compaction event is missing its session ID.")
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError(
                f"OpenCode emitted a compaction event for unknown session {session_id!r}."
            )
        span_id = session.span_id or session.last_span_id
        if span_id is None:
            raise RuntimeError(
                f"OpenCode session {session_id!r} has no Inspect span for compaction."
            )
        transcript()._event(
            CompactionEvent(
                source="opencode",
                span_id=span_id,
                metadata={"opencode_session_id": session_id},
            )
        )

    def reset(self) -> None:
        """Balance unfinished child spans before the wrapper starts another attempt."""
        for session in reversed(list(self._sessions.values())):
            if session.begin_event is None or session.closed:
                continue
            if session.span_id is None:
                raise RuntimeError(
                    "OpenCode child span is missing its native session ID."
                )
            transcript()._event(SpanEndEvent(id=session.span_id))
            session.closed = True
        self._sessions.clear()
        self._emitted_events.clear()

    def _span_for(self, identity: OpenCodeRequestIdentity) -> str | None:
        session = self._sessions.get(identity.session_id)
        if session is not None:
            if session.closed:
                raise RuntimeError(
                    f"OpenCode closed session {identity.session_id!r} emitted another request."
                )
            if session.parent_session_id != identity.parent_session_id:
                raise RuntimeError(
                    "OpenCode session changed parent from "
                    f"{session.parent_session_id!r} to {identity.parent_session_id!r}."
                )
            span_id = session.span_id or current_span_id()
            session.last_span_id = span_id
            return span_id

        if identity.parent_session_id is None:
            span_id = current_span_id()
            self._sessions[identity.session_id] = _OpenCodeSession(
                session_id=identity.session_id,
                parent_session_id=None,
                span_id=None,
                last_span_id=span_id,
                begin_event=None,
            )
            return span_id

        parent = self._sessions.get(identity.parent_session_id)
        if parent is None:
            raise RuntimeError(
                "OpenCode child session "
                f"{identity.session_id!r} arrived before parent session "
                f"{identity.parent_session_id!r}."
            )
        if parent.last_span_id is None:
            raise RuntimeError(
                "OpenCode parent session "
                f"{identity.parent_session_id!r} has no active Inspect span."
            )

        span_id = f"opencode-session-{identity.session_id}"
        begin_event = SpanBeginEvent(
            id=span_id,
            parent_id=parent.last_span_id,
            type="agent",
            name="opencode",
            metadata={
                "opencode_session_id": identity.session_id,
                "opencode_parent_session_id": identity.parent_session_id,
            },
        )
        transcript()._event(begin_event)
        self._sessions[identity.session_id] = _OpenCodeSession(
            session_id=identity.session_id,
            parent_session_id=identity.parent_session_id,
            span_id=span_id,
            last_span_id=span_id,
            begin_event=begin_event,
        )
        return span_id

    def _handle_task_results(
        self,
        parent: OpenCodeRequestIdentity,
        messages: list[ChatMessage],
    ) -> None:
        """Bind and close child spans from OpenCode's structured task results."""
        for message in messages:
            result = _task_result(message)
            if result is None:
                continue
            child = self._sessions.get(result.session_id)
            if child is None or child.parent_session_id != parent.session_id:
                continue

            if (
                isinstance(message, ChatMessageTool)
                and message.tool_call_id is not None
            ):
                self._bind_task_call(child, message.tool_call_id)
            if result.state in ("completed", "error"):
                self._close_child(result.session_id)

    def _bind_task_call(self, child: _OpenCodeSession, tool_call_id: str) -> None:
        """Attach the parent OpenCode task call ID once its native result arrives."""
        begin_event = child.begin_event
        if begin_event is None:
            return
        metadata = begin_event.metadata
        if metadata is None:
            raise RuntimeError("OpenCode child span is missing its identity metadata.")
        existing = metadata.get("opencode_task_call_id")
        if existing is not None and existing != tool_call_id:
            raise RuntimeError(
                "OpenCode child session "
                f"{child.session_id!r} was linked to conflicting task calls."
            )
        if existing == tool_call_id:
            return
        metadata["opencode_task_call_id"] = tool_call_id
        transcript()._event_updated(begin_event)

    def _close_child(self, session_id: str) -> None:
        """End exactly one known child span after its native task result settles."""
        child = self._sessions.get(session_id)
        if child is None or child.begin_event is None or child.closed:
            return
        if child.span_id is None:
            raise RuntimeError("OpenCode child span is missing its native session ID.")
        transcript()._event(SpanEndEvent(id=child.span_id))
        child.closed = True


def _task_result(message: ChatMessage) -> _TaskResult | None:
    """Parse OpenCode's ``TaskTool.renderOutput`` result envelope only.

    Arbitrary user text is not an identity source. A background completion is
    accepted only for an already-known child session by ``_handle_task_results``.
    """
    if isinstance(message, ChatMessageTool):
        if message.function != "task":
            return None
    elif not isinstance(message, ChatMessageUser):
        return None

    match = _TASK_RESULT_START.match(message.text)
    if match is None:
        return None
    state = match.group("state")
    if state == "running":
        task_state: Literal["running", "completed", "error"] = "running"
    elif state == "completed":
        task_state = "completed"
    elif state == "error":
        task_state = "error"
    else:
        return None
    return _TaskResult(
        session_id=match.group("session_id"),
        state=task_state,
    )
