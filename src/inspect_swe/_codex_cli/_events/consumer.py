"""Bridge `ModelEventSink` for Codex CLI sub-agent spans.

The sink opens spans from a parent's native `spawn_agent` tool call and binds
the result's exact native routing key to that tool-call ID. V1 returns an
`agent_id`; v2 returns a slash-prefixed `task_name`, which the bridge preserves
as the raw `agent_message` recipient on the child `ModelEvent`. A child event
whose native recipient arrives before its root receives the matching result is
held until that exact result binds it to one open span.

When a root request carries the bridge's exact native Codex thread identity,
the consumer keeps it in the outer span even while a child is live. Root
identity takes precedence over recipient routing, and ambiguous or unresolved
child bootstrap records fail rather than guessing from history or being
silently emitted unscoped.

Codex's bridge-local compaction marker is emitted as a `CompactionEvent`.
"""

from dataclasses import dataclass

from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import ChatMessage, ChatMessageTool
from inspect_ai.model._model import ModelEventSink
from inspect_ai.util._span import current_span_id

from .detection import (
    agent_message_recipients,
    completed_agent_keys,
    find_close_targets,
    find_spawned_agents,
    is_compaction_request,
    spawn_result,
)
from .toolview import tool_view


@dataclass
class _OpenAgent:
    """A sub-agent span currently open (spawn_agent seen, not yet closed)."""

    call_id: str
    span_id: str
    native_key: str | None = None


@dataclass
class _PendingChildEvent:
    """A child event awaiting an exact native spawn-result binding."""

    event: ModelEvent
    recipients: set[str]


class CodexConsumer(ModelEventSink):
    def __init__(self) -> None:
        # spawn tool_call_id → open sub-agent span. Insertion order = open order
        # (used to close innermost-first in reset()).
        self._agents: dict[str, _OpenAgent] = {}
        # native routing key → all spawn tool-call IDs that returned it. A
        # duplicate key is ambiguous and must never select a child span.
        self._agent_key_index: dict[str, set[str]] = {}

        # Pinned Codex 0.153.1 `spawn_agent_internal()` sends the initial
        # inter-agent communication before it returns the `LiveAgent` whose
        # result supplies this recipient, so this ordering is native.
        # Preserve the event until that result identifies one open child span.
        self._pending_child_events: list[_PendingChildEvent] = []

        # A held event may complete before the parent receives the result that
        # gives it a span. Delay its completion processing so any descendants
        # inherit the correct child span after the exact binding arrives.
        self._completed_pending_events: set[int] = set()

        # native routing key → nickname (Codex's friendly per-agent name).
        self._nicknames: dict[str, str] = {}

        # Root Codex Responses thread. The bridge projects it from native
        # client_metadata, so a root continuation remains distinguishable while
        # a child span is live.
        self._root_thread_id: str | None = None

        # ModelEvents we've _event()'d, so on_complete knows to _event_updated.
        self._emitted_events: set[int] = set()

    @property
    def outer_span_id(self) -> str | None:
        """Span for main-agent attribution, resolved at emission time.

        Must not be captured once at construction: with checkpointing
        active, the enclosing checkpoint span rotates at each fire and a
        frozen id would pin every event to the first checkpoint.
        """
        return current_span_id()

    def reset(self) -> None:
        """Close any open spans and clear per-attempt state.

        Called between Codex attempts and after the attempt loop, so the span
        tree stays balanced even if Codex exited before a child binds.
        """
        self._flush_pending_child_events()
        unresolved_recipients = sorted(
            {
                recipient
                for pending in self._pending_child_events
                for recipient in pending.recipients
            }
        )

        for call_id in reversed(list(self._agents.keys())):
            agent = self._agents.pop(call_id)
            transcript()._event(SpanEndEvent(id=agent.span_id))
        self._agent_key_index.clear()
        self._pending_child_events.clear()
        self._nicknames.clear()
        self._emitted_events.clear()
        self._completed_pending_events.clear()
        self._root_thread_id = None

        if unresolved_recipients:
            raise RuntimeError(
                "Codex child events were not bound to an exact native "
                f"spawn result: {', '.join(unresolved_recipients)}."
            )

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_pending(self, event: ModelEvent) -> None:
        # A root's native thread ID wins over child routing even while a child
        # is open. Bind any earlier child bootstrap records first because a
        # root receives the spawn result after the child can start.
        self._harvest_bindings(event.input)
        self._flush_pending_child_events()
        completed_keys = completed_agent_keys(event.input)
        for native_key in completed_keys:
            self._close_agent_key(native_key)

        root_thread_id = _root_thread_id(event.metadata)
        if root_thread_id is not None:
            if self._root_thread_id is None:
                self._root_thread_id = root_thread_id
            if root_thread_id == self._root_thread_id:
                event.span_id = self.outer_span_id
                self._emit_event(event)
                return

        recipients = agent_message_recipients(event.input)
        span_id = self._child_span_for_recipients(recipients)
        if span_id is not None:
            event.span_id = span_id
            self._emit_event(event)
            return
        if recipients and self._agents and not completed_keys:
            self._pending_child_events.append(
                _PendingChildEvent(event=event, recipients=recipients)
            )
            self._flush_pending_child_events()
            return

        event.span_id = None if self._agents else self.outer_span_id
        self._emit_event(event)

    def on_complete(self, event: ModelEvent) -> None:
        if any(pending.event is event for pending in self._pending_child_events):
            self._completed_pending_events.add(id(event))
            return
        self._complete_event(event)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _complete_event(self, event: ModelEvent) -> None:
        msg = event.output.message if event.output else None
        if msg is not None and msg.tool_calls:
            # custom rendering for Codex built-in tools (see toolview.py)
            for tc in msg.tool_calls:
                if tc.view is None:
                    custom = tool_view(tc.function, tc.arguments or {}, self._nicknames)
                    if custom is not None:
                        tc.view = custom

            # Open each child before its first call can arrive. A buffered
            # parent reaches here only after `_flush_pending_child_events()`
            # has assigned its exact span.
            parent_span_id = event.span_id or self.outer_span_id
            for spawned in find_spawned_agents(msg.tool_calls):
                if spawned.call_id in self._agents:
                    continue  # idempotent (defensive against retries)
                span_id = f"agent-{spawned.call_id}"
                self._agents[spawned.call_id] = _OpenAgent(
                    call_id=spawned.call_id,
                    span_id=span_id,
                )
                metadata: dict[str, str] = {"agent_type": spawned.agent_type}
                if spawned.reasoning_effort:
                    metadata["reasoning_effort"] = spawned.reasoning_effort
                transcript()._event(
                    SpanBeginEvent(
                        id=span_id,
                        parent_id=parent_span_id,
                        type="agent",
                        name=spawned.agent_type,
                        metadata=metadata,
                    )
                )

            # explicit close_agent calls
            for target in find_close_targets(msg.tool_calls):
                self._close_agent_key(target)

        if id(event) in self._emitted_events:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)

    def _emit_event(self, event: ModelEvent) -> None:
        if is_compaction_request(event.input):
            transcript()._event(
                CompactionEvent(
                    source="codex_cli",
                    span_id=event.span_id,
                    metadata={"trigger": "auto"},
                )
            )
        self._emitted_events.add(id(event))
        transcript()._event(event)

    def _harvest_bindings(self, input_messages: list[ChatMessage]) -> None:
        """Bind a native spawn result key to its exact open tool-call span."""
        for msg in input_messages:
            if not isinstance(msg, ChatMessageTool):
                continue
            result = spawn_result(msg)
            if result is None or msg.tool_call_id is None:
                continue
            if result.nickname is not None:
                self._nicknames[result.native_key] = result.nickname
            agent = self._agents.get(msg.tool_call_id)
            if agent is not None and agent.native_key is None:
                agent.native_key = result.native_key
                self._agent_key_index.setdefault(result.native_key, set()).add(
                    msg.tool_call_id
                )

    def _flush_pending_child_events(self) -> None:
        """Emit held native-recipient events after an exact unique binding."""
        pending_events: list[_PendingChildEvent] = []
        for pending in self._pending_child_events:
            span_id = self._child_span_for_recipients(pending.recipients)
            if span_id is None:
                pending_events.append(pending)
                continue
            pending.event.span_id = span_id
            self._emit_event(pending.event)
            if id(pending.event) in self._completed_pending_events:
                self._completed_pending_events.discard(id(pending.event))
                self._complete_event(pending.event)
        self._pending_child_events = pending_events

    def _child_span_for_recipients(self, recipients: set[str]) -> str | None:
        """Return one open span only when every recipient maps to that span."""
        if not recipients:
            return None
        call_ids: set[str] = set()
        for recipient in recipients:
            candidates = self._agent_key_index.get(recipient)
            if candidates is None or len(candidates) != 1:
                return None
            call_ids.update(candidates)
        if len(call_ids) != 1:
            return None
        agent = self._agents.get(call_ids.pop())
        return agent.span_id if agent is not None else None

    def _close_agent_key(self, native_key: str) -> None:
        call_ids = self._agent_key_index.pop(native_key, set())
        for call_id in call_ids:
            agent = self._agents.pop(call_id, None)
            if agent is not None:
                transcript()._event(SpanEndEvent(id=agent.span_id))


def _root_thread_id(metadata: object) -> str | None:
    """Return the bridge-projected native root thread identity, if present."""
    if not isinstance(metadata, dict):
        return None
    agent_bridge = metadata.get("agent_bridge")
    if not isinstance(agent_bridge, dict):
        return None
    codex = agent_bridge.get("codex")
    if not isinstance(codex, dict):
        return None
    thread_id = codex.get("thread_id")
    if (
        isinstance(thread_id, str)
        and thread_id
        and "parent_thread_id" not in codex
        and "subagent" not in codex
    ):
        return thread_id
    return None
