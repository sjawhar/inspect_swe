"""Bridge `ModelEventSink` for Codex CLI sub-agent spans.

The sink opens spans from a parent's native `spawn_agent` tool call and binds
the returned Codex thread ID to that exact tool-call ID. `close_agent`,
completion notifications, and `reset()` close those spans through the same
native IDs.

The consumer has no verified child thread ID in a bridge `ModelEvent`. It
therefore does not reconstruct an identity by matching the child prompt. While
child spans are open, unidentifiable model events remain unscoped instead of
being silently attributed to the parent.

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
    completed_thread_ids,
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
    thread_id: str | None = None


class CodexConsumer(ModelEventSink):
    def __init__(self) -> None:
        self._agents: dict[str, _OpenAgent] = {}
        self._thread_index: dict[str, str] = {}
        self._nicknames: dict[str, str] = {}
        self._emitted_events: set[int] = set()

    @property
    def outer_span_id(self) -> str | None:
        """Span for main-agent attribution, resolved at emission time."""
        return current_span_id()

    def reset(self) -> None:
        """Close any open spans and clear per-attempt state."""
        for call_id in reversed(list(self._agents.keys())):
            agent = self._agents.pop(call_id)
            transcript()._event(SpanEndEvent(id=agent.span_id))
        self._thread_index.clear()
        self._nicknames.clear()
        self._emitted_events.clear()

    def on_pending(self, event: ModelEvent) -> None:
        self._harvest_bindings(event.input)
        for thread_id in completed_thread_ids(event.input):
            self._close_thread(thread_id)

        span_id = self._attribute()
        event.span_id = span_id

        if is_compaction_request(event.input):
            transcript()._event(
                CompactionEvent(
                    source="codex_cli",
                    span_id=span_id,
                    metadata={"trigger": "auto"},
                )
            )

        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        msg = event.output.message if event.output else None
        if msg is not None and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.view is None:
                    custom = tool_view(tc.function, tc.arguments or {}, self._nicknames)
                    if custom is not None:
                        tc.view = custom

            parent_span_id = event.span_id or self.outer_span_id
            for spawned in find_spawned_agents(msg.tool_calls):
                if spawned.call_id in self._agents:
                    continue
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

            for target in find_close_targets(msg.tool_calls):
                self._close_thread(target)

        if id(event) in self._emitted_events:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)

    def _harvest_bindings(self, input_messages: list[ChatMessage]) -> None:
        """Bind thread_id → span from spawn_agent tool results by tool-call ID."""
        for msg in input_messages:
            if not isinstance(msg, ChatMessageTool):
                continue
            result = spawn_result(msg)
            if result is None or msg.tool_call_id is None:
                continue
            if result.nickname is not None:
                self._nicknames[result.agent_id] = result.nickname
            agent = self._agents.get(msg.tool_call_id)
            if agent is not None and agent.thread_id is None:
                agent.thread_id = result.agent_id
                self._thread_index[result.agent_id] = msg.tool_call_id

    def _close_thread(self, thread_id: str) -> None:
        call_id = self._thread_index.pop(thread_id, None)
        if call_id is None:
            return
        agent = self._agents.pop(call_id, None)
        if agent is None:
            return
        transcript()._event(SpanEndEvent(id=agent.span_id))

    def _attribute(self) -> str | None:
        """Resolve a bridge call without reconstructing child identity."""
        if self._agents:
            return None
        return self.outer_span_id
