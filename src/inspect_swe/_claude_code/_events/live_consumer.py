"""Real-time and native-transcript consumer for Claude Code events.

The bridge installs this `ModelEventSink` so it can route each native CLI
model call to the Inspect transcript. A parent Task/Agent tool call opens a
native agent span immediately, before Claude Code can issue the child request.

Claude's durable JSONL records supply the identity bridge events do not:
`tool_result.toolUseResult.agentId` links a completed native agent to its
Task/Agent tool-use ID, and a sidechain assistant `message.id` links an
individual bridged response to that native agent. Centaur buffers child
ModelEvents until those exact native IDs arrive through its explicit
session-drain hook. This retains overlapping same-model siblings without
reconstructing identity from prompt text or timing.

The same transcript drain observes native `compact_boundary` events, so an
interactive `/compact` is recorded even though Centaur does not use Claude's
unattended stream-json stdout channel.
"""

import json
import shlex
from dataclasses import dataclass
from typing import Any

from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import ChatMessage, ChatMessageTool
from inspect_ai.model._model import ModelEventSink
from inspect_ai.model._model_output import StopReason
from inspect_ai.util import SandboxEnvironment
from inspect_ai.util._span import current_span_id

from .toolview import tool_view


@dataclass
class _OpenAgent:
    """Native child span keyed by its originating Task/Agent tool-use ID."""

    span_id: str
    agent_id: str | None = None
    complete: bool = False


class LiveConsumer(ModelEventSink):
    """Sink for bridge model events and one owned Claude JSONL session.

    The bridge invokes `on_pending` / `on_complete` for each `ModelEvent`.
    `refresh` reads the configured native session and direct subagent files at
    command boundaries; unattended stream-json callers may still forward
    individual lines through `process_jsonl_line`.
    """

    def __init__(self) -> None:
        # Task/Agent tool_use ID → an opened native agent span.
        self._open_agents: dict[str, _OpenAgent] = {}

        # Native agent ID → originating Task/Agent tool_use ID. This is only
        # populated from Claude's `toolUseResult.agentId`, never from content.
        self._agent_to_tool: dict[str, str] = {}

        # Native assistant message ID → native agent ID. `None` represents the
        # root Claude session and therefore resolves to `outer_span_id`.
        self._response_agents: dict[str, str | None] = {}

        # Child ModelEvents wait here until their completed response ID appears
        # in the native session transcript. Dict insertion order preserves their
        # bridge order when the drain releases them.
        self._pending_events: dict[int, ModelEvent] = {}

        # The bridge callback runs under the correct wrapper span. Retain that
        # parent for a buffered root response rather than resolving it later
        # from a score/submit service callback.
        self._pending_outer_spans: dict[int, str | None] = {}
        self._centaur_outer_span_id: str | None = None
        self._pending_compactions: list[dict[str, Any]] = []

        # Native event UUIDs prevent a score/submit refresh from re-emitting
        # already-consumed JSONL records.
        self._seen_native_events: set[str] = set()

        # Centaur installs these after its wrapper-owned session ID is known.
        self._sandbox: SandboxEnvironment | None = None
        self._user: str | None = None
        self._session_id: str | None = None

        # Track which event objects we emitted pending so on_complete knows
        # whether to update that event or append a fully-completed one.
        self._emitted_events: set[int] = set()

        # Stop reason of the most recent completed ModelEvent. Used by the
        # unattended runner to distinguish a refusal from a scaffold crash.
        self._last_stop_reason: StopReason | None = None

    def configure_centaur_session(
        self,
        sandbox: SandboxEnvironment,
        user: str | None,
        session_id: str,
    ) -> None:
        """Configure the native session transcript drained by Centaur hooks."""
        self._sandbox = sandbox
        self._user = user
        self._session_id = session_id
        self._centaur_outer_span_id = self.outer_span_id

    async def refresh(self, command: str) -> None:
        """Drain the configured Claude session and subagent transcript files.

        ``command`` identifies the native human-agent lifecycle boundary that
        requested this drain (currently score, submit, or teardown). Identity
        resolution itself only consumes the wrapper-owned native session files.
        """
        if self._sandbox is None or self._session_id is None:
            raise RuntimeError("Claude Centaur session has not been configured.")

        session_file = shlex.quote(f"{self._session_id}.jsonl")
        subagent_file = shlex.quote(
            f"*/{self._session_id}/subagents/agent-*.jsonl"
        )
        result = await self._sandbox.exec(
            [
                "sh",
                "-c",
                'if [ -d "$HOME/.claude/projects" ]; then '
                f'find "$HOME/.claude/projects" -type f \\( -name {session_file} '
                f"-o -path {subagent_file} \\) -print; fi",
            ],
            user=self._user,
        )
        if not result.success:
            raise RuntimeError(
                f"Unable to enumerate Claude session transcript: {result.stderr}"
            )

        for path in result.stdout.splitlines():
            content = await self._sandbox.read_file(path)
            for line in content.splitlines():
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as ex:
                    raise RuntimeError(
                        f"Malformed Claude session JSONL in {path}."
                    ) from ex
                if not isinstance(raw, dict):
                    raise RuntimeError(
                        f"Unexpected non-object Claude session JSONL record in {path}."
                    )
                self.process_jsonl_line(raw)

        self._flush_pending_events()

    @property
    def last_stop_reason(self) -> StopReason | None:
        """Stop reason of the last completed model event this attempt."""
        return self._last_stop_reason

    @property
    def outer_span_id(self) -> str | None:
        """Span for main-agent attribution, resolved at emission time."""
        return current_span_id()

    def reset(self) -> None:
        """Release unresolved events and close every remaining native span."""
        pending = list(self._pending_events.values())
        self._pending_events.clear()
        self._pending_outer_spans.clear()
        for event in pending:
            # Preserve evidence on cancellation even if the process exited
            # before Claude persisted a joinable native response record.
            event.span_id = None
            self._complete_event(event, emitted=False)

        for tool_use_id in reversed(list(self._open_agents.keys())):
            agent = self._open_agents.pop(tool_use_id)
            transcript()._event(SpanEndEvent(id=agent.span_id))

        self._agent_to_tool.clear()
        self._response_agents.clear()
        self._pending_compactions.clear()
        self._seen_native_events.clear()
        self._emitted_events.clear()
        self._last_stop_reason = None
        self._centaur_outer_span_id = None
        self._sandbox = None
        self._user = None
        self._session_id = None

    # ------------------------------------------------------------------
    # ModelEventSink callbacks (called from the bridge)
    # ------------------------------------------------------------------

    def on_pending(self, event: ModelEvent) -> None:
        self._close_bridge_agents(event.input)
        outer_span_id = self.outer_span_id
        if self._open_agents:
            # A native child may share its parent model and prompt. Its
            # completed Claude response ID, not those ambiguous fields, tells
            # the drain which span owns this call.
            self._pending_events[id(event)] = event
            self._pending_outer_spans[id(event)] = outer_span_id
            return

        event.span_id = outer_span_id
        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        # Record the terminal stop reason (guard against empty choices, where
        # ModelOutput.stop_reason raises). This fires for child calls too, but
        # the unattended runner only consults it on non-zero process exit.
        if event.output and event.output.choices:
            self._last_stop_reason = event.output.stop_reason

        if id(event) in self._pending_events:
            # The native transcript may have been drained between pending and
            # complete (for example, an operator score command). Re-check the
            # exact response ID now that the output exists.
            self._flush_pending_events()
            return

        self._complete_event(event, emitted=id(event) in self._emitted_events)

    # ------------------------------------------------------------------
    # JSONL consumer
    # ------------------------------------------------------------------

    def process_jsonl_line(self, raw: dict[str, Any]) -> None:
        """Consume one native JSONL event exactly once by its Claude UUID."""
        native_id = raw.get("uuid")
        if not isinstance(native_id, str) or not native_id:
            raise RuntimeError("Claude session event is missing its native UUID.")
        if native_id in self._seen_native_events:
            return
        self._seen_native_events.add(native_id)

        event_type = raw.get("type")
        if event_type == "assistant":
            self._handle_assistant(raw)
        elif event_type == "user":
            self._handle_user(raw)
        elif event_type == "system":
            self._handle_system(raw)

        self._flush_pending_events()

    # ------------------------------------------------------------------
    # Native identity and event emission
    # ------------------------------------------------------------------

    def _complete_event(self, event: ModelEvent, *, emitted: bool) -> None:
        """Render tool calls, open native child spans, and publish completion."""
        msg = event.output.message if event.output else None
        if msg is not None and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.view is None:
                    custom = tool_view(tc.function, tc.arguments or {})
                    if custom is not None:
                        tc.view = custom

            parent_span_id = event.span_id or self.outer_span_id
            for tc in msg.tool_calls:
                if tc.function not in ("Task", "Agent"):
                    continue
                if tc.id in self._open_agents:
                    continue
                agent_span_id = f"agent-{tc.id}"
                self._open_agents[tc.id] = _OpenAgent(span_id=agent_span_id)
                args = tc.arguments or {}
                span_name = args.get("subagent_type") or args.get("name") or "agent"
                description = args.get("description") or ""
                transcript()._event(
                    SpanBeginEvent(
                        id=agent_span_id,
                        parent_id=parent_span_id,
                        type="agent",
                        name=str(span_name),
                        metadata={"description": description} if description else None,
                    )
                )

        if emitted:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)
        else:
            transcript()._event(event)

    def _flush_pending_events(self) -> None:
        """Publish buffered calls whose native response IDs now identify a span."""
        for event_id, event in list(self._pending_events.items()):
            response_id = event.output.message.id if event.output else None
            if response_id is None or response_id not in self._response_agents:
                continue

            agent_id = self._response_agents[response_id]
            if agent_id is None:
                event.span_id = self._pending_outer_spans.get(event_id)
            else:
                tool_use_id = self._agent_to_tool.get(agent_id)
                agent = (
                    self._open_agents.get(tool_use_id)
                    if tool_use_id is not None
                    else None
                )
                if agent is None:
                    continue
                event.span_id = agent.span_id

            self._pending_outer_spans.pop(event_id, None)
            self._pending_events.pop(event_id)
            self._complete_event(event, emitted=False)

        self._flush_pending_compactions()
        self._close_completed_agents()

    def _bind_agent(self, tool_use_id: str, agent_id: str) -> None:
        """Bind a native agent ID to the exact Task/Agent tool-use that created it."""
        agent = self._open_agents.get(tool_use_id)
        if agent is None:
            return
        existing_call_id = self._agent_to_tool.get(agent_id)
        if existing_call_id is not None and existing_call_id != tool_use_id:
            raise RuntimeError("Claude native agent ID bound to multiple Task calls.")
        agent.agent_id = agent_id
        self._agent_to_tool[agent_id] = tool_use_id

    def _close_bridge_agents(self, input_messages: list[ChatMessage]) -> None:
        """Mark native agents complete when their bridge tool result returns."""
        for msg in input_messages:
            if isinstance(msg, ChatMessageTool):
                self._close_agent(msg.tool_call_id)

    def _close_agent(self, tool_use_id: str) -> None:
        """Mark one native agent complete; drain owns its exact final span end."""
        agent = self._open_agents.get(tool_use_id)
        if agent is not None:
            agent.complete = True
        self._close_completed_agents()

    def _close_completed_agents(self) -> None:
        """End completed children only after their native provenance is drained."""
        if self._pending_events or self._pending_compactions:
            return
        for tool_use_id, agent in list(self._open_agents.items()):
            if not agent.complete or agent.agent_id is None:
                continue
            self._open_agents.pop(tool_use_id)
            self._agent_to_tool.pop(agent.agent_id, None)
            transcript()._event(SpanEndEvent(id=agent.span_id))

    # ------------------------------------------------------------------
    # JSONL event handlers
    # ------------------------------------------------------------------

    def _handle_assistant(self, raw: dict[str, Any]) -> None:
        """Link a native assistant response ID to its root or subagent owner."""
        message = raw.get("message")
        if not isinstance(message, dict):
            return
        response_id = message.get("id")
        if not isinstance(response_id, str) or not response_id:
            return
        agent_id = raw.get("agentId")
        response_agent_id = (
            agent_id if isinstance(agent_id, str) and agent_id else None
        )
        if (
            response_id in self._response_agents
            and self._response_agents[response_id] != response_agent_id
        ):
            raise RuntimeError("Claude native response ID changed agent ownership.")
        self._response_agents[response_id] = response_agent_id

    def _handle_user(self, raw: dict[str, Any]) -> None:
        """Bind native child identity and completion from Task/Agent tool results."""
        message = raw.get("message", {})
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            return

        agent_id: str | None = None
        tool_use_result = raw.get("toolUseResult")
        if isinstance(tool_use_result, dict):
            value = tool_use_result.get("agentId", tool_use_result.get("agent_id"))
            if isinstance(value, str) and value:
                agent_id = value

        tool_result_ids: list[str] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("tool_use_id"), str)
                and block["tool_use_id"]
            ):
                tool_result_ids.append(block["tool_use_id"])

        # Claude's one-result event associates `toolUseResult.agentId` with
        # that exact native tool-use. Multiple result blocks cannot be safely
        # paired with a single agent ID, so reject the changed schema rather
        # than inventing an ordering rule.
        if agent_id is not None:
            if len(tool_result_ids) != 1:
                raise RuntimeError(
                    "Claude native agent result did not contain one tool result."
                )
            self._bind_agent(tool_result_ids[0], agent_id)

        for tool_use_id in tool_result_ids:
            self._close_agent(tool_use_id)

    def _handle_system(self, raw: dict[str, Any]) -> None:
        """Record native Claude compaction boundaries under their real owner."""
        if raw.get("subtype") != "compact_boundary":
            return

        agent_id = raw.get("agentId")
        if isinstance(agent_id, str) and agent_id:
            if agent_id not in self._agent_to_tool:
                self._pending_compactions.append(raw)
                return
            tool_use_id = self._agent_to_tool[agent_id]
            agent = self._open_agents.get(tool_use_id)
            if agent is not None:
                self._emit_compaction(raw, agent.span_id)
                return

        self._emit_compaction(raw, self._centaur_outer_span_id or self.outer_span_id)

    def _flush_pending_compactions(self) -> None:
        remaining: list[dict[str, Any]] = []
        for raw in self._pending_compactions:
            agent_id = raw.get("agentId")
            tool_use_id = (
                self._agent_to_tool.get(agent_id)
                if isinstance(agent_id, str)
                else None
            )
            agent = self._open_agents.get(tool_use_id) if tool_use_id else None
            if agent is None:
                remaining.append(raw)
            else:
                self._emit_compaction(raw, agent.span_id)
        self._pending_compactions = remaining

    def _emit_compaction(self, raw: dict[str, Any], span_id: str | None) -> None:
        compact_meta = raw.get("compactMetadata")
        if not isinstance(compact_meta, dict):
            compact_meta = {}
        transcript()._event(
            CompactionEvent(
                source="claude_code",
                tokens_before=compact_meta.get("preTokens"),
                span_id=span_id,
                metadata={
                    "trigger": compact_meta.get("trigger", "auto"),
                    "content": raw.get("content") or "Conversation compacted",
                },
            )
        )
