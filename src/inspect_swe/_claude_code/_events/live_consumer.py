"""Bridge and native-transcript recorder for Claude Code.

Child model events are associated only through the native session JSONL IDs
emitted by Claude. The recorder never infers an agent relationship from prompt
text, model name, timing, or generic conversation structure.
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
    span_id: str
    agent_id: str | None = None
    complete: bool = False


class LiveConsumer(ModelEventSink):
    """Attribute bridge calls from exact native Claude session identifiers."""

    def __init__(self) -> None:
        self._open_agents: dict[str, _OpenAgent] = {}
        self._agent_to_tool: dict[str, str] = {}
        self._response_agents: dict[str, str | None] = {}
        self._pending_events: dict[int, ModelEvent] = {}
        self._pending_outer_spans: dict[int, str | None] = {}
        self._pending_compactions: list[dict[str, Any]] = []
        self._seen_native_events: set[str] = set()
        self._sandbox: SandboxEnvironment | None = None
        self._user: str | None = None
        self._session_id: str | None = None
        self._centaur_outer_span_id: str | None = None
        self._emitted_events: set[int] = set()
        self._last_stop_reason: StopReason | None = None

    @property
    def last_stop_reason(self) -> StopReason | None:
        return self._last_stop_reason

    @property
    def outer_span_id(self) -> str | None:
        return current_span_id()

    def configure_centaur_session(
        self, sandbox: SandboxEnvironment, user: str | None, session_id: str
    ) -> None:
        """Configure the wrapper-owned transcript source drained by refresh()."""
        self._sandbox = sandbox
        self._user = user
        self._session_id = session_id
        self._centaur_outer_span_id = self.outer_span_id

    async def refresh(self, command: str) -> None:
        """Drain the configured native session before a terminal lifecycle action."""
        del command
        if self._sandbox is None or self._session_id is None:
            raise RuntimeError("Claude Centaur session has not been configured.")
        session_file = shlex.quote(f"{self._session_id}.jsonl")
        subagent_file = shlex.quote(f"*/{self._session_id}/subagents/agent-*.jsonl")
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
            raise RuntimeError(f"Unable to enumerate Claude session transcript: {result.stderr}")
        for path in result.stdout.splitlines():
            content = await self._sandbox.read_file(path)
            for line in content.splitlines():
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as ex:
                    raise RuntimeError(f"Malformed Claude session JSONL in {path}.") from ex
                if not isinstance(raw, dict):
                    raise RuntimeError(f"Unexpected non-object Claude session JSONL record in {path}.")
                self.process_jsonl_line(raw)
        self._flush_pending_events()

    def reset(self) -> None:
        """Close outstanding native spans and release buffered bridge events."""
        pending = list(self._pending_events.values())
        self._pending_events.clear()
        self._pending_outer_spans.clear()
        for event in pending:
            event.span_id = None
            self._complete_event(event, emitted=False)
        for tool_use_id in reversed(list(self._open_agents)):
            transcript()._event(SpanEndEvent(id=self._open_agents.pop(tool_use_id).span_id))
        self._agent_to_tool.clear()
        self._response_agents.clear()
        self._pending_compactions.clear()
        self._seen_native_events.clear()
        self._emitted_events.clear()
        self._last_stop_reason = None
        self._sandbox = None
        self._user = None
        self._session_id = None
        self._centaur_outer_span_id = None

    def on_pending(self, event: ModelEvent) -> None:
        self._close_bridge_agents(event.input)
        outer_span_id = self.outer_span_id
        if self._open_agents:
            self._pending_events[id(event)] = event
            self._pending_outer_spans[id(event)] = outer_span_id
            return
        event.span_id = outer_span_id
        self._emitted_events.add(id(event))
        transcript()._event(event)

    def on_complete(self, event: ModelEvent) -> None:
        if event.output and event.output.choices:
            self._last_stop_reason = event.output.stop_reason
        if id(event) in self._pending_events:
            self._flush_pending_events()
            return
        self._complete_event(event, emitted=id(event) in self._emitted_events)

    def _complete_event(self, event: ModelEvent, *, emitted: bool) -> None:
        msg = event.output.message if event.output else None
        if msg is not None and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.view is None:
                    tool_call.view = tool_view(tool_call.function, tool_call.arguments or {})
                if tool_call.function not in ("Task", "Agent") or tool_call.id in self._open_agents:
                    continue
                args = tool_call.arguments or {}
                span_id = f"agent-{tool_call.id}"
                self._open_agents[tool_call.id] = _OpenAgent(span_id)
                transcript()._event(
                    SpanBeginEvent(
                        id=span_id,
                        parent_id=event.span_id or self.outer_span_id,
                        type="agent",
                        name=str(args.get("subagent_type") or args.get("name") or "agent"),
                        metadata={"description": str(args["description"])} if args.get("description") else None,
                    )
                )
        if emitted:
            self._emitted_events.discard(id(event))
            transcript()._event_updated(event)
        else:
            transcript()._event(event)

    def process_jsonl_line(self, raw: dict[str, Any]) -> None:
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

    def _handle_assistant(self, raw: dict[str, Any]) -> None:
        message = raw.get("message")
        if not isinstance(message, dict):
            return
        response_id = message.get("id")
        if not isinstance(response_id, str) or not response_id:
            return
        agent_id = raw.get("agentId")
        owner = agent_id if isinstance(agent_id, str) and agent_id else None
        if (
            response_id in self._response_agents
            and self._response_agents[response_id] != owner
        ):
            raise RuntimeError("Claude native response ID changed agent ownership.")
        self._response_agents[response_id] = owner

    def _handle_user(self, raw: dict[str, Any]) -> None:
        message = raw.get("message")
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            return
        result = raw.get("toolUseResult")
        agent_id = result.get("agentId", result.get("agent_id")) if isinstance(result, dict) else None
        agent_id = agent_id if isinstance(agent_id, str) and agent_id else None
        tool_ids = [
            block["tool_use_id"] for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
            and isinstance(block.get("tool_use_id"), str) and block["tool_use_id"]
        ]
        if agent_id is not None:
            if len(tool_ids) != 1:
                raise RuntimeError("Claude native agent result did not contain one tool result.")
            self._bind_agent(tool_ids[0], agent_id)
        for tool_use_id in tool_ids:
            self._close_agent(tool_use_id)

    def _handle_system(self, raw: dict[str, Any]) -> None:
        if raw.get("subtype") != "compact_boundary":
            return
        agent_id = raw.get("agentId")
        tool_id = self._agent_to_tool.get(agent_id) if isinstance(agent_id, str) else None
        agent = self._open_agents.get(tool_id) if tool_id else None
        if agent is None and isinstance(agent_id, str) and agent_id:
            self._pending_compactions.append(raw)
        else:
            self._emit_compaction(raw, agent.span_id if agent else self._centaur_outer_span_id or self.outer_span_id)

    def _bind_agent(self, tool_use_id: str, agent_id: str) -> None:
        agent = self._open_agents.get(tool_use_id)
        if agent is None:
            return
        previous = self._agent_to_tool.get(agent_id)
        if previous is not None and previous != tool_use_id:
            raise RuntimeError("Claude native agent ID bound to multiple Task calls.")
        agent.agent_id = agent_id
        self._agent_to_tool[agent_id] = tool_use_id

    def _close_bridge_agents(self, messages: list[ChatMessage]) -> None:
        for message in messages:
            if isinstance(message, ChatMessageTool):
                self._close_agent(message.tool_call_id)

    def _close_agent(self, tool_use_id: str) -> None:
        agent = self._open_agents.get(tool_use_id)
        if agent is not None:
            agent.complete = True
        self._close_completed_agents()

    def _close_completed_agents(self) -> None:
        if self._pending_events or self._pending_compactions:
            return
        for tool_id, agent in list(self._open_agents.items()):
            if agent.complete and agent.agent_id is not None:
                self._open_agents.pop(tool_id)
                self._agent_to_tool.pop(agent.agent_id, None)
                transcript()._event(SpanEndEvent(id=agent.span_id))

    def _flush_pending_events(self) -> None:
        for event_id, event in list(self._pending_events.items()):
            response_id = event.output.message.id if event.output else None
            if response_id not in self._response_agents:
                continue
            owner = self._response_agents[response_id]
            if owner is None:
                event.span_id = self._pending_outer_spans[event_id]
            else:
                tool_id = self._agent_to_tool.get(owner)
                agent = self._open_agents.get(tool_id) if tool_id else None
                if agent is None:
                    continue
                event.span_id = agent.span_id
            self._pending_events.pop(event_id)
            self._pending_outer_spans.pop(event_id)
            self._complete_event(event, emitted=False)
        self._flush_pending_compactions()
        self._close_completed_agents()

    def _flush_pending_compactions(self) -> None:
        remaining: list[dict[str, Any]] = []
        for raw in self._pending_compactions:
            agent_id = raw.get("agentId")
            tool_id = self._agent_to_tool.get(agent_id) if isinstance(agent_id, str) else None
            agent = self._open_agents.get(tool_id) if tool_id else None
            if agent is None:
                remaining.append(raw)
            else:
                self._emit_compaction(raw, agent.span_id)
        self._pending_compactions = remaining

    def _emit_compaction(self, raw: dict[str, Any], span_id: str | None) -> None:
        metadata = raw.get("compactMetadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        transcript()._event(
            CompactionEvent(
                source="claude_code",
                tokens_before=metadata.get("preTokens"),
                span_id=span_id,
                metadata={
                    "trigger": metadata.get("trigger", "auto"),
                    "content": raw.get("content") or "Conversation compacted",
                },
            )
        )
