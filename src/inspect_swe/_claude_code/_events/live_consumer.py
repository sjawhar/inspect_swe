"""Real-time and native-transcript consumer for Claude Code events.

The bridge installs this `ModelEventSink` so it can route each native CLI
model call to the Inspect transcript. A parent Task/Agent tool call opens a
native agent span immediately, before Claude Code can issue the child request.

Claude's durable JSONL records and stream-json stdout supply the identity bridge
events do not: `tool_result.toolUseResult.agentId` links a completed native
agent to its Task/Agent tool-use ID; sidechain assistant `message.id` links a
bridged response to that native agent; and stream-json child assistant records
link their response directly to `parent_tool_use_id`. The recorder buffers child
ModelEvents until one of those exact native IDs arrives through its native-session
drain. This retains overlapping same-model siblings without reconstructing
identity from prompt text or timing.

The same transcript drain observes native `compact_boundary` events, so an
interactive `/compact` is recorded even though the shell's unattended
stream-json stdout channel does not include child-session records.
"""

import json
import shlex
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, TypeVar

import anyio
from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model._chat_message import ChatMessage, ChatMessageTool
from inspect_ai.model._model import ModelEventSink
from inspect_ai.model._model_output import StopReason
from inspect_ai.util import SandboxEnvironment
from inspect_ai.util._span import current_span_id

from ..._util.jsonl import jsonl_lines
from .toolview import tool_view

# Match the native parser's typed conversation domain. Session JSONL also
# persists heterogeneous scheduler and attachment metadata, which has no
# transcript identity for this consumer.
_CONVERSATION_EVENT_TYPES = frozenset({"user", "assistant", "system"})

# Opens a subagent sidecar, ahead of the replayed turn that spawned it.
_FORK_CONTEXT_REF = "fork-context-ref"

T = TypeVar("T")

# Bounds on a native-session drain, one per step, none cumulative. A whole-drain
# budget false-fails a long, healthy session (replaying a real 1470-row capture at
# a plausible per-read latency crossed 120 s on a working sandbox), so each step
# -- the enumerating `find`, then every file read -- gets its own deadline and
# nothing sums.
#
# `NATIVE_DRAIN_STEP_TIMEOUT_SECONDS` is the deadline for the enumeration and the
# floor for every read. A read's deadline grows with the file: the enumeration
# reports each size, and the read may take that many bytes at
# `NATIVE_DRAIN_FLOOR_BYTES_PER_SECOND` on top of the floor. So a large session
# file that is slow but progressing is never mistaken for a dead pod (the
# provider allows 100 MiB per file; at the floor rate that is ~7 min, not 60 s),
# while a dead pod -- the case this exists for, agent-c#19253: hawk's stop had
# interrupted the sample, the pod was gone, the teardown drain's exec never
# returned, and the runner reported the run live for 15 and 25 hours -- still
# fails within the floor on the enumeration or any ordinary file.
#
# The bound has to live here, client-side: the K8s provider's own `timeout=`
# runs the `timeout` binary inside the pod, which a dead pod never executes,
# `read_file` takes no deadline at all, and its websocket read blocks with
# none. Cancelling our await does not unblock the provider's worker thread; on
# a dead pod that thread stays parked on a socket that will never answer. One
# parked thread per abandoned step is the residual cost of not hanging forever;
# closing it needs a deadline inside the provider's transport, not here.
NATIVE_DRAIN_STEP_TIMEOUT_SECONDS: float = 60.0
NATIVE_DRAIN_FLOOR_BYTES_PER_SECOND: float = 256 * 1024

# Distinguishes "root owns this response" (None) from "nobody has claimed it".
_UNSET = object()


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
    lifecycle boundaries; unattended stream-json callers may still forward
    individual parent-session lines through `process_jsonl_line`.
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

        # Stream-json child assistant message ID → originating Task/Agent
        # tool-use ID. Unlike `agentId`, `parent_tool_use_id` is emitted by
        # Claude's stdout before the authoritative sidecar reaches the drain.
        self._response_tools: dict[str, str] = {}

        # Child assistant response IDs discovered in native sidecars must each
        # be matched to the corresponding bridge ModelEvent before that child
        # span can close. This protects the completion-drain/bridge callback
        # race without inferring a relation from content or timing.
        self._native_agent_responses: dict[str, set[str]] = {}
        self._delivered_native_responses: set[str] = set()

        # A subagent sidecar opens with a `fork-context-ref` row and then
        # replays the turn that spawned it: the root's response ID under this
        # agent's ID, with a UUID of its own. That replay is inherited
        # context, not the child's work, so the agent is marked here and its
        # next assistant record claims nothing. The drain reads sidecars
        # before the root session, so the marker -- not arrival order -- is
        # what keeps the response with the root.
        self._fork_context_replay: set[str] = set()

        # That first record is HELD rather than dropped, one per fork agent.
        # A sidecar's genuine first turn carries `isSidechain` too, so the
        # replay is only confirmed once the root claims the same response.
        # Resolved at the end of the drain: claimed means replay (discard),
        # unclaimed means the child really produced it (register it then).
        self._held_fork_records: dict[str, dict[str, Any]] = {}

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

        # The wrapper configures these before the native drain may run. The
        # session id is known only on the unattended path, which pins its own
        # `claude -p`; an interactive operator starts as many sessions as they
        # like, none of them named in advance, and the drain reads them all.
        self._sandbox: SandboxEnvironment | None = None
        self._user: str | None = None
        self._session_id: str | None = None
        self._native_drain_configured = False

        # A root stream can reach its tool_result before sidecar callbacks are
        # delivered. Keep native children open until a complete transcript
        # drain has made every exact child response ID observable.
        self._native_transcript_drained = True
        self._draining_native_session = False

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
        session_id: str | None,
    ) -> None:
        """Configure the native session transcripts drained by lifecycle hooks.

        ``session_id`` names the one session to drain when the caller created it
        (the unattended ``claude -p`` path). ``None`` means every session under
        the user's Claude projects directory: an interactive operator may start
        several, and pinning one id onto their ``claude`` so the drain could
        predict a filename is exactly what made a second ``claude`` fail with
        "Session ID ... is already in use".
        """
        self._sandbox = sandbox
        self._user = user
        self._session_id = session_id
        self._native_drain_configured = True
        self._centaur_outer_span_id = self.outer_span_id
        self._native_transcript_drained = False
        self._draining_native_session = False

    async def refresh(self, command: str) -> None:
        """Drain the configured Claude session and subagent transcript files.

        ``command`` identifies the native lifecycle boundary that requested this
        drain (score, submit, process completion, or teardown). Identity
        resolution consumes only native session files, never prompt text.
        """
        if self._sandbox is None or not self._native_drain_configured:
            raise RuntimeError("Claude native session has not been configured.")

        # Root transcripts are `<project>/<session>.jsonl`; sub-agent sidecars are
        # `<project>/<session>/subagents/agent-*.jsonl`. With a pinned id both
        # patterns name that session; without one they match every session.
        session = self._session_id if self._session_id is not None else "*"
        session_file = shlex.quote(f"{session}.jsonl")
        subagent_file = shlex.quote(f"*/{session}/subagents/agent-*.jsonl")
        await self._drain_session_files(command, session_file, subagent_file)

        self._resolve_fork_context_holds()
        self._native_transcript_drained = True
        self._flush_pending_events()

    async def _drain_session_files(
        self, command: str, session_file: str, subagent_file: str
    ) -> None:
        """Enumerate and consume the session files, each step under its own bound."""
        assert self._sandbox is not None  # refresh checked
        sandbox = self._sandbox

        async def step(what: str, op: Awaitable[T], deadline: float) -> T:
            # Only THIS scope's expiry is translated. A TimeoutError the provider
            # raises itself (the K8s exec's in-pod `timeout`, exit 124) is not
            # ours to rename and propagates with its own message and cause.
            with anyio.move_on_after(deadline):
                return await op
            # Reached only when the scope above cancelled the await.
            raise RuntimeError(
                f"Claude native session drain ({command}): {what} did not answer "
                f"within {deadline:g}s; the sandbox is not answering."
            )

        result = await step(
            "enumerating session files",
            sandbox.exec(
                [
                    "sh",
                    "-c",
                    'if [ -d "$HOME/.claude/projects" ]; then '
                    f'find "$HOME/.claude/projects" -type f \\( -name {session_file} '
                    f"-o -path {subagent_file} \\) -printf '%s\\t%p\\n'; fi",
                ],
                user=self._user,
            ),
            NATIVE_DRAIN_STEP_TIMEOUT_SECONDS,
        )
        if not result.success:
            raise RuntimeError(
                f"Unable to enumerate Claude session transcripts: {result.stderr}"
            )

        self._draining_native_session = True
        try:
            sized: list[tuple[str, int]] = []
            for line in result.stdout.splitlines():
                size_text, sep, path = line.partition("\t")
                if not sep or not size_text.isdigit():
                    raise RuntimeError(
                        f"Claude session enumeration returned an unreadable line: {line!r}"
                    )
                sized.append((path, int(size_text)))
            # Claude mirrors child records in the root session without their
            # agent ID. Consume the authoritative sidecar first so its UUID
            # claims the child response before the mirror is deduplicated.
            sized.sort(key=lambda entry: "/subagents/agent-" not in entry[0])
            for path, size in sized:
                deadline = (
                    NATIVE_DRAIN_STEP_TIMEOUT_SECONDS
                    + size / NATIVE_DRAIN_FLOOR_BYTES_PER_SECOND
                )
                content = await step(
                    f"reading {path} ({size} bytes)", sandbox.read_file(path), deadline
                )
                for line in jsonl_lines(content):
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
        finally:
            self._draining_native_session = False

    def _resolve_fork_context_holds(self) -> None:
        """Settle every held fork record once the whole drain has been read.

        A response the root also wrote is the spawning turn replayed into the
        sidecar: it belongs to the root, so the held copy is discarded and the
        child never waits on it. A response nothing else claims was the fork's
        own first turn, so it registers now, exactly as it would have.
        """
        held = self._held_fork_records
        self._held_fork_records = {}
        self._fork_context_replay.clear()
        for raw in held.values():
            message = raw.get("message")
            response_id = message.get("id") if isinstance(message, dict) else None
            if (
                isinstance(response_id, str)
                and self._response_agents.get(response_id, _UNSET) is None
            ):
                # The ROOT wrote this response too, which is what makes the
                # held record the replayed spawning turn. Only that confirms
                # it: another agent holding the id is a genuine conflict, and
                # reprocessing below is what raises on it.
                continue
            self._handle_assistant(raw)

    async def drain_completion(self) -> None:
        """Drain native transcript files after an unattended CLI process exits."""
        await self.refresh("completion")

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
        self._response_tools.clear()
        self._native_agent_responses.clear()
        self._delivered_native_responses.clear()
        self._fork_context_replay.clear()
        self._held_fork_records.clear()
        self._pending_compactions.clear()
        self._seen_native_events.clear()
        self._emitted_events.clear()
        self._last_stop_reason = None
        self._centaur_outer_span_id = None
        self._sandbox = None
        self._user = None
        self._session_id = None
        self._native_drain_configured = False
        self._native_transcript_drained = True
        self._draining_native_session = False

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

    def process_jsonl_line(
        self, raw: dict[str, Any], *, from_stdout: bool = False
    ) -> None:
        """Consume a supported native conversation event exactly once by UUID.

        Claude's stream-json output can mirror sidechain records but omit the
        native agent ID that owns them. Such a record remains unowned unless it
        provides the exact `parent_tool_use_id` that identifies its Task/Agent
        span; otherwise the durable sidecar is authoritative.
        """
        event_type = raw.get("type")
        if event_type == _FORK_CONTEXT_REF:
            # Opens a subagent sidecar. The record that follows it replays the
            # spawning turn as inherited context; mark the agent so that
            # replay claims nothing.
            forked_agent = raw.get("agentId")
            if isinstance(forked_agent, str) and forked_agent:
                self._fork_context_replay.add(forked_agent)
            return
        if event_type not in _CONVERSATION_EVENT_TYPES:
            return

        native_agent_id = raw.get("agentId")
        if (
            from_stdout
            and raw.get("isSidechain") is True
            and (not isinstance(native_agent_id, str) or not native_agent_id)
        ):
            return

        native_id = raw.get("uuid")
        if not isinstance(native_id, str) or not native_id:
            raise RuntimeError("Claude session event is missing its native UUID.")
        if native_id in self._seen_native_events:
            return
        self._seen_native_events.add(native_id)

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

            tool_use_id = self._response_tools.get(response_id)
            if tool_use_id is not None:
                agent = self._open_agents.get(tool_use_id)
                if agent is None:
                    continue
                event.span_id = agent.span_id
                self._delivered_native_responses.add(response_id)
            else:
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
                    self._delivered_native_responses.add(response_id)

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
            if isinstance(msg, ChatMessageTool) and msg.tool_call_id is not None:
                self._close_agent(msg.tool_call_id)

    def _close_agent(self, tool_use_id: str) -> None:
        """Mark one native agent complete; drain owns its exact final span end."""
        agent = self._open_agents.get(tool_use_id)
        if agent is not None:
            agent.complete = True
        self._close_completed_agents()

    def _close_completed_agents(self) -> None:
        """End completed children only after their native provenance is drained."""
        if (
            self._draining_native_session
            or not self._native_transcript_drained
            or self._pending_events
            or self._pending_compactions
        ):
            return
        for tool_use_id, agent in list(self._open_agents.items()):
            if not agent.complete or agent.agent_id is None:
                continue
            expected_responses = self._native_agent_responses.get(agent.agent_id, set())
            if not expected_responses.issubset(self._delivered_native_responses):
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
        response_agent_id = agent_id if isinstance(agent_id, str) and agent_id else None

        if (
            raw.get("isSidechain") is True
            and response_agent_id is not None
            and response_agent_id in self._fork_context_replay
        ):
            self._fork_context_replay.discard(response_agent_id)
            if response_agent_id not in self._held_fork_records:
                # The first record after this agent's `fork-context-ref` is
                # the spawning turn replayed as inherited context -- unless
                # this fork wrote its own first turn there instead, which the
                # record itself cannot tell us. Hold it: the root claiming
                # this response later proves it was the replay, and nothing
                # claiming it proves it was the child's own work.
                self._held_fork_records[response_agent_id] = raw
                return
        parent_tool_use_id = raw.get("parent_tool_use_id")
        if parent_tool_use_id is not None and (
            not isinstance(parent_tool_use_id, str) or not parent_tool_use_id
        ):
            raise RuntimeError("Claude native assistant parent tool ID is invalid.")

        if (
            response_id in self._response_agents
            and self._response_agents[response_id] != response_agent_id
        ):
            # A sidechain record claiming a response the root already holds is
            # either of two things, and only one of them is a conflict.
            if (
                raw.get("isSidechain") is True
                and response_agent_id is not None
                and self._response_agents[response_id] is None
            ):
                if response_id not in self._response_tools:
                    # A subagent sidecar opens by replaying the turn that
                    # spawned it: the root's response ID under the child's
                    # agent ID, with a UUID of its own, so the dedupe above
                    # does not reach it. That replay is context, not the
                    # child's work. Owning it would attribute the root's call
                    # to the child and leave the child's span waiting on a
                    # response it never produces.
                    return
                # Otherwise the response is already tied to a Task span by a
                # stdout mirror that could not name its agent, and the durable
                # sidecar is authoritative for the owner: adopt it.
            else:
                raise RuntimeError("Claude native response ID changed agent ownership.")
        if (
            parent_tool_use_id is not None
            and response_id in self._response_tools
            and self._response_tools[response_id] != parent_tool_use_id
        ):
            raise RuntimeError(
                "Claude native response ID changed parent tool ownership."
            )

        self._response_agents[response_id] = response_agent_id
        if parent_tool_use_id is not None:
            self._response_tools[response_id] = parent_tool_use_id
        if response_agent_id is not None:
            self._native_agent_responses.setdefault(response_agent_id, set()).add(
                response_id
            )

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
                self._agent_to_tool.get(agent_id) if isinstance(agent_id, str) else None
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
