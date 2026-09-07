"""Native Gemini CLI telemetry recording.

Gemini CLI writes completed OpenTelemetry spans and logs to the configured
``telemetry.outfile``.  This consumer drains that file at explicit native-command
boundaries.  It records only IDs and parent links the CLI exports; bridge model
calls wait until their selected W3C traceparent identifies an exact native LLM
span.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, TypeAlias

from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import BRIDGE_REQUEST_HEADERS
from inspect_ai.model._model import ModelEventSink
from inspect_ai.util._span import current_span_id

_JsonObject: TypeAlias = dict[str, object]
_HrTime: TypeAlias = tuple[int, int]
_NativeKey: TypeAlias = tuple[str, str]


class _TelemetrySandbox(Protocol):
    """Minimal sandbox surface needed to read native telemetry."""

    async def read_file(self, path: str) -> str: ...


_AGENT_CALL = "agent_call"
_TOOL_CALL = "tool_call"
_LLM_CALL = "llm_call"
_SCHEDULE_TOOL_CALLS = "schedule_tool_calls"
_COMPRESSION_EVENT = "gemini_cli.chat_compression"


@dataclass(frozen=True)
class _NativeSpan:
    """A completed native Gemini CLI OpenTelemetry span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    tool_call_id: str | None
    name: str
    type: str
    start: _HrTime
    end: _HrTime


@dataclass(frozen=True)
class _NativeCompaction:
    """A native Gemini CLI chat-compression telemetry log."""

    record_key: str
    trace_id: str
    span_id: str
    timestamp: _HrTime
    tokens_before: int
    tokens_after: int


@dataclass
class _PendingBridgeModelEvent:
    """One bridge event awaiting its exact exported native LLM span."""

    event: ModelEvent
    completed: bool = False
    emitted: bool = False


_TRACEPARENT = re.compile(r"\A00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}\Z")


class GeminiConsumer(ModelEventSink):
    """Record bridge calls plus native Gemini CLI OTEL spans.

    ``refresh()`` is intentionally re-entrant: every call reads the whole
    append-only telemetry file and emits only records not drained before.  The
    Centaur lifecycle invokes it after a completed Gemini process and before a
    score or submit command is handled.
    """

    def __init__(
        self,
        sandbox: _TelemetrySandbox | None = None,
        telemetry_path: str | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._telemetry_path = telemetry_path
        self._pending_model_events: dict[_NativeKey, _PendingBridgeModelEvent] = {}
        self._model_event_keys: dict[int, _NativeKey] = {}
        self._claimed_model_keys: set[_NativeKey] = set()
        self._emitted_spans: dict[_NativeKey, _NativeSpan] = {}
        self._pending_spans: dict[_NativeKey, _NativeSpan] = {}
        self._emitted_compactions: set[str] = set()
        self._pending_compactions: dict[str, _NativeCompaction] = {}

    @property
    def outer_span_id(self) -> str | None:
        """Resolve the outer parent for a root native Gemini span."""
        return current_span_id()

    def on_pending(self, event: ModelEvent) -> None:
        """Buffer a bridge event until its exact native LLM span is available."""
        key = _bridge_model_key(event)
        if key in self._claimed_model_keys:
            raise RuntimeError(
                f"duplicate Gemini bridge ModelEvent claims native LLM span {key!r}"
            )
        self._claimed_model_keys.add(key)
        self._pending_model_events[key] = _PendingBridgeModelEvent(event=event)
        self._model_event_keys[id(event)] = key
        self._flush_pending_model_events()

    def on_complete(self, event: ModelEvent) -> None:
        """Update a bridge event only after native-span emission."""
        key = self._model_event_keys.get(id(event))
        if key is None:
            raise RuntimeError(
                "Gemini bridge ModelEvent completed before it was pending"
            )
        pending = self._pending_model_events.get(key)
        if pending is None or pending.event is not event:
            raise RuntimeError(
                "Gemini bridge ModelEvent completion lost its pending identity"
            )
        if pending.completed:
            raise RuntimeError("Gemini bridge ModelEvent completed more than once")
        pending.completed = True
        if pending.emitted:
            transcript()._event_updated(event)
            self._remove_pending_model_event(key, pending)

    async def refresh(self, command: str) -> None:
        """Drain completed native telemetry before a score or submit command."""
        del command
        await self._drain_configured_telemetry()

    async def finalize(self) -> None:
        """Drain final telemetry and reject unresolved bridge or native identities."""
        await self._drain_configured_telemetry()
        self._assert_no_pending_model_events()
        self._assert_no_pending_native_records()

    async def _drain_configured_telemetry(self) -> None:
        if self._sandbox is None or self._telemetry_path is None:
            raise RuntimeError(
                "Gemini telemetry refresh requires sandbox and telemetry path"
            )
        await self.process_telemetry_from_sandbox(self._sandbox, self._telemetry_path)

    async def process_telemetry_from_sandbox(
        self, sandbox: _TelemetrySandbox, telemetry_path: str
    ) -> None:
        """Read and consume the CLI's append-only OTEL file from the sandbox."""
        contents = await sandbox.read_file(telemetry_path)
        self.process_telemetry(contents)

    def process_telemetry(self, contents: str) -> None:
        """Emit only native records whose full parent chain is available."""
        records = _decode_concatenated_json(contents)
        self._stage_spans(
            [span for record in records if (span := _native_span(record)) is not None]
        )
        self._stage_compactions(
            [
                compaction
                for record in records
                if (compaction := _native_compaction(record)) is not None
            ]
        )

        ready_spans = self._ready_spans()
        ready_keys = {_span_key(span) for span in ready_spans}
        ready_compactions = [
            compaction
            for compaction in self._pending_compactions.values()
            if _span_key_from_parts(compaction.trace_id, compaction.span_id)
            in self._emitted_spans
            or _span_key_from_parts(compaction.trace_id, compaction.span_id)
            in ready_keys
        ]

        depths = _span_depths(
            {_span_key(span): span for span in ready_spans},
            set(self._emitted_spans),
        )
        timeline: list[tuple[_HrTime, int, int, _NativeSpan | _NativeCompaction]] = []
        for span in ready_spans:
            depth = depths[_span_key(span)]
            timeline.append((span.start, 0, depth, span))
            timeline.append((span.end, 2, -depth, span))
        for compaction in ready_compactions:
            timeline.append((compaction.timestamp, 1, 0, compaction))

        for _, phase, _, payload in sorted(timeline, key=lambda event: event[:3]):
            if phase == 0:
                if not isinstance(payload, _NativeSpan):
                    raise TypeError("Gemini telemetry timeline expected a native span")
                parent_id = (
                    self.outer_span_id
                    if payload.parent_span_id is None
                    else payload.parent_span_id
                )
                metadata: dict[str, str] = {"native_trace_id": payload.trace_id}
                if payload.parent_span_id is not None:
                    metadata["native_parent_span_id"] = payload.parent_span_id
                if payload.tool_call_id is not None:
                    metadata["tool_call_id"] = payload.tool_call_id
                transcript()._event(
                    SpanBeginEvent(
                        id=payload.span_id,
                        parent_id=parent_id,
                        type=payload.type,
                        name=payload.name,
                        metadata=metadata,
                        timestamp=_as_datetime(payload.start),
                    )
                )
            elif phase == 2:
                if not isinstance(payload, _NativeSpan):
                    raise TypeError("Gemini telemetry timeline expected a native span")
                transcript()._event(
                    SpanEndEvent(
                        id=payload.span_id,
                        timestamp=_as_datetime(payload.end),
                    )
                )
            else:
                if not isinstance(payload, _NativeCompaction):
                    raise TypeError(
                        "Gemini telemetry timeline expected a native compaction"
                    )
                transcript()._event(
                    CompactionEvent(
                        source="gemini_cli",
                        span_id=payload.span_id,
                        tokens_before=payload.tokens_before,
                        tokens_after=payload.tokens_after,
                        timestamp=_as_datetime(payload.timestamp),
                    )
                )

        for span in ready_spans:
            key = _span_key(span)
            self._pending_spans.pop(key)
            self._emitted_spans[key] = span
        self._flush_pending_model_events()
        for compaction in ready_compactions:
            self._pending_compactions.pop(compaction.record_key)
            self._emitted_compactions.add(compaction.record_key)

    def reset(self) -> None:
        """Clear state only after every bridge and native record is resolved."""
        self._assert_no_pending_model_events()
        self._assert_no_pending_native_records()
        self._model_event_keys.clear()
        self._claimed_model_keys.clear()
        self._emitted_spans.clear()
        self._emitted_compactions.clear()

    def _flush_pending_model_events(self) -> None:
        """Emit bridge events only after their exact native LLM span has emitted."""
        for key, pending in list(self._pending_model_events.items()):
            native = self._emitted_spans.get(key) or self._pending_spans.get(key)
            if native is None:
                continue
            if native.type != "model":
                raise RuntimeError(
                    "Gemini bridge traceparent resolves to a span that is not a "
                    f"native Gemini LLM span: {key!r}"
                )
            if key not in self._emitted_spans:
                continue
            if pending.emitted:
                continue
            pending.event.span_id = native.span_id
            transcript()._event(pending.event)
            pending.emitted = True
            if pending.completed:
                self._remove_pending_model_event(key, pending)

    def _remove_pending_model_event(
        self, key: _NativeKey, pending: _PendingBridgeModelEvent
    ) -> None:
        """Forget one emitted event while retaining its claimed native identity."""
        self._pending_model_events.pop(key)
        recorded_key = self._model_event_keys.pop(id(pending.event), None)
        if recorded_key != key:
            raise RuntimeError("Gemini bridge ModelEvent identity bookkeeping diverged")

    def _assert_no_pending_model_events(self) -> None:
        if not self._pending_model_events:
            return
        keys = ", ".join(repr(key) for key in self._pending_model_events)
        raise RuntimeError(
            "Gemini telemetry finished with unresolved bridge ModelEvent identities: "
            + keys
        )

    def _stage_spans(self, spans: list[_NativeSpan]) -> None:
        for span in spans:
            key = _span_key(span)
            existing = self._emitted_spans.get(key) or self._pending_spans.get(key)
            if existing is None:
                self._pending_spans[key] = span
            elif existing != span:
                raise ValueError(
                    f"conflicting Gemini telemetry records for span {key!r}"
                )

    def _stage_compactions(self, compactions: list[_NativeCompaction]) -> None:
        for compaction in compactions:
            existing = self._pending_compactions.get(compaction.record_key)
            if compaction.record_key in self._emitted_compactions:
                continue
            if existing is None:
                self._pending_compactions[compaction.record_key] = compaction
            elif existing != compaction:
                raise ValueError("conflicting Gemini telemetry compaction record")

    def _ready_spans(self) -> list[_NativeSpan]:
        ready: dict[_NativeKey, bool] = {}

        def is_ready(key: _NativeKey, trail: set[_NativeKey]) -> bool:
            cached = ready.get(key)
            if cached is not None:
                return cached
            if key in trail:
                raise ValueError(f"cycle in Gemini native span ancestry at {key!r}")
            span = self._pending_spans[key]
            parent_key = _parent_key(span)
            if parent_key is None or parent_key in self._emitted_spans:
                result = True
            elif parent_key not in self._pending_spans:
                result = False
            else:
                result = is_ready(parent_key, trail | {key})
            ready[key] = result
            return result

        return [
            span for key, span in self._pending_spans.items() if is_ready(key, set())
        ]

    def _assert_no_pending_native_records(self) -> None:
        if not self._pending_spans and not self._pending_compactions:
            return
        spans = ", ".join(
            f"{key!r} -> {span.parent_span_id!r}"
            for key, span in self._pending_spans.items()
        )
        compactions = ", ".join(
            f"{compaction.trace_id}/{compaction.span_id}"
            for compaction in self._pending_compactions.values()
        )
        raise RuntimeError(
            "Gemini telemetry finished with unresolved native parent records: "
            + "; ".join(item for item in (spans, compactions) if item)
        )


def _bridge_model_key(event: ModelEvent) -> _NativeKey:
    """Require the exact W3C identity selected into bridge request metadata."""
    metadata = event.metadata
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Gemini bridge ModelEvent is missing traceparent metadata")
    headers = metadata.get(BRIDGE_REQUEST_HEADERS)
    if not isinstance(headers, Mapping):
        raise RuntimeError("Gemini bridge ModelEvent is missing traceparent headers")
    traceparent = headers.get("traceparent")
    if not isinstance(traceparent, str):
        raise RuntimeError("Gemini bridge ModelEvent is missing traceparent")
    match = _TRACEPARENT.fullmatch(traceparent)
    if match is None:
        raise RuntimeError(
            f"Gemini bridge ModelEvent has malformed traceparent {traceparent!r}"
        )
    return match.group(1), match.group(2)


def _decode_concatenated_json(contents: str) -> list[_JsonObject]:
    """Decode the pretty JSON values appended by Gemini's FileSpanExporter."""
    decoder = json.JSONDecoder()
    records: list[_JsonObject] = []
    offset = 0
    while True:
        while offset < len(contents) and contents[offset].isspace():
            offset += 1
        if offset == len(contents):
            return records
        try:
            value, offset = decoder.raw_decode(contents, offset)
        except json.JSONDecodeError as ex:
            raise ValueError(
                f"invalid JSON in Gemini telemetry at byte {ex.pos}"
            ) from ex
        if not isinstance(value, dict):
            raise ValueError("Gemini telemetry record must be a JSON object")
        records.append(value)


def _native_span(record: _JsonObject) -> _NativeSpan | None:
    attributes = _object_or_none(record, "attributes")
    if attributes is None:
        return None
    operation = attributes.get("gen_ai.operation.name")
    if operation not in (
        _TOOL_CALL,
        _SCHEDULE_TOOL_CALLS,
        _AGENT_CALL,
        _LLM_CALL,
    ):
        return None

    context = _file_exporter_object(record, "spanContext")
    trace_id = _string(context, "traceId")
    span_id = _string(context, "spanId")
    parent_context = _file_exporter_parent_context(record)
    parent_span_id = (
        _string(parent_context, "spanId") if parent_context is not None else None
    )
    name = _span_name(operation, record, attributes)
    tool_call_id = _optional_string(attributes, "gen_ai.tool.call_id")
    if operation == _TOOL_CALL and name == "invoke_agent" and tool_call_id is None:
        raise ValueError("Gemini invoke_agent span has no gen_ai.tool.call_id")
    span_type = (
        "tool"
        if operation in (_TOOL_CALL, _SCHEDULE_TOOL_CALLS)
        else "agent"
        if operation == _AGENT_CALL
        else "model"
    )
    return _NativeSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        tool_call_id=tool_call_id,
        name=name,
        type=span_type,
        start=_hrtime(record, "startTime"),
        end=_hrtime(record, "endTime"),
    )


def _native_compaction(record: _JsonObject) -> _NativeCompaction | None:
    attributes = _object_or_none(record, "attributes")
    if attributes is None:
        return None
    if attributes.get("event.name") != _COMPRESSION_EVENT:
        return None
    context = _file_exporter_object(record, "spanContext")
    return _NativeCompaction(
        record_key=json.dumps(record, sort_keys=True, separators=(",", ":")),
        trace_id=_string(context, "traceId"),
        span_id=_string(context, "spanId"),
        timestamp=_hrtime(record, "hrTime"),
        tokens_before=_integer(attributes, "tokens_before"),
        tokens_after=_integer(attributes, "tokens_after"),
    )


def _span_name(operation: object, record: _JsonObject, attributes: _JsonObject) -> str:
    if operation == _TOOL_CALL:
        tool_name = attributes.get("gen_ai.tool.name")
        if isinstance(tool_name, str) and tool_name:
            return tool_name
    if operation == _AGENT_CALL:
        agent_name = attributes.get("gen_ai.agent.name")
        if isinstance(agent_name, str) and agent_name:
            return agent_name
    if operation == _LLM_CALL:
        model_name = attributes.get("gen_ai.request.model")
        if isinstance(model_name, str) and model_name:
            return model_name
    return _string(record, "name")


def _object(record: _JsonObject, key: str) -> _JsonObject:
    value = _object_or_none(record, key)
    if value is None:
        raise ValueError(f"Gemini telemetry record has no object {key!r}")
    return value


def _object_or_none(record: _JsonObject, key: str) -> _JsonObject | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Gemini telemetry field {key!r} must be an object")
    return value


def _file_exporter_object(record: _JsonObject, key: str) -> _JsonObject:
    """Read a required private field emitted by Gemini's FileSpanExporter."""
    return _object(record, f"_{key}")


def _file_exporter_parent_context(record: _JsonObject) -> _JsonObject | None:
    """Read the exported public parent context without accepting synthetic aliases."""
    public = _object_or_none(record, "parentSpanContext")
    private = _object_or_none(record, "_parentSpanContext")
    if private is not None and public is None:
        raise ValueError("Gemini telemetry parent context must use 'parentSpanContext'")
    if private is not None and private != public:
        raise ValueError("Gemini telemetry has conflicting parent context fields")
    return public


def _string(record: _JsonObject, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Gemini telemetry field {key!r} must be a non-empty string")
    return value


def _optional_string(record: _JsonObject, key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Gemini telemetry field {key!r} must be a non-empty string")
    return value


def _span_key(span: _NativeSpan) -> _NativeKey:
    return _span_key_from_parts(span.trace_id, span.span_id)


def _span_key_from_parts(trace_id: str, span_id: str) -> _NativeKey:
    return trace_id, span_id


def _parent_key(span: _NativeSpan) -> _NativeKey | None:
    if span.parent_span_id is None:
        return None
    return _span_key_from_parts(span.trace_id, span.parent_span_id)


def _integer(record: _JsonObject, key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Gemini telemetry field {key!r} must be an integer")
    return value


def _hrtime(record: _JsonObject, key: str) -> _HrTime:
    value = record.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(
            isinstance(part, int) and not isinstance(part, bool) for part in value
        )
    ):
        raise ValueError(f"Gemini telemetry field {key!r} must be a two-integer hrtime")
    return value[0], value[1]


def _span_depths(
    spans: dict[_NativeKey, _NativeSpan], emitted: set[_NativeKey]
) -> dict[_NativeKey, int]:
    """Return native parent depth for records whose ancestry is resolved."""
    depths: dict[_NativeKey, int] = {}

    def depth(key: _NativeKey, trail: set[_NativeKey]) -> int:
        if key in depths:
            return depths[key]
        if key in trail:
            raise ValueError(f"cycle in Gemini native span ancestry at {key!r}")
        parent_key = _parent_key(spans[key])
        if parent_key is None or parent_key in emitted:
            result = 0
        else:
            result = depth(parent_key, trail | {key}) + 1
        depths[key] = result
        return result

    for key in spans:
        depth(key, set())
    return depths


def _as_datetime(value: _HrTime) -> datetime:
    seconds, nanoseconds = value
    return datetime.fromtimestamp(seconds + nanoseconds / 1_000_000_000, timezone.utc)
