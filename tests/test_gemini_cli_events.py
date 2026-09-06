"""Focused regression tests for the Gemini CLI native telemetry consumer."""

import json
from dataclasses import dataclass, field
from unittest.mock import patch

import anyio
import pytest
from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent

import inspect_swe._gemini_cli._events.consumer as consumer_module
from inspect_swe._gemini_cli._events.consumer import GeminiConsumer


@dataclass
class _TranscriptRecorder:
    events: list[object] = field(default_factory=list)

    def _event(self, event: object) -> None:
        self.events.append(event)

    def _event_updated(self, event: object) -> None:
        self.events.append(event)


def _span(
    span_id: str,
    parent_span_id: str | None,
    *,
    name: str,
    operation: str,
    start: int,
    end: int,
    attributes: dict[str, object] | None = None,
    trace_id: str = "trace-1",
) -> dict[str, object]:
    record: dict[str, object] = {
        "name": name,
        "_spanContext": {"traceId": trace_id, "spanId": span_id},
        "startTime": [start, 0],
        "endTime": [end, 0],
        "attributes": {
            "gen_ai.operation.name": operation,
            **(attributes or {}),
        },
    }
    if parent_span_id is not None:
        record["parentSpanContext"] = {"spanId": parent_span_id}
    return record




def _compression(span_id: str, before: int, after: int, timestamp: int) -> dict[str, object]:
    return {
        "_spanContext": {"traceId": "trace-1", "spanId": span_id},
        "hrTime": [timestamp, 0],
        "attributes": {
            "event.name": "gemini_cli.chat_compression",
            "tokens_before": before,
            "tokens_after": after,
        },
    }


def _actual_child_delegation_telemetry() -> str:
    """Minimal native hierarchy from the pinned v0.58 child delegation capture."""
    trace_id = "aa0822907bc15f57ddac3218141d5364"
    records = [
        _span(
            "986e05d67a7e0a0e",
            None,
            name="schedule_tool_calls",
            operation="schedule_tool_calls",
            start=100,
            end=190,
            trace_id=trace_id,
            attributes={"gen_ai.agent.name": "gemini-cli"},
        ),
        _span(
            "1d65cda9b237db76",
            "986e05d67a7e0a0e",
            name="tool_call",
            operation="tool_call",
            start=101,
            end=180,
            trace_id=trace_id,
            attributes={
                "gen_ai.tool.name": "invoke_agent",
                "gen_ai.tool.call_id": "invoke_agent__invoke_agent_1788673951848_0",
            },
        ),
        _span(
            "625d259a22caa7e5",
            "1d65cda9b237db76",
            name="agent_call",
            operation="agent_call",
            start=102,
            end=170,
            trace_id=trace_id,
            attributes={"gen_ai.agent.name": "gemini-cli"},
        ),
        _span(
            "5532ec2bd08cdb4e",
            "625d259a22caa7e5",
            name="llm_call",
            operation="llm_call",
            start=103,
            end=120,
            trace_id=trace_id,
            attributes={"gen_ai.request.model": "gemini-2.5-pro"},
        ),
        _span(
            "9e3ae8370bfc7e8a",
            "625d259a22caa7e5",
            name="schedule_tool_calls",
            operation="schedule_tool_calls",
            start=121,
            end=150,
            trace_id=trace_id,
            attributes={"gen_ai.agent.name": "gemini-cli"},
        ),
        _span(
            "49d3b3b8304792e4",
            "9e3ae8370bfc7e8a",
            name="tool_call",
            operation="tool_call",
            start=122,
            end=140,
            trace_id=trace_id,
            attributes={
                "gen_ai.tool.name": "read_file",
                "gen_ai.tool.call_id": "f9028e0a-79f2-4185-820c-376d9eb0d0a5#0-0",
            },
        ),
    ]
    return "\n".join(json.dumps(record) for record in records)


def _native_telemetry() -> str:
    records = [
        _span(
            "llm-parent",
            None,
            name="llm_call",
            operation="llm_call",
            start=100,
            end=108,
            attributes={"gen_ai.request.model": "gemini-2.5-pro"},
        ),
        _span(
            "tool-invoke-1",
            None,
            name="tool_call",
            operation="tool_call",
            start=101,
            end=107,
            attributes={
                "gen_ai.tool.name": "invoke_agent",
                "gen_ai.tool.call_id": "invoke-agent-call-1",
            },
        ),
        _span(
            "agent-1",
            "tool-invoke-1",
            name="agent_call",
            operation="agent_call",
            start=102,
            end=106,
            attributes={"gen_ai.agent.name": "investigator"},
        ),
        _span(
            "llm-child",
            "agent-1",
            name="llm_call",
            operation="llm_call",
            start=103,
            end=105,
            attributes={"gen_ai.request.model": "gemini-2.5-pro"},
        ),
        _compression("llm-child", before=900, after=300, timestamp=104),
    ]
    # Gemini's FileSpanExporter writes one pretty JSON value at a time, not JSONL.
    return "\n".join(json.dumps(record, indent=2) for record in records)


def test_native_telemetry_creates_real_tool_agent_tree_and_compaction() -> None:
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        consumer.process_telemetry(_native_telemetry())

    assert [type(event) for event in recorder.events] == [
        SpanBeginEvent,
        SpanBeginEvent,
        SpanBeginEvent,
        SpanBeginEvent,
        CompactionEvent,
        SpanEndEvent,
        SpanEndEvent,
        SpanEndEvent,
        SpanEndEvent,
    ]

    parent_llm, invoke_tool, child_agent, child_llm = recorder.events[:4]
    assert isinstance(parent_llm, SpanBeginEvent)
    assert parent_llm.id == "llm-parent"
    assert parent_llm.parent_id == "outer-span"
    assert parent_llm.type == "model"
    assert parent_llm.name == "gemini-2.5-pro"
    assert parent_llm.metadata == {"native_trace_id": "trace-1"}

    assert isinstance(invoke_tool, SpanBeginEvent)
    assert invoke_tool.id == "tool-invoke-1"
    assert invoke_tool.parent_id == "outer-span"
    assert invoke_tool.type == "tool"
    assert invoke_tool.name == "invoke_agent"
    assert invoke_tool.metadata == {
        "native_trace_id": "trace-1",
        "tool_call_id": "invoke-agent-call-1",
    }

    assert isinstance(child_agent, SpanBeginEvent)
    assert child_agent.id == "agent-1"
    assert child_agent.parent_id == "tool-invoke-1"
    assert child_agent.type == "agent"
    assert child_agent.name == "investigator"
    assert child_agent.metadata == {
        "native_trace_id": "trace-1",
        "native_parent_span_id": "tool-invoke-1",
    }

    assert isinstance(child_llm, SpanBeginEvent)
    assert child_llm.id == "llm-child"
    assert child_llm.parent_id == "agent-1"
    assert child_llm.type == "model"

    compaction = recorder.events[4]
    assert isinstance(compaction, CompactionEvent)
    assert compaction.source == "gemini_cli"
    assert compaction.span_id == "llm-child"
    assert compaction.tokens_before == 900
    assert compaction.tokens_after == 300

    ends = recorder.events[5:]
    assert [event.id for event in ends if isinstance(event, SpanEndEvent)] == [
        "llm-child",
        "agent-1",
        "tool-invoke-1",
        "llm-parent",
    ]


def test_native_telemetry_is_idempotent_at_command_refresh_boundaries() -> None:
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        consumer.process_telemetry(_native_telemetry())
        consumer.process_telemetry(_native_telemetry())

    assert len(recorder.events) == 9



def test_same_name_native_agent_siblings_keep_distinct_exported_ids() -> None:
    records = [
        _span(
            "llm-parent",
            None,
            name="llm_call",
            operation="llm_call",
            start=100,
            end=108,
            attributes={"gen_ai.request.model": "gemini-2.5-pro"},
        ),
        _span(
            "agent-1",
            "llm-parent",
            name="agent_call",
            operation="agent_call",
            start=101,
            end=103,
            attributes={"gen_ai.agent.name": "investigator"},
        ),
        _span(
            "agent-2",
            "llm-parent",
            name="agent_call",
            operation="agent_call",
            start=104,
            end=106,
            attributes={"gen_ai.agent.name": "investigator"},
        ),
    ]
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        consumer.process_telemetry(
            "\n".join(json.dumps(record, indent=2) for record in records)
        )

    begins = [event for event in recorder.events if isinstance(event, SpanBeginEvent)]
    assert [(event.id, event.parent_id, event.name) for event in begins] == [
        ("llm-parent", "outer-span", "gemini-2.5-pro"),
        ("agent-1", "llm-parent", "investigator"),
        ("agent-2", "llm-parent", "investigator"),
    ]


def test_native_telemetry_buffers_out_of_order_native_parents() -> None:
    tool = _span(
        "tool-invoke-1",
        None,
        name="tool_call",
        operation="tool_call",
        start=101,
        end=107,
        attributes={
            "gen_ai.tool.name": "invoke_agent",
            "gen_ai.tool.call_id": "invoke-agent-call-1",
        },
    )
    agent = _span(
        "agent-1",
        "tool-invoke-1",
        name="agent_call",
        operation="agent_call",
        start=102,
        end=106,
        attributes={"gen_ai.agent.name": "investigator"},
    )
    llm = _span(
        "llm-child",
        "agent-1",
        name="llm_call",
        operation="llm_call",
        start=103,
        end=105,
        attributes={"gen_ai.request.model": "gemini-2.5-pro"},
    )
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        consumer.process_telemetry("\n".join(json.dumps(record) for record in [llm, agent]))
        assert recorder.events == []
        consumer.process_telemetry(
            "\n".join(json.dumps(record) for record in [llm, agent, tool])
        )

    begins = [event for event in recorder.events if isinstance(event, SpanBeginEvent)]
    assert [(event.id, event.parent_id) for event in begins] == [
        ("tool-invoke-1", "outer-span"),
        ("agent-1", "tool-invoke-1"),
        ("llm-child", "agent-1"),
    ]



def test_native_telemetry_accepts_file_span_exporter_context_keys() -> None:
    tool = _span(
        "tool-invoke-1",
        None,
        name="tool_call",
        operation="tool_call",
        start=101,
        end=107,
        attributes={
            "gen_ai.tool.name": "invoke_agent",
            "gen_ai.tool.call_id": "invoke-agent-call-1",
        },
    )
    agent = _span(
        "agent-1",
        "tool-invoke-1",
        name="agent_call",
        operation="agent_call",
        start=102,
        end=106,
        attributes={"gen_ai.agent.name": "investigator"},
    )
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        consumer.process_telemetry("\n".join(json.dumps(record) for record in [tool, agent]))

    begins = [event for event in recorder.events if isinstance(event, SpanBeginEvent)]
    assert [(event.id, event.parent_id) for event in begins] == [
        ("tool-invoke-1", "outer-span"),
        ("agent-1", "tool-invoke-1"),
    ]


def test_native_telemetry_accepts_actual_public_parent_context() -> None:
    tool = _span(
        "tool-invoke-1",
        None,
        name="tool_call",
        operation="tool_call",
        start=101,
        end=107,
        attributes={
            "gen_ai.tool.name": "invoke_agent",
            "gen_ai.tool.call_id": "invoke-agent-call-1",
        },
    )
    agent = _span(
        "agent-1",
        "tool-invoke-1",
        name="agent_call",
        operation="agent_call",
        start=102,
        end=106,
        attributes={"gen_ai.agent.name": "investigator"},
    )
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        consumer.process_telemetry("\n".join(json.dumps(record) for record in [tool, agent]))

    begins = [event for event in recorder.events if isinstance(event, SpanBeginEvent)]
    assert [(event.id, event.parent_id) for event in begins] == [
        ("tool-invoke-1", "outer-span"),
        ("agent-1", "tool-invoke-1"),
    ]


def test_native_telemetry_rejects_conflicting_parent_context_fields() -> None:
    agent = _span(
        "agent-1",
        "tool-invoke-1",
        name="agent_call",
        operation="agent_call",
        start=102,
        end=106,
        attributes={"gen_ai.agent.name": "investigator"},
    )
    agent["_parentSpanContext"] = {"spanId": "conflicting-parent"}
    consumer = GeminiConsumer()
    recorder = _TranscriptRecorder()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
        pytest.raises(ValueError, match="conflicting parent context"),
    ):
        consumer.process_telemetry(json.dumps(agent))


def test_native_telemetry_rejects_private_parent_context_fixture_key() -> None:
    agent = _span(
        "agent-1",
        "tool-invoke-1",
        name="agent_call",
        operation="agent_call",
        start=102,
        end=106,
        attributes={"gen_ai.agent.name": "investigator"},
    )
    agent["_parentSpanContext"] = agent.pop("parentSpanContext")
    consumer = GeminiConsumer()
    recorder = _TranscriptRecorder()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
        pytest.raises(ValueError, match="parentSpanContext"),
    ):
        consumer.process_telemetry(json.dumps(agent))


def test_native_telemetry_rejects_public_context_fixture_keys() -> None:
    public_fixture = {
        "name": "llm_call",
        "spanContext": {"traceId": "trace-1", "spanId": "llm-parent"},
        "startTime": [100, 0],
        "endTime": [102, 0],
        "attributes": {
            "gen_ai.operation.name": "llm_call",
            "gen_ai.request.model": "gemini-2.5-pro",
        },
    }
    consumer = GeminiConsumer()
    recorder = _TranscriptRecorder()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
        pytest.raises(ValueError, match="_spanContext"),
    ):
        consumer.process_telemetry(json.dumps(public_fixture))


def test_native_telemetry_ignores_file_exporter_records_without_attributes() -> None:
    unrelated = {
        "_spanContext": {"traceId": "trace-1", "spanId": "non-genai-span"},
        "kind": 0,
        "name": "non-genai",
    }
    llm = _span(
        "llm-parent",
        None,
        name="llm_call",
        operation="llm_call",
        start=100,
        end=102,
        attributes={"gen_ai.request.model": "gemini-2.5-pro"},
    )
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        consumer.process_telemetry(
            "\n".join(json.dumps(record) for record in [unrelated, llm])
        )

    assert [type(event) for event in recorder.events] == [
        SpanBeginEvent,
        SpanEndEvent,
    ]



def test_native_telemetry_rejects_malformed_attributes() -> None:
    consumer = GeminiConsumer()

    with pytest.raises(ValueError, match="attributes.*object"):
        consumer.process_telemetry(json.dumps({"attributes": []}))

def test_native_compaction_accepts_file_span_exporter_context_key() -> None:
    llm = _span(
        "llm-parent",
        None,
        name="llm_call",
        operation="llm_call",
        start=100,
        end=102,
        attributes={"gen_ai.request.model": "gemini-2.5-pro"},
    )
    compaction = _compression("llm-parent", before=900, after=300, timestamp=101)
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        consumer.process_telemetry(
            "\n".join(json.dumps(record) for record in [llm, compaction])
        )

    assert [type(event) for event in recorder.events] == [
        SpanBeginEvent,
        CompactionEvent,
        SpanEndEvent,
    ]

def test_native_compaction_waits_for_its_exported_span() -> None:
    llm = _span(
        "llm-parent",
        None,
        name="llm_call",
        operation="llm_call",
        start=100,
        end=102,
        attributes={"gen_ai.request.model": "gemini-2.5-pro"},
    )
    compaction = _compression("llm-parent", before=900, after=300, timestamp=101)
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        consumer.process_telemetry(json.dumps(compaction))
        assert recorder.events == []
        consumer.process_telemetry(
            "\n".join(json.dumps(record) for record in [compaction, llm])
        )

    assert [type(event) for event in recorder.events] == [
        SpanBeginEvent,
        CompactionEvent,
        SpanEndEvent,
    ]


def test_invoke_agent_span_requires_native_tool_call_id() -> None:
    consumer = GeminiConsumer()
    missing_id = _span(
        "tool-invoke-1",
        None,
        name="tool_call",
        operation="tool_call",
        start=101,
        end=107,
        attributes={"gen_ai.tool.name": "invoke_agent"},
    )

    with pytest.raises(ValueError, match="gen_ai.tool.call_id"):
        consumer.process_telemetry(json.dumps(missing_id))


def test_native_telemetry_rejects_partial_json_instead_of_dropping_it() -> None:
    consumer = GeminiConsumer()

    with pytest.raises(ValueError, match="invalid JSON"):
        consumer.process_telemetry('{"name": "agent_call"')


class _TelemetrySandbox:
    def __init__(self, contents: str) -> None:
        self.contents = contents
        self.paths: list[str] = []

    async def read_file(self, path: str) -> str:
        self.paths.append(path)
        return self.contents


def test_final_drain_rejects_unresolved_native_parent() -> None:
    orphan = _span(
        "agent-1",
        "missing-tool-span",
        name="agent_call",
        operation="agent_call",
        start=102,
        end=106,
        attributes={"gen_ai.agent.name": "investigator"},
    )
    sandbox = _TelemetrySandbox(json.dumps(orphan))
    consumer = GeminiConsumer(
        sandbox=sandbox,
        telemetry_path="/home/agent/.gemini/inspect-swe.otel.json",
    )

    with pytest.raises(RuntimeError, match="unresolved native parent"):
        anyio.run(consumer.finalize)

    assert sandbox.paths == ["/home/agent/.gemini/inspect-swe.otel.json"]



def test_final_drain_rejects_a_known_span_parented_by_an_unsupported_operation() -> None:
    unsupported_parent = _span(
        "unsupported-parent",
        None,
        name="unsupported_operation",
        operation="unsupported_operation",
        start=100,
        end=101,
        attributes={},
    )
    child_tool = _span(
        "child-tool",
        "unsupported-parent",
        name="tool_call",
        operation="tool_call",
        start=102,
        end=103,
        attributes={
            "gen_ai.tool.name": "read_file",
            "gen_ai.tool.call_id": "read_file-1",
        },
    )
    consumer = GeminiConsumer(
        sandbox=_TelemetrySandbox(
            "\n".join(json.dumps(record) for record in [unsupported_parent, child_tool])
        ),
        telemetry_path="/home/agent/.gemini/inspect-swe.otel.json",
    )

    with pytest.raises(RuntimeError, match="unresolved native parent"):
        anyio.run(consumer.finalize)


def test_final_drain_preserves_actual_child_agent_tool_llm_hierarchy() -> None:
    recorder = _TranscriptRecorder()
    sandbox = _TelemetrySandbox(_actual_child_delegation_telemetry())
    consumer = GeminiConsumer(
        sandbox=sandbox,
        telemetry_path="/home/agent/.gemini/inspect-swe.otel.json",
    )

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        anyio.run(consumer.finalize)
        consumer.reset()

    begins = [event for event in recorder.events if isinstance(event, SpanBeginEvent)]
    assert [(event.id, event.parent_id, event.type, event.name) for event in begins] == [
        ("986e05d67a7e0a0e", "outer-span", "tool", "schedule_tool_calls"),
        ("1d65cda9b237db76", "986e05d67a7e0a0e", "tool", "invoke_agent"),
        ("625d259a22caa7e5", "1d65cda9b237db76", "agent", "gemini-cli"),
        ("5532ec2bd08cdb4e", "625d259a22caa7e5", "model", "gemini-2.5-pro"),
        ("9e3ae8370bfc7e8a", "625d259a22caa7e5", "tool", "schedule_tool_calls"),
        ("49d3b3b8304792e4", "9e3ae8370bfc7e8a", "tool", "read_file"),
    ]
    assert sandbox.paths == ["/home/agent/.gemini/inspect-swe.otel.json"]


def test_native_telemetry_can_be_drained_from_sandbox() -> None:
    recorder = _TranscriptRecorder()
    consumer = GeminiConsumer()
    sandbox = _TelemetrySandbox(_native_telemetry())

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        anyio.run(
            consumer.process_telemetry_from_sandbox,
            sandbox,
            "/home/agent/.gemini/inspect-swe.otel.json",
        )

    assert sandbox.paths == ["/home/agent/.gemini/inspect-swe.otel.json"]
    assert len(recorder.events) == 9

def test_refresh_drains_before_a_native_score_command() -> None:
    recorder = _TranscriptRecorder()
    sandbox = _TelemetrySandbox(_native_telemetry())
    consumer = GeminiConsumer(
        sandbox=sandbox,
        telemetry_path="/home/agent/.gemini/inspect-swe.otel.json",
    )

    with (
        patch.object(consumer_module, "transcript", return_value=recorder),
        patch.object(consumer_module, "current_span_id", return_value="outer-span"),
    ):
        anyio.run(consumer.refresh, "task score")

    assert sandbox.paths == ["/home/agent/.gemini/inspect-swe.otel.json"]
    assert len(recorder.events) == 9
