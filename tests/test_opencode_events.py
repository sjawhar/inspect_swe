"""Regression coverage for OpenCode's native session recorder.

These focused fixtures use characterized OpenCode request headers and its
documented ``task`` tool-result envelope. They are intentionally not acceptance
evidence: LocalHarness drives the real CLI/bridge path separately.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log._transcript import Transcript, _transcript
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._chat_message import ChatMessage, ChatMessageTool
from inspect_ai.util._span import _current_span_id
from inspect_swe._opencode._events.consumer import OpenCodeConsumer
from inspect_swe._opencode._events.identity import (
    OpenCodeRequestIdentity,
    request_identity,
)
from inspect_swe._opencode._events.plugin import (
    AppendOnlyCompactionLog,
    compaction_events,
)

_IDENTITY_METADATA = "test_opencode_identity_headers"


def _event_identity(event: ModelEvent) -> OpenCodeRequestIdentity | None:
    """Resolve the fixture's already-allowlisted native identity context."""
    metadata = event.metadata
    assert metadata is not None
    headers = metadata[_IDENTITY_METADATA]
    assert isinstance(headers, dict)
    assert all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in headers.items()
    )
    return request_identity(headers)


@contextmanager
def _recording_context() -> Iterator[Transcript]:
    """Install one explicit outer agent span for recorder assertions."""
    captured = Transcript()
    transcript_token = _transcript.set(captured)
    span_token = _current_span_id.set("outer-agent")
    captured._event(SpanBeginEvent(id="outer-agent", name="operator", type="agent"))
    try:
        yield captured
    finally:
        _current_span_id.reset(span_token)
        _transcript.reset(transcript_token)


def _model_event(
    session_id: str | None,
    *,
    parent_session_id: str | None = None,
    input: list[ChatMessage] | None = None,
    model: str = "anthropic/claude-sonnet-4-5",
    session_header: str = "x-opencode-session",
) -> ModelEvent:
    """Build one bridge event with OpenCode's selected identity headers."""
    headers: dict[str, str] = {}
    if session_id is not None:
        headers[session_header] = session_id
    if parent_session_id is not None:
        headers["x-parent-session-id"] = parent_session_id
    return ModelEvent(
        model=model,
        input=input or [],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput.from_content(model, "response"),
        metadata={
            _IDENTITY_METADATA: headers,
            "unrelated_request_data": "must-not-reach-span-metadata",
        },
    )


def _task_result(
    session_id: str,
    state: str,
    tool_call_id: str,
) -> ChatMessageTool:
    """Build the exact outer envelope emitted by OpenCode TaskTool.renderOutput."""
    return ChatMessageTool(
        function="task",
        tool_call_id=tool_call_id,
        content=(
            f'<task id="{session_id}" state="{state}">\n'
            "<task_result>sub-agent result</task_result>\n"
            "</task>"
        ),
    )


def test_same_model_children_use_native_session_ids_and_task_links() -> None:
    """Sibling children remain separate even when their models and prompts overlap."""
    with _recording_context() as captured:
        consumer = OpenCodeConsumer(_event_identity)

        root = _model_event("root-session")
        consumer.on_pending(root)
        consumer.on_complete(root)

        first = _model_event("child-one", parent_session_id="root-session")
        second = _model_event("child-two", parent_session_id="root-session")
        consumer.on_pending(first)
        consumer.on_complete(first)
        consumer.on_pending(second)
        consumer.on_complete(second)

        complete_first = _model_event(
            "root-session",
            input=[_task_result("child-one", "completed", "call-one")],
        )
        consumer.on_pending(complete_first)
        consumer.on_complete(complete_first)

        complete_second = _model_event(
            "root-session",
            input=[_task_result("child-two", "completed", "call-two")],
        )
        consumer.on_pending(complete_second)
        consumer.on_complete(complete_second)

        beginnings = [
            event for event in captured.events if isinstance(event, SpanBeginEvent)
        ]
        children = {
            event.id: event for event in beginnings if event.id != "outer-agent"
        }
        assert set(children) == {
            "opencode-session-child-one",
            "opencode-session-child-two",
        }
        assert first.span_id == "opencode-session-child-one"
        assert second.span_id == "opencode-session-child-two"
        assert children["opencode-session-child-one"].parent_id == "outer-agent"
        assert children["opencode-session-child-two"].parent_id == "outer-agent"
        assert children["opencode-session-child-one"].metadata == {
            "opencode_session_id": "child-one",
            "opencode_parent_session_id": "root-session",
            "opencode_task_call_id": "call-one",
        }
        assert children["opencode-session-child-two"].metadata == {
            "opencode_session_id": "child-two",
            "opencode_parent_session_id": "root-session",
            "opencode_task_call_id": "call-two",
        }
        assert all(
            event.metadata is None or "Authorization" not in event.metadata
            for event in beginnings
        )
        assert [
            event.id for event in captured.events if isinstance(event, SpanEndEvent)
        ] == ["opencode-session-child-one", "opencode-session-child-two"]


def test_recorder_rejects_an_unidentified_bridge_request() -> None:
    """A missing native session ID cannot be silently attributed to the parent."""
    consumer = OpenCodeConsumer(_event_identity)
    with pytest.raises(RuntimeError, match="native session"):
        consumer.on_pending(_model_event(None))


def test_generic_provider_session_header_uses_the_outer_span() -> None:
    """The generic provider's native session header needs no model-based branch."""
    with _recording_context():
        consumer = OpenCodeConsumer(_event_identity)
        root = _model_event("generic-root", session_header="x-session-id")
        consumer.on_pending(root)
        consumer.on_complete(root)
        assert root.span_id == "outer-agent"


def test_recorder_rejects_conflicting_native_session_headers() -> None:
    """OpenCode's provider variants cannot be silently merged."""
    with pytest.raises(RuntimeError, match="conflicting native session"):
        request_identity({"x-opencode-session": "managed", "x-session-id": "generic"})


def test_append_only_plugin_log_preserves_events_appended_after_prior_drain() -> None:
    event_log = AppendOnlyCompactionLog()
    first = '{"type":"session.compacted","properties":{"sessionID":"first"}}\n'
    second = '{"type":"session.compacted","properties":{"sessionID":"second"}}\n'

    assert event_log.drain(first) == compaction_events(first)
    assert event_log.drain(first + second) == compaction_events(second)
    with pytest.raises(RuntimeError, match="append-only"):
        event_log.drain("")


def test_plugin_compaction_event_marks_the_closed_native_child_span() -> None:
    """The supported plugin preserves a child compaction after task completion."""
    with _recording_context() as captured:
        consumer = OpenCodeConsumer(_event_identity)
        root = _model_event("root")
        child = _model_event("child", parent_session_id="root")
        consumer.on_pending(root)
        consumer.on_complete(root)
        consumer.on_pending(child)
        consumer.on_complete(child)
        completed = _model_event(
            "root",
            input=[_task_result("child", "completed", "child-task")],
        )
        consumer.on_pending(completed)
        consumer.on_complete(completed)

        for event in compaction_events(
            '{"type":"session.compacted","properties":{"sessionID":"child"}}\n'
        ):
            consumer.on_native_event(event)

        compactions = [
            event for event in captured.events if isinstance(event, CompactionEvent)
        ]
        assert len(compactions) == 1
        assert compactions[0].span_id == "opencode-session-child"
        assert compactions[0].metadata == {"opencode_session_id": "child"}


def test_reset_closes_orphaned_child_before_reusing_the_consumer() -> None:
    """An interrupted child is balanced before a subsequent CLI attempt starts."""
    with _recording_context() as captured:
        consumer = OpenCodeConsumer(_event_identity)
        root = _model_event("first-root")
        child = _model_event("first-child", parent_session_id="first-root")
        consumer.on_pending(root)
        consumer.on_complete(root)
        consumer.on_pending(child)
        consumer.on_complete(child)

        consumer.reset()

        second_root = _model_event("second-root")
        consumer.on_pending(second_root)
        consumer.on_complete(second_root)
        assert second_root.span_id == "outer-agent"
        assert [
            event.id for event in captured.events if isinstance(event, SpanEndEvent)
        ] == ["opencode-session-first-child"]
