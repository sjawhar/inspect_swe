"""Codex multi-agent native-identity span reconstruction.

Multi-Agent V2 uses opaque `task_name` routing keys and encrypted spawn
messages. The consumer opens a generic child span for each `spawn_agent`, then
binds it only after the matching native spawn result returns its routing key.
Subsequent raw Responses `agent_message` recipients select the child exactly;
their human-readable text and model name are never identity signals.

Completion is likewise native: a `FINAL_ANSWER` agent-message envelope reports
the finished author's routing key. The bridge preserves these envelopes in
`ContentText.internal` on each `ChatMessageUser`.

Fixture shapes below are copied from a real gpt-5.6-sol eval log
(codex 0.147.0, 2026-08-07).
"""

import asyncio
from typing import Any

import pytest
from inspect_ai._util.content import ContentText
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_swe._codex_cli._events import consumer as consumer_module
from inspect_swe._codex_cli._events.consumer import CodexConsumer
from inspect_swe._codex_cli._events.detection import (
    agent_message_recipients,
    completed_agent_keys,
    find_spawned_agents,
    spawn_result,
)
from inspect_swe._util.centaur import reset_recorder_preserving_session_exception

# ---------------------------------------------------------------------------
# fixtures (shapes from a real codex 0.147.0 / gpt-5.6-sol run)
# ---------------------------------------------------------------------------


def _v2_spawn_call(call_id: str, task_name: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={
            "task_name": task_name,
            "fork_turns": "all",
            "message": "gAAAAABqdjXI1bVTsv-encrypted-payload",
        },
    )


def _v1_spawn_call(call_id: str, prompt: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={"agent_type": "explore", "message": prompt},
    )


def _agent_message_user(
    author: str,
    recipient: str,
    message_type: str = "MESSAGE",
    payload: str = "",
    message_id: str = "amsg_fixture",
) -> ChatMessageUser:
    """A ChatMessageUser whose internal payload preserves native identity."""
    envelope = (
        f"Message Type: {message_type}\n"
        f"Task name: {recipient}\n"
        f"Sender: {author}\n"
        f"Payload:\n{payload}"
    )
    raw_item: dict[str, Any] = {
        "type": "agent_message",
        "id": message_id,
        "author": author,
        "recipient": recipient,
        "content": [{"type": "input_text", "text": envelope}],
    }
    return ChatMessageUser(
        content=[
            ContentText(
                text=f"Agent message from {author}:\n{envelope}",
                internal={"agent_message": raw_item},
            )
        ]
    )


def _spawn_result_tool_message(call_id: str, text: str) -> ChatMessageTool:
    return ChatMessageTool(content=text, tool_call_id=call_id, function="spawn_agent")


def _model_event(
    input: list[ChatMessage],
    tool_calls: list[ToolCall] | None = None,
    *,
    metadata: dict[str, object] | None = None,
) -> ModelEvent:
    message = ChatMessageAssistant(content="ok", tool_calls=tool_calls)
    return ModelEvent(
        model="openai/gpt-5.6-sol",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_message(message),
        metadata=metadata,
    )


class _TranscriptStub:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def _event(self, event: Any) -> None:
        self.events.append(event)

    def _event_updated(self, event: Any) -> None:
        pass

    def span_begins(self) -> list[SpanBeginEvent]:
        return [e for e in self.events if isinstance(e, SpanBeginEvent)]

    def span_ends(self) -> list[SpanEndEvent]:
        return [e for e in self.events if isinstance(e, SpanEndEvent)]


def _consumer(monkeypatch: Any) -> tuple[CodexConsumer, _TranscriptStub]:
    stub = _TranscriptStub()
    monkeypatch.setattr(consumer_module, "transcript", lambda: stub)
    return CodexConsumer(), stub


# ---------------------------------------------------------------------------
# 1. detection: V2 spawn calls and results
# ---------------------------------------------------------------------------


def test_find_spawned_agents_defaults_v2_calls_to_generic_type() -> None:
    spawned = find_spawned_agents([_v2_spawn_call("call_1", "write_fizzbuzz")])
    assert len(spawned) == 1
    assert spawned[0].agent_type == "agent"
    assert spawned[0].message == "gAAAAABqdjXI1bVTsv-encrypted-payload"


def test_find_spawned_agents_keeps_v1_agent_type() -> None:
    spawned = find_spawned_agents(
        [_v1_spawn_call("call_1", "write a fizzbuzz program to /tmp/fizzbuzz.py")]
    )
    assert len(spawned) == 1
    assert spawned[0].agent_type == "explore"


def test_spawn_result_parses_v2_task_name_as_native_key() -> None:
    result = spawn_result(
        _spawn_result_tool_message("call_1", '{"task_name":"/root/write_fizzbuzz"}')
    )
    assert result is not None
    assert result.native_key == "/root/write_fizzbuzz"
    assert result.nickname is None


def test_spawn_result_parses_v1_agent_id_as_native_key() -> None:
    result = spawn_result(
        _spawn_result_tool_message(
            "call_1", '{"agent_id":"thread_abc","nickname":"Explorer"}'
        )
    )
    assert result is not None
    assert result.native_key == "thread_abc"
    assert result.nickname == "Explorer"


# ---------------------------------------------------------------------------
# 2. detection: agent_message identity and completion signals
# ---------------------------------------------------------------------------


def test_agent_message_recipients_identifies_requester() -> None:
    input: list[ChatMessage] = [
        _agent_message_user("/root", "/root/write_fizzbuzz"),
        _agent_message_user(
            "/root/write_fizzbuzz/implement_file", "/root/write_fizzbuzz"
        ),
    ]
    assert agent_message_recipients(input) == {"/root/write_fizzbuzz"}


def test_agent_message_recipients_empty_without_agent_messages() -> None:
    assert agent_message_recipients([ChatMessageUser(content="plain task")]) == set()


def test_completed_agent_keys_detects_final_native_handoff() -> None:
    input: list[ChatMessage] = [
        _agent_message_user("/root", "/root/write_primes"),
        _agent_message_user(
            "/root/write_primes", "/root", "FINAL_ANSWER", "primes written"
        ),
    ]
    assert completed_agent_keys(input) == {"/root/write_primes"}


def test_completed_agent_keys_ignores_nonfinal_agent_messages() -> None:
    input: list[ChatMessage] = [
        _agent_message_user("/root/write_primes", "/root", "MESSAGE")
    ]
    assert completed_agent_keys(input) == set()


# 3. consumer: V2 calls retain their generic native tool type
# ---------------------------------------------------------------------------


def test_consumer_uses_generic_span_type_for_v2_calls(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="spawn two agents")],
        tool_calls=[
            _v2_spawn_call("call_fb", "write_fizzbuzz"),
            _v2_spawn_call("call_pr", "write_primes"),
        ],
    )
    consumer.on_complete(parent)

    assert [event.name for event in stub.span_begins()] == ["agent", "agent"]


# 4. consumer: attribution by exact agent-message recipient
# ---------------------------------------------------------------------------


def test_consumer_attributes_subagent_call_by_native_recipient(
    monkeypatch: Any,
) -> None:
    consumer, stub = _consumer(monkeypatch)
    root_metadata: dict[str, object] = {
        "agent_bridge": {"codex": {"thread_id": "thread-root"}}
    }
    parent = _model_event(
        [ChatMessageUser(content="spawn two agents")],
        tool_calls=[
            _v2_spawn_call("call_fb", "write_fizzbuzz"),
            _v2_spawn_call("call_pr", "write_primes"),
        ],
        metadata=root_metadata,
    )
    consumer.on_pending(parent)
    consumer.on_complete(parent)
    fb_span, pr_span = (event.id for event in stub.span_begins())

    bindings = _model_event(
        [
            _spawn_result_tool_message(
                "call_fb", '{"task_name":"/root/write_fizzbuzz"}'
            ),
            _spawn_result_tool_message("call_pr", '{"task_name":"/root/write_primes"}'),
        ],
        metadata=root_metadata,
    )
    consumer.on_pending(bindings)
    consumer.on_complete(bindings)

    fb_call = _model_event([_agent_message_user("/root", "/root/write_fizzbuzz")])
    consumer.on_pending(fb_call)
    assert fb_call.span_id == fb_span

    pr_call = _model_event([_agent_message_user("/root", "/root/write_primes")])
    consumer.on_pending(pr_call)
    assert pr_call.span_id == pr_span

    # The native root identity wins over a child-like inbound message.
    parent_call = _model_event(
        [
            ChatMessageUser(content="spawn two agents"),
            _agent_message_user("/root/write_primes", "/root"),
        ],
        metadata=root_metadata,
    )
    consumer.on_pending(parent_call)
    assert parent_call.span_id is None


def test_consumer_nests_grandchild_span_under_bound_child(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_fb", "write_fizzbuzz")],
    )
    consumer.on_pending(parent)
    consumer.on_complete(parent)
    fb_span = stub.span_begins()[0].id

    binding = _model_event(
        [_spawn_result_tool_message("call_fb", '{"task_name":"/root/write_fizzbuzz"}')]
    )
    consumer.on_pending(binding)
    consumer.on_complete(binding)

    child_call = _model_event(
        [_agent_message_user("/root", "/root/write_fizzbuzz")],
        tool_calls=[_v2_spawn_call("call_impl", "implement_file")],
    )
    consumer.on_pending(child_call)
    consumer.on_complete(child_call)

    impl_begin = stub.span_begins()[1]
    assert impl_begin.name == "agent"
    assert impl_begin.parent_id == fb_span


# 5. consumer: native FINAL_ANSWER handoff closes the child's span
# ---------------------------------------------------------------------------


def test_consumer_closes_span_on_final_native_handoff(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v2_spawn_call("call_pr", "write_primes")],
    )
    consumer.on_pending(parent)
    consumer.on_complete(parent)
    pr_span = stub.span_begins()[0].id

    binding = _model_event(
        [_spawn_result_tool_message("call_pr", '{"task_name":"/root/write_primes"}')]
    )
    consumer.on_pending(binding)
    consumer.on_complete(binding)

    child_call = _model_event([_agent_message_user("/root", "/root/write_primes")])
    consumer.on_pending(child_call)
    assert child_call.span_id == pr_span

    completion = _model_event(
        [
            ChatMessageUser(content="go"),
            _agent_message_user(
                "/root/write_primes", "/root", "FINAL_ANSWER", "all primes written"
            ),
        ]
    )
    consumer.on_pending(completion)

    assert [event.id for event in stub.span_ends()] == [pr_span]


# 6. consumer: V1 prompt text cannot infer child identity
# ---------------------------------------------------------------------------


def test_consumer_does_not_infer_v1_child_from_prompt(monkeypatch: Any) -> None:
    consumer, stub = _consumer(monkeypatch)

    prompt = "write a fizzbuzz program and save it to /tmp/fizzbuzz.py"
    parent = _model_event(
        [ChatMessageUser(content="go")],
        tool_calls=[_v1_spawn_call("call_1", prompt)],
    )
    consumer.on_pending(parent)
    consumer.on_complete(parent)
    span = stub.span_begins()[0].id

    # A V1 prompt is prose, not a native identity. It must stay unscoped until
    # an exact native spawn result/recipient join proves the child span.
    child_call = _model_event([ChatMessageUser(content=prompt)])
    consumer.on_pending(child_call)
    assert child_call.span_id is None
    consumer.reset()
    assert [event.id for event in stub.span_ends()] == [span]


def test_codex_centaur_reset_preserves_cancellation_with_buffered_child(
    monkeypatch: Any,
) -> None:
    """A native child awaiting its spawn result cannot replace cancellation."""
    consumer, stub = _consumer(monkeypatch)
    parent = _model_event(
        [ChatMessageUser(content="delegate")],
        tool_calls=[_v2_spawn_call("spawn-1", "write_child_proof")],
    )
    consumer.on_pending(parent)
    consumer.on_complete(parent)

    buffered_child = _model_event(
        [_agent_message_user("/root", "/root/write_child_proof")]
    )
    consumer.on_pending(buffered_child)
    assert buffered_child not in stub.events

    with pytest.raises(asyncio.CancelledError):
        try:
            raise asyncio.CancelledError()
        finally:
            reset_recorder_preserving_session_exception(consumer.reset)

    assert [event.id for event in stub.span_ends()] == ["agent-spawn-1"]
