from collections.abc import Iterator

import pytest
from inspect_ai._util.content import ContentText
from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import Transcript
from inspect_ai.log._transcript import init_transcript, transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import span
from inspect_swe._claude_code._events.live_consumer import LiveConsumer
from inspect_swe._codex_cli._events.consumer import CodexConsumer


@pytest.fixture(autouse=True)
def _isolated_transcript() -> Iterator[None]:
    init_transcript(Transcript())
    yield
    init_transcript(Transcript())


def _model_event(
    input_messages: list[ChatMessage],
    output: ModelOutput | None = None,
    *,
    model: str = "mock/model",
) -> ModelEvent:
    return ModelEvent(
        model=model,
        input=input_messages,
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=output or ModelOutput.from_content(model, "done"),
    )


def _span_events(event_type: type[SpanBeginEvent] | type[SpanEndEvent]) -> list[str]:
    return [event.id for event in transcript().events if isinstance(event, event_type)]


def test_claude_binds_overlapping_child_call_from_native_response_and_agent_ids() -> None:
    consumer = LiveConsumer()
    spawn = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(
                        id="task-1",
                        function="Agent",
                        arguments={"prompt": "Inspect and report."},
                    ),
                    ToolCall(
                        id="task-2",
                        function="Agent",
                        arguments={"prompt": "Inspect and report."},
                    ),
                ],
            )
        ),
    )

    consumer.on_pending(spawn)
    consumer.on_complete(spawn)
    assert _span_events(SpanBeginEvent) == ["agent-task-1", "agent-task-2"]

    first_child = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                id="response-child-1",
                content="first result",
                tool_calls=[
                    ToolCall(
                        id="child-bash-1",
                        function="Bash",
                        arguments={"command": "printf child"},
                    )
                ],
            )
        ),
    )
    # The first child starts with the parent's model, then switches models
    # before it completes. The second sibling starts from the alternate
    # model too, so no model field can identify either relationship.
    first_child_followup = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(id="response-child-1-followup", content="revised")
        ),
        model="mock/alternate",
    )
    second_child = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(id="response-child-2", content="second result")
        ),
        model="mock/alternate",
    )
    for child in (first_child, first_child_followup, second_child):
        consumer.on_pending(child)
        consumer.on_complete(child)

    # Native responses may persist in either order. Their response IDs, not
    # their prompt or model names, select the child spans.
    consumer.process_jsonl_line(
        {
            "uuid": "native-second-response",
            "type": "assistant",
            "agentId": "native-agent-2",
            "message": {"id": "response-child-2"},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-first-followup",
            "type": "assistant",
            "agentId": "native-agent-1",
            "message": {"id": "response-child-1-followup"},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-first-response",
            "type": "assistant",
            "agentId": "native-agent-1",
            "message": {"id": "response-child-1"},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-first-result",
            "type": "user",
            "toolUseResult": {"agentId": "native-agent-1"},
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "task-1"}]
            },
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-second-result",
            "type": "user",
            "toolUseResult": {"agentId": "native-agent-2"},
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "task-2"}]
            },
        }
    )

    assert first_child.span_id == "agent-task-1"
    assert first_child_followup.span_id == "agent-task-1"
    assert second_child.span_id == "agent-task-2"
    assert (first_child.model, first_child_followup.model, second_child.model) == (
        "mock/model",
        "mock/alternate",
        "mock/alternate",
    )
    assert _span_events(SpanEndEvent) == ["agent-task-1", "agent-task-2"]


def test_claude_records_native_child_compaction_after_identity_binding() -> None:
    consumer = LiveConsumer()
    spawn = _model_event(
        [],
        ModelOutput.for_tool_call(
            "mock/model",
            "Agent",
            {"prompt": "Inspect the repository."},
            tool_call_id="task-1",
        ),
    )
    consumer.on_pending(spawn)
    consumer.on_complete(spawn)

    consumer.process_jsonl_line(
        {
            "uuid": "native-child-compaction",
            "type": "system",
            "subtype": "compact_boundary",
            "agentId": "native-agent-1",
            "compactMetadata": {"trigger": "manual", "preTokens": 42},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-child-result",
            "type": "user",
            "toolUseResult": {"agentId": "native-agent-1"},
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "task-1"}]
            },
        }
    )

    events = [
        event for event in transcript().events if isinstance(event, CompactionEvent)
    ]
    assert len(events) == 1
    assert events[0].span_id == "agent-task-1"
    assert events[0].metadata == {
        "trigger": "manual",
        "content": "Conversation compacted",
    }


def _codex_spawn(call_id: str, message: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={"agent_type": "explorer", "message": message},
    )


def _codex_v2_spawn(call_id: str, task_name: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={
            "task_name": task_name.removeprefix("/root/"),
            "message": "Create the child proof file.",
        },
    )


@pytest.mark.anyio
async def test_codex_does_not_merge_overlapping_child_prompts_into_parent() -> None:
    consumer = CodexConsumer()
    parent = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    _codex_spawn("spawn-1", "Inspect the repository and report."),
                    _codex_spawn("spawn-2", "Inspect the repository and report."),
                ],
            )
        ),
    )

    async with span("human_cli", type="agent", id="human-cli"):
        consumer.on_pending(parent)
        consumer.on_complete(parent)
        child = _model_event(
            [ChatMessageUser(content="Inspect the repository and report.")]
        )
        consumer.on_pending(child)
        assert child.span_id is None
        consumer.reset()


@pytest.mark.anyio
async def test_codex_binds_same_model_v2_siblings_from_native_agent_message_recipients() -> (
    None
):
    consumer = CodexConsumer()
    first_path = "/root/write_child_proof"
    second_path = "/root/write_second_proof"
    parent = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    _codex_v2_spawn("spawn-1", first_path),
                    _codex_v2_spawn("spawn-2", second_path),
                ],
            )
        ),
    )
    binding = _model_event(
        [
            ChatMessageTool(
                content='{"task_name": "/root/write_second_proof"}',
                tool_call_id="spawn-2",
                function="spawn_agent",
            ),
            ChatMessageTool(
                content='{"task_name": "/root/write_child_proof"}',
                tool_call_id="spawn-1",
                function="spawn_agent",
            ),
        ]
    )

    def child_event(task_path: str, agent_message_id: str) -> ModelEvent:
        handoff = {
            "type": "agent_message",
            "id": agent_message_id,
            "author": "/root",
            "recipient": task_path,
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Message Type: NEW_TASK\n"
                        f"Task name: {task_path}\n"
                        "Sender: /root\n"
                        "Payload:\n"
                    ),
                },
                {
                    "type": "encrypted_content",
                    "encrypted_content": "Create the child proof file.",
                },
            ],
        }
        return _model_event(
            [
                ChatMessageUser(
                    content=[
                        ContentText(
                            text="Agent message from /root:\nCreate the child proof file.",
                            internal={"agent_message": handoff},
                        )
                    ]
                )
            ]
        )

    first_child = child_event(
        first_path, "amsg_01a07523-ade2-7363-949c-08a1e028863a"
    )
    second_child = child_event(
        second_path, "amsg_01a07523-ade2-7363-949c-08a1e028863b"
    )
    completion = _model_event(
        [
            ChatMessageUser(
                content=[
                    ContentText(
                        text=(
                            "Agent message from /root/write_child_proof:\n"
                            "The child proof file is complete."
                        ),
                        internal={
                            "agent_message": {
                                "type": "agent_message",
                                "id": "amsg_01a0752c-b962-7db3-8566-816a80a74092",
                                "author": first_path,
                                "recipient": "/root",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": (
                                            "Message Type: FINAL_ANSWER\n"
                                            "Task name: /root\n"
                                            "Sender: /root/write_child_proof\n"
                                            "Payload:\n"
                                            "The child proof file is complete."
                                        ),
                                    }
                                ],
                            }
                        },
                    )
                ]
            )
        ]
    )


    async with span("human_cli", type="agent", id="human-cli"):
        consumer.on_pending(parent)
        consumer.on_complete(parent)
        consumer.on_pending(binding)
        # The native child requests may arrive in either order.
        consumer.on_pending(second_child)
        consumer.on_pending(first_child)
        consumer.on_pending(completion)

    # All three calls use mock/model. Only the exact task_name/recipient join,
    # not model or prompt similarity, selects the sibling spans.
    assert parent.model == first_child.model == second_child.model == "mock/model"
    assert first_child.span_id == "agent-spawn-1"
    assert second_child.span_id == "agent-spawn-2"
    assert [
        event_id for event_id in _span_events(SpanBeginEvent) if event_id.startswith("agent-")
    ] == ["agent-spawn-1", "agent-spawn-2"]
    assert [
        event_id for event_id in _span_events(SpanEndEvent) if event_id.startswith("agent-")
    ] == ["agent-spawn-1"]
    consumer.reset()
    assert [
        event_id for event_id in _span_events(SpanEndEvent) if event_id.startswith("agent-")
    ] == ["agent-spawn-1", "agent-spawn-2"]


def test_codex_reset_closes_unbound_child_once() -> None:
    consumer = CodexConsumer()
    spawn = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="", tool_calls=[_codex_spawn("spawn-1", "Inspect this.")]
            )
        ),
    )

    consumer.on_pending(spawn)
    consumer.on_complete(spawn)
    consumer.reset()
    consumer.reset()

    assert _span_events(SpanEndEvent) == ["agent-spawn-1"]
